"""One-time, pre-registered phase 2 shipping acceptance test."""
import math
import statistics
from symmetric_engine import play_match


def interval(values):
    if len(values) < 2:
        raise ValueError('at least two pairs required')
    mean = statistics.mean(values)
    margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {'mean': mean, 'lower': mean - margin, 'upper': mean + margin, 'margin': margin}


def decision(direct, held_out):
    if direct['lower'] > .50 and held_out['lower'] > -.03:
        return 'ships'
    if direct['upper'] < .50 or held_out['upper'] < -.03:
        return 'regression'
    return 'inconclusive'


def make_policy(spec, seed):
    from phase1_network import NetworkPolicy
    from phase2_torch import Phase2Policy
    if spec['kind'] == 'phase1':
        return NetworkPolicy(spec['weights'], seed=seed, encoder='v1')
    return Phase2Policy(spec['model'], seed=seed, adapt=True,
                        temperature=spec['temperature'], profile_prior=spec['profile_prior'])


def play_pair(evaluated, opponent, deal_seed, target, schedule_seed, index):
    import numpy as np
    wins = 0
    for match in range(2):
        specs = (evaluated, opponent) if match == 0 else (opponent, evaluated)
        seeds = [int(np.random.SeedSequence([schedule_seed, index, match, seat]).generate_state(1)[0])
                 for seat in range(2)]
        policies = [make_policy(spec, seed) for spec, seed in zip(specs, seeds)]
        result = play_match(*policies, seed=deal_seed, target=target, starting_score=(0, 0), deal_partition=0)
        wins += result.winner == match
    return wins / 2


def _hash(path):
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metadata(path, temperature, prior):
    import json
    from pathlib import Path
    sidecar = Path(path).parent / 'reigning_champion.json'
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
        temperature, prior = meta.get('temperature'), meta.get('profile_prior')
    if (not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0
            or not isinstance(prior, (int, float)) or not math.isfinite(prior) or prior <= 0):
        raise ValueError('temperature and profile_prior required and must be positive and finite')
    return {'temperature': temperature, 'profile_prior': prior}


def _initialize_worker(config):
    import numpy as np
    import torch
    from phase2_loader import load_phase2_policy
    torch.set_num_threads(1)
    global _WORKER_CONFIG, _WORKER_NETWORKS, _WORKER_OPPONENTS
    networks = {}
    for name in ('candidate', 'baseline'):
        item = config[name]
        policy = load_phase2_policy(item['path'], seed=0, metadata=item['settings'])
        networks[name] = {'kind': 'phase2', 'model': policy.model, **item['settings']}
    opponents = {item['name']: {'kind': 'phase1', 'weights': np.load(item['path'])}
                 for item in config['held_out']}
    _WORKER_CONFIG, _WORKER_NETWORKS, _WORKER_OPPONENTS = config, networks, opponents


def _worker(indices):
    config, networks, opponents = _WORKER_CONFIG, _WORKER_NETWORKS, _WORKER_OPPONENTS
    rows = []
    for i in indices:
        direct_deal, direct_target = config['direct_schedule'][i]
        held_deal, held_target = config['held_out_schedule'][i]
        row = {'index': i, 'direct': play_pair(networks['candidate'], networks['baseline'],
                     direct_deal, direct_target, config['direct_seed'], i), 'held_out': {}}
        for name, opponent in opponents.items():
            row['held_out'][name] = {who: play_pair(networks[who], opponent, held_deal, held_target,
                                       config['held_out_seed'], i) for who in ('candidate', 'baseline')}
        rows.append(row)
    return rows


def summarize(rows, names):
    direct = interval([row['direct'] for row in rows])
    differences = {name: [row['held_out'][name]['candidate'] - row['held_out'][name]['baseline']
                          for row in rows] for name in names}
    individual = {name: interval(differences[name]) for name in names}
    clustered = interval([statistics.mean(differences[name][i] for name in names) for i in range(len(rows))])
    return {'direct': direct, 'held_out': clustered, 'held_out_individual': individual,
            'decision': decision(direct, clustered)}


def evaluate(*, candidate, baseline, held_out_manifest, direct_seed, held_out_seed, pairs, workers, output,
             candidate_temperature=None, candidate_prior=None, baseline_temperature=None, baseline_prior=None,
             resume=False):
    import json
    import multiprocessing as mp
    import os
    from pathlib import Path
    import subprocess
    from phase2_promotion import RESERVED_SEEDS
    from symmetric_engine import ENGINE_VERSION
    from validation_schedule import validation_schedule
    if (pairs < 2 or workers < 1 or direct_seed < 0 or held_out_seed < 0 or direct_seed == held_out_seed
            or direct_seed in RESERVED_SEEDS or held_out_seed in RESERVED_SEEDS):
        raise ValueError('invalid pairs, workers or schedule seeds')
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(str(output))
    if resume and not output.is_dir():
        raise ValueError('resume output directory does not exist')
    manifest = Path(held_out_manifest).resolve()
    held_manifest = json.loads(manifest.read_text(encoding='utf-8'))
    members = held_manifest['members']
    if (len(members) != 2 or len({m['name'] for m in members}) != 2
            or sorted(m['sha256'] for m in members) != sorted(held_manifest['sha256'])):
        raise ValueError('held-out manifest must identify two networks')
    held = []
    for member in members:
        path = (manifest.parent / member['path']).resolve()
        if _hash(path) != member['sha256']:
            raise ValueError('held-out hash mismatch')
        held.append({'name': member['name'], 'path': str(path), 'sha256': member['sha256'], 'encoder': 'v1'})
    config = {'candidate': {'path': str(Path(candidate).resolve()), 'sha256': _hash(candidate),
                            'settings': _metadata(candidate, candidate_temperature, candidate_prior)},
              'baseline': {'path': str(Path(baseline).resolve()), 'sha256': _hash(baseline),
                           'settings': _metadata(baseline, baseline_temperature, baseline_prior)},
              'held_out_manifest_sha256': _hash(manifest), 'held_out': held,
              'direct_seed': direct_seed, 'held_out_seed': held_out_seed, 'pairs': pairs,
              'direct_schedule': validation_schedule(direct_seed, None, pairs),
              'held_out_schedule': validation_schedule(held_out_seed, None, pairs),
              'engine_version': ENGINE_VERSION, 'deal_partition': 0, 'adapt': True,
              'tool_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    # Canonical JSON normalizes tuples, so saved configuration compares across resume.
    config = json.loads(json.dumps(config))
    if resume:
        old = json.loads((output / 'config.json').read_text(encoding='utf-8'))
        if old != config:
            raise ValueError('resume configuration mismatch')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    partial = output / 'partial.jsonl'
    rows = {}
    if partial.exists():
        lines = partial.read_bytes().splitlines(keepends=True)
        valid = []
        for line in lines:
            if not line.endswith(b'\n'):
                break
            row = json.loads(line)
            index = row['index']
            if not isinstance(index, int) or index < 0 or index >= pairs or index in rows:
                raise ValueError('invalid partial pair')
            rows[index] = row
            valid.append(line)
        if len(valid) != len(lines):
            partial.write_bytes(b''.join(valid))
    missing = [i for i in range(pairs) if i not in rows]
    jobs = [[i] for i in missing]
    if jobs:
        with mp.get_context('spawn').Pool(min(workers, len(jobs)), initializer=_initialize_worker, initargs=(config,)) as pool, partial.open('a', encoding='utf-8', newline='\n') as stream:
            for batch in pool.imap_unordered(_worker, jobs, chunksize=1):
                for row in batch:
                    stream.write(json.dumps(row, sort_keys=True) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows[row['index']] = row
    ordered = [rows[i] for i in range(pairs)]
    summaries = summarize(ordered, [m['name'] for m in held])
    result = {'config': config, 'pairs': ordered, **summaries}
    (output / 'results.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    lines = ['Comparison  Mean  Lower  Upper']
    for name, value in [('Direct', summaries['direct']), ('Held-out equal weight', summaries['held_out']),
                        *summaries['held_out_individual'].items()]:
        lines.append(f"{name}  {value['mean']:.6f}  {value['lower']:.6f}  {value['upper']:.6f}")
    lines.append('Decision: ' + summaries['decision'])
    (output / 'output.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'baseline', 'held-out-manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    for name in ('direct-seed', 'held-out-seed', 'pairs', 'workers'):
        parser.add_argument('--' + name, type=int, required=True)
    for name in ('candidate-temperature', 'candidate-prior', 'baseline-temperature', 'baseline-prior'):
        parser.add_argument('--' + name, type=float)
    parser.add_argument('--resume', action='store_true')
    evaluate(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
