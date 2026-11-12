"""Frozen four-mode descriptive round robin; invoke once after review."""
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'proyecto'))
from symmetric_engine import play_match

NAMES = ('reactive', 'adaptive_utility', 'phase1', 'phase2')
PAIRS = tuple(itertools.combinations(NAMES, 2))
PHASE1 = ROOT / 'champions/phase1/reigning_champion.npy'
PHASE2 = ROOT / 'champions/phase2/retired/run1-g28/reigning_champion.pt'
META = ROOT / 'champions/phase2/retired/run1-g28/reigning_champion.json'
HASHES = {PHASE1: '0b3e04fad09334f60d2d416b9c6848c7b3aabcfA6bb0d6593ac825feb32e47f7'.lower(),
          PHASE2: '05f189491a1d6ca7a4d7cebfa556a16e70d7512028e6bb224d61d87a35d4cdcd',
          META: '87b18d2cdd1414d324cc0a52f250978aa9298fa1519e1b32bc544a6fe704e75b'}
COUNT = 1000
MODELS = {}


def deal_seed(pair, index):
    return 2 * ((70001 + pair) * 10**9 + index)


def action_seed(pair, index, match, seat):
    return 2**128 + (((pair * COUNT + index) * 2 + match) * 2 + seat)


def initialize():
    import numpy as np
    import torch
    from phase2_torch import Phase2Network
    torch.set_num_threads(1)
    MODELS['phase1'] = np.load(PHASE1)
    MODELS['phase2'] = Phase2Network.load(PHASE2)
    MODELS['phase2'].eval()


def make_policy(name, seed):
    from engine_policies import ReactivePolicy, AdaptiveUtilityPolicy
    from phase1_network import NetworkPolicy
    from phase2_torch import Phase2Policy
    if name == 'reactive':
        return ReactivePolicy(seed)
    if name == 'adaptive_utility':
        return AdaptiveUtilityPolicy(seed)
    if name == 'phase1':
        return NetworkPolicy(MODELS['phase1'], seed=seed, encoder='v1')
    if name == 'phase2':
        return Phase2Policy(MODELS['phase2'], seed=seed, adapt=True, temperature=.05, profile_prior=5)
    raise ValueError(name)


def play_pair(job):
    pair, index = job
    a, b = PAIRS[pair]
    target = 15 if index % 2 == 0 else 30
    winners = []
    for match, names in enumerate(((a, b), (b, a))):
        policies = [make_policy(name, action_seed(pair, index, match, seat)) for seat, name in enumerate(names)]
        result = play_match(*policies, seed=deal_seed(pair, index), target=target,
                            starting_score=(0, 0), deal_partition=0)
        winners.append(result.winner)
    return winners


def summarize(winners):
    values = [(int(row[0] == 0) + int(row[1] == 1)) / 2 for row in winners]
    mean = statistics.mean(values)
    margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {'mean': mean, 'lower': mean - margin, 'upper': mean + margin, 'margin': margin}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def record(pair, index, winners):
    return {'matchup': pair, 'index': index, 'deal_seed': deal_seed(pair, index),
            'target': 15 if index % 2 == 0 else 30,
            'action_seeds': [[str(action_seed(pair, index, match, seat)) for seat in range(2)]
                             for match in range(2)], 'winners': winners}


def validate_records(records, *, matchup_count=6, pair_count=COUNT, complete=True):
    seen = set()
    for row in records:
        p, i = row['matchup'], row['index']
        if type(p) is not int or type(i) is not int or not 0 <= p < matchup_count or not 0 <= i < pair_count:
            raise ValueError('out-of-range pair index')
        key = (p, i)
        if key in seen:
            raise ValueError('duplicate pair')
        seen.add(key)
        winners = row['winners']
        if (not isinstance(winners, list) or len(winners) != 2 or
                any(type(x) is not int or x not in (0, 1) for x in winners) or
                row != record(p, i, winners)):
            raise ValueError('invalid pair record')
    if complete and len(seen) != matchup_count * pair_count:
        raise ValueError('incomplete pairs')
    return seen


def provenance():
    paths = sorted((ROOT / 'proyecto').glob('*.py'))
    paths += [Path(__file__).resolve(), ROOT / 'experiments/four-mode-round-robin/RULE.txt', PHASE1, PHASE2, META]
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'proyecto',
                                     'experiments/four-mode-round-robin/run_rr.py',
                                     'experiments/four-mode-round-robin/RULE.txt',
                                     'champions/phase1/reigning_champion.npy',
                                     'champions/phase2/retired/run1-g28/reigning_champion.pt',
                                     'champions/phase2/retired/run1-g28/reigning_champion.json'], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError('Sources, rule, or models are not clean: ' + dirty)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return {'git_revision': revision, 'sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths},
            'modes': list(NAMES), 'matchups': [list(x) for x in PAIRS], 'pairs_per_matchup': COUNT,
            'workers': 12, 'partition': 0, 'starting_score': [0, 0],
            'target': '15 if pair index even; otherwise 30',
            'deal_seed': '2*((70001+matchup_index)*10^9+pair_index)',
            'action_seed': '2^128+(((matchup_index*1000+pair_index)*2+match_index)*2+seat)',
            'action_seed_encoding': 'decimal strings',
            'settings': {'references': 'fixed_initial_state, fresh in-memory history',
                         'phase1': 'encoder v1, default profile and sampling, no style',
                         'phase2': 'adapt=true, temperature=0.05, profile_prior=5, weighted_beliefs=true'},
            'fresh_instances': 'each seat of each game'}


def main():
    for path, expected in HASHES.items():
        if digest(path) != expected:
            raise RuntimeError(f'Frozen identity mismatch: {path}')
    config = provenance()
    out = ROOT / 'experiments/four-mode-round-robin'
    marker = out / 'started.lock'
    with marker.open('x', encoding='utf-8') as f:
        f.write('One run started; do not retry automatically.\n')
        f.flush()
        os.fsync(f.fileno())
    atomic_json(out / 'config.json', config)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    rows = []
    with (out / 'partial.jsonl').open('x', encoding='utf-8', buffering=1) as raw:
        with ProcessPoolExecutor(max_workers=12, initializer=initialize) as pool:
            for pair, (a, b) in enumerate(PAIRS):
                futures = {pool.submit(play_pair, (pair, i)): i for i in range(COUNT)}
                current = []
                for future in as_completed(futures):
                    i = futures[future]
                    item = record(pair, i, future.result())
                    raw.write(json.dumps(item, separators=(',', ':')) + '\n')
                    raw.flush()
                    current.append(item)
                    if len(current) % 32 == 0:
                        os.fsync(raw.fileno())
                os.fsync(raw.fileno())
                validate_records(current, complete=False)
                if len(current) != COUNT:
                    raise ValueError('incomplete matchup')
                winners = [x['winners'] for x in sorted(current, key=lambda x: x['index'])]
                rows.append({'a': a, 'b': b, 'schedule_seed': 70001 + pair,
                             'interval': summarize(winners), 'pairs': len(winners)})
                atomic_json(out / 'progress.json', {'completed_matchups': rows})
                print(a, b, rows[-1]['interval'], flush=True)
    records = [json.loads(line) for line in (out / 'partial.jsonl').read_text(encoding='utf-8').splitlines()]
    validate_records(records)
    output = {'config': config, 'raw_file': 'partial.jsonl', 'raw_sha256': digest(out / 'partial.jsonl'),
              'pair_count': len(records), 'matchups': rows}
    atomic_json(out / 'results.json', output)
    with (out / 'complete.txt').open('x', encoding='utf-8') as done:
        done.write('Completed and validated all 6000 unique pairs.\n')
        done.flush()
        os.fsync(done.fileno())


if __name__ == '__main__':
    main()
