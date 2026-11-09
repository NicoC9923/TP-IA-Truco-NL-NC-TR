"""Descriptive paired reference evaluation; each mirror pair starts a fresh profile."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess

from engine_policies import ReactivePolicy, AdaptiveUtilityPolicy
from phase2_loader import load_phase2_policy
from phase2_promotion import RESERVED_SEEDS
from phase2_torch import Phase2Policy
from symmetric_engine import ENGINE_VERSION, play_mirror_pair
from validation_schedule import validation_schedule

OPPONENTS = {'reactive': ReactivePolicy, 'adaptive_utility': AdaptiveUtilityPolicy}


def paired_interval(values):
    if len(values) < 2:
        raise ValueError('at least two pairs required for an interval')
    mean = statistics.mean(values)
    margin = 1.96 * math.sqrt(statistics.variance(values) / len(values))
    return {'mean': mean, 'margin': margin, 'lower': mean - margin, 'upper': mean + margin}


def _chunk(job):
    path, metadata, schedule_seed, chunk = job
    import torch
    torch.set_num_threads(1)
    model = load_phase2_policy(path, seed=0, metadata=metadata).model
    rows = []
    for index, deal_seed, target in chunk:
        # Side identifies the network and reference random streams; all cells share them.
        action_seed = 2 * (schedule_seed * 1000000000 + index)
        for opponent, factory in OPPONENTS.items():
            for adapt in (True, False):
                policy = Phase2Policy(model, seed=action_seed, adapt=adapt, **metadata)
                reference = factory(action_seed + 1)
                first, second = play_mirror_pair(policy, reference, seed=deal_seed, target=target, deal_partition=0)
                score = ((first.winner == 0) + (second.winner == 1)) / 2
                rows.append((index, opponent, 'on' if adapt else 'off', score))
    return rows


def evaluate(networks, *, pairs, schedule_seed, target, workers, metadata):
    if schedule_seed in RESERVED_SEEDS or schedule_seed < 0:
        raise ValueError('reserved or negative schedule seed')
    if pairs < 2 or workers < 1 or target not in (None, 15, 30) or not networks:
        raise ValueError('invalid evaluation configuration')
    schedule = validation_schedule(schedule_seed, target, pairs)
    indexed = [(i, seed, match_target) for i, (seed, match_target) in enumerate(schedule)]
    jobs = []
    for name, path in networks.items():
        for chunk in (indexed[i::workers] for i in range(min(workers, pairs))):
            jobs.append((name, (str(path), metadata, schedule_seed, chunk)))
    scores = {name: {opponent: {state: [None] * pairs for state in ('on', 'off')}
                     for opponent in OPPONENTS} for name in networks}
    if workers == 1:
        outputs = map(_chunk, (job for _, job in jobs))
        for (name, _), rows in zip(jobs, outputs):
            for i, opponent, state, score in rows:
                scores[name][opponent][state][i] = score
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for (name, _), rows in zip(jobs, pool.map(_chunk, (job for _, job in jobs))):
                for i, opponent, state, score in rows:
                    scores[name][opponent][state][i] = score
    summaries = {}
    equal = {}
    for name in networks:
        summaries[name] = {opponent: {state: paired_interval(scores[name][opponent][state])
                                      for state in ('on', 'off')} for opponent in OPPONENTS}
        equal[name] = {state: [statistics.mean(scores[name][opponent][state][i] for opponent in OPPONENTS)
                               for i in range(pairs)] for state in ('on', 'off')}
        summaries[name]['equal_weight'] = {state: paired_interval(equal[name][state]) for state in ('on', 'off')}
    network_differences = {f'{a} minus {b}': paired_interval([x-y for x, y in zip(equal[a]['on'], equal[b]['on'])])
                           for a, b in itertools.combinations(networks, 2)}
    adaptation_differences = {name: paired_interval([x-y for x, y in zip(equal[name]['on'], equal[name]['off'])])
                              for name in networks}
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {'config': {'networks': {name: {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                                   for name, path in networks.items()},
                       'schedule_seed': schedule_seed, 'pairs': pairs, 'target': target or 'mixed',
                       'temperature': metadata['temperature'], 'profile_prior': metadata['profile_prior'],
                       'engine_version': ENGINE_VERSION, 'code_revision': revision,
                       'deal_partition': 0, 'deals': schedule},
            'scores': scores, 'summaries': summaries, 'network_differences': network_differences,
            'adaptation_differences': adaptation_differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--profile-prior', type=float, required=True)
    parser.add_argument('--pairs', type=int, required=True)
    parser.add_argument('--schedule-seed', type=int, required=True)
    parser.add_argument('--target', choices=('mixed', '15', '30'), required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    networks = {}
    for item in args.network:
        name, sep, path = item.partition('=')
        if not sep or not name or not path or name in networks:
            parser.error('network must be a unique NAME=PATH')
        networks[name] = path
    result = evaluate(networks, pairs=args.pairs, schedule_seed=args.schedule_seed,
                      target=None if args.target == 'mixed' else int(args.target), workers=args.workers,
                      metadata={'temperature': args.temperature, 'profile_prior': args.profile_prior})
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Network  Opponent  Adapt  Win rate +/- 95%')
    for name, cells in result['summaries'].items():
        for opponent, states in cells.items():
            for state, interval in states.items():
                print(f"{name}  {opponent}  {state}  {interval['mean']:.3f} +/- {interval['margin']:.3f}")

if __name__ == '__main__':
    main()
