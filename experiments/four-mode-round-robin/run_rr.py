"""Frozen four-mode descriptive round robin; invoke once after review."""
import hashlib
import itertools
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'proyecto'))
from symmetric_engine import play_match

NAMES = ('reactive', 'adaptive_utility', 'phase1', 'phase2')
PAIRS = tuple(itertools.combinations(NAMES, 2))
PHASE1 = ROOT / 'champions/phase1/reigning_champion.npy'
PHASE2 = ROOT / 'champions/phase2/reigning_champion.pt'
META = ROOT / 'champions/phase2/reigning_champion.json'
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


def main():
    for path, expected in HASHES.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen identity mismatch: {path}')
    out = ROOT / 'experiments/four-mode-round-robin'
    marker = out / 'started.lock'
    with marker.open('x', encoding='utf-8') as f:
        f.write('One run started; do not retry automatically.\n')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    rows = []
    with ProcessPoolExecutor(max_workers=12, initializer=initialize) as pool:
        for pair, (a, b) in enumerate(PAIRS):
            winners = list(pool.map(play_pair, ((pair, i) for i in range(COUNT)), chunksize=8))
            rows.append({'a': a, 'b': b, 'schedule_seed': 70001 + pair,
                         'winners_by_seat': winners, 'interval': summarize(winners)})
            print(a, b, rows[-1]['interval'], flush=True)
    output = {'pairs_per_matchup': COUNT, 'workers': 12, 'matchups': rows}
    (out / 'results.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    (out / 'complete.txt').write_text('Completed all six matchups.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
