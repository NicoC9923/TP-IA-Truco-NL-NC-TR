"""Compare a candidate with a baseline and a frozen network panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from phase1_network import NetworkPolicy
from phase1_trainer import _evaluate
from validation_schedule import validation_schedule

PANEL = Path(__file__).resolve().parent.parent / 'experiments/panel-v1'
BASELINE_SEED = 9001
PANEL_SEED = 9002


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _measure(a, b, pairs, seed, workers):
    jobs = [(a, b, s, t, (0, 0), None) for s, t in validation_schedule(seed, None, pairs)]
    if workers == 1:
        results = list(map(_evaluate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_evaluate, jobs, chunksize=8))
    rates = [sum(result.winner == seat for result, seat in pair) / 2 for pair in results]
    mean = float(np.mean(rates))
    margin = 1.96 * float(np.std(rates, ddof=1)) / math.sqrt(pairs)
    return {'win_rate': mean, 'margin_of_error': margin, 'pairs': pairs}


def decision(direct, candidate_rates, baseline_rates):
    if (candidate_rates and len(candidate_rates) == len(baseline_rates)
            and direct['win_rate'] - direct['margin_of_error'] > .5
            and np.mean(candidate_rates) > np.mean(baseline_rates)
            and all(c >= b - .02 for c, b in zip(candidate_rates, baseline_rates))):
        return 'promising'
    return 'no demonstrated benefit'


def evaluate(candidate, baseline, *, panel_dir=PANEL, baseline_pairs=4000, panel_pairs=1000, workers=12):
    if min(baseline_pairs, panel_pairs) < 2 or workers < 1:
        raise ValueError('pairs must be at least 2 and workers positive')
    candidate, baseline, panel_dir = Path(candidate), Path(baseline), Path(panel_dir)
    chash, bhash = digest(candidate), digest(baseline)
    a, b = np.load(candidate, allow_pickle=False), np.load(baseline, allow_pickle=False)
    NetworkPolicy(a); NetworkPolicy(b)
    members = json.loads((panel_dir / 'panel.json').read_text(encoding='utf-8'))
    if not members:
        raise ValueError('panel is empty')
    for member in members:
        if digest(panel_dir / member['file']) != member['sha256']:
            raise ValueError('panel hash mismatch: ' + member['name'])
    direct = _measure(a, b, baseline_pairs, BASELINE_SEED, workers)
    rows = []
    for member in members:
        row = {'name': member['name'], 'sha256': member['sha256']}
        if member['sha256'] == chash:
            row['skipped'] = 'candidate is panel member'
        elif member['sha256'] == bhash:
            # The baseline vs itself is ~50% by construction and would distort the panel mean and regression check.
            row['skipped'] = 'baseline is panel member'
        else:
            opponent = np.load(panel_dir / member['file'], allow_pickle=False)
            NetworkPolicy(opponent)
            row['candidate'] = _measure(a, opponent, panel_pairs, PANEL_SEED, workers)
            row['baseline'] = _measure(b, opponent, panel_pairs, PANEL_SEED, workers)
        rows.append(row)
    compared = [r for r in rows if 'candidate' in r]
    candidate_rates = [r['candidate']['win_rate'] for r in compared]
    baseline_rates = [r['baseline']['win_rate'] for r in compared]
    return {'candidate': {'path': str(candidate), 'sha256': chash},
            'baseline': {'path': str(baseline), 'sha256': bhash},
            'schedule_seeds': {'baseline': BASELINE_SEED, 'panel': PANEL_SEED},
            'candidate_vs_baseline': direct, 'panel': rows,
            'candidate_panel_mean': float(np.mean(candidate_rates)) if compared else None,
            'baseline_panel_mean': float(np.mean(baseline_rates)) if compared else None,
            'decision': decision(direct, candidate_rates, baseline_rates)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--panel-dir', default=str(PANEL))
    parser.add_argument('--baseline-pairs', type=int, default=4000)
    parser.add_argument('--panel-pairs', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if min(args.baseline_pairs, args.panel_pairs) < 2 or args.workers < 1:
        parser.error('pairs must be at least 2 and workers positive')
    report = evaluate(args.candidate, args.baseline, panel_dir=args.panel_dir, baseline_pairs=args.baseline_pairs, panel_pairs=args.panel_pairs, workers=args.workers)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
