"""Measure tournament parent selection on an independent league."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from phase1_trainer import league_schedule, league_scores

SEED_BASE = 10**23


def schedule(size, pairs, config, hall, reigning, generation, stream):
    rng = np.random.default_rng(438187 + stream)
    result = league_schedule(rng, generation, size, pairs, config[3], config[2], hall, reigning, config[14])
    for i, item in enumerate(result):
        item['seed'] = 2 * (SEED_BASE + stream * 1000000 + i) + 1
    return result


def summarize(first, independent, tournaments=200, bootstrap=2000, seed=60419):
    first, independent = np.asarray(first, dtype=float), np.asarray(independent, dtype=float)
    if first.ndim != 1 or first.shape != independent.shape or len(first) < 2 or tournaments < 1 or bootstrap < 1:
        raise ValueError('scores must have matching lengths of at least two and counts must be positive')
    rng = np.random.default_rng(seed)
    n = len(first)
    def sample(pool):
        candidates = pool[rng.integers(n, size=(tournaments, 3))]
        scores = independent[candidates]
        winners = np.argmax(first[candidates], axis=1)
        def metrics(position):
            chosen = scores[np.arange(tournaments), position]
            delta = chosen - (scores.sum(axis=1) - chosen) / 2
            # Split the credit for ties, including repeated candidates.
            best = (chosen == scores.max(axis=1)) / (scores == scores.max(axis=1, keepdims=True)).sum(axis=1)
            return float(delta.mean()), float(best.mean())
        return metrics(winners), metrics(np.zeros(tournaments, dtype=int))
    observed = sample(np.arange(n))
    draws = np.empty((bootstrap, 2))
    for b in range(bootstrap):
        selection, random = sample(rng.integers(n, size=n))
        draws[b] = selection[0], random[0]
    spread = float(np.std(independent, ddof=1))
    result = {}
    for i, name in enumerate(('selection', 'random')):
        result[name] = {'advantage': observed[i][0], 'ci95': np.percentile(draws[:, i], [2.5, 97.5]).tolist(),
                        'best_fraction': observed[i][1], 'advantage_over_spread': observed[i][0] / spread if spread else None}
    result.update(population_spread=spread, tournaments=tournaments, bootstrap_replicates=bootstrap,
                  bootstrap_method='resample individuals with replacement, then redraw tournaments; percentile interval',
                  best_tie_rule='split credit among tied candidate slots')
    return result


def diagnose(checkpoint, tournaments=200, measurement_pairs=64, workers=12, bootstrap=2000):
    start = perf_counter()
    if min(tournaments, measurement_pairs, workers, bootstrap) < 1 or measurement_pairs >= 1000000:
        raise ValueError('counts must be positive and measurement pairs below 1000000')
    with np.load(checkpoint, allow_pickle=False) as data:
        population, hall, reigning = data['population'], list(data['hall']), data['reigning']
        state = json.loads(str(data['state']))
    if state.get('encoder', 'v1') != 'v1':
        raise ValueError('This diagnostic is v1-only')
    config = state['config']
    if len(population) < 2 or not 0 < config[1] < 1000000:
        raise ValueError('population must have at least two members and valid first-pass pair count')
    generation = state['generation'] + 1
    schedules = [schedule(len(population), p, config, hall, reigning, generation, stream)
                 for stream, p in enumerate((config[1], measurement_pairs))]
    scores = [league_scores(population, s, hall, reigning, workers, config[13], config[11], config[12])[0] for s in schedules]
    report = summarize(*scores, tournaments=tournaments, bootstrap=bootstrap)
    report.update(checkpoint=str(checkpoint), checkpoint_generation=state['generation'], evaluated_generation=generation,
                  population_size=len(population), first_pass_pairs=config[1], measurement_pairs=measurement_pairs,
                  first_pass_scores=scores[0].tolist(), independent_scores=scores[1].tolist(), schedules=schedules,
                  seed_namespace='odd 2*(10^23 + stream*1000000 + pair)+1', seconds=perf_counter()-start)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--tournaments', type=int, default=200)
    parser.add_argument('--measurement-pairs', type=int, default=64)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    if min(args.tournaments, args.measurement_pairs, args.workers, args.bootstrap) < 1 or args.measurement_pairs >= 1000000:
        parser.error('counts must be positive and measurement pairs below 1000000')
    report = diagnose(args.checkpoint, args.tournaments, args.measurement_pairs, args.workers, args.bootstrap)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
