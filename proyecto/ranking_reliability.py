"""Measure ranking reliability on a frozen phase 1 checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from phase1_trainer import league_schedule, league_scores


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2
        i = j
    return result


def spearman(a, b):
    x, y = ranks(a), ranks(b)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(measurements, reference):
    result = {}
    count = min(8, len(reference))
    for pairs, splits in measurements.items():
        correlations, overlaps, reference_scores = [], [], []
        for a, b in splits:
            correlations.append(spearman(a, b))
            tops = [np.argsort(-np.asarray(scores), kind='stable')[:count] for scores in (a, b)]
            overlaps.append(len(set(tops[0]) & set(tops[1])))
            reference_scores.append(float(np.mean([np.mean(np.asarray(reference)[top]) for top in tops])))
        result[str(pairs)] = {'correlations': correlations,
            'mean_correlation': float(np.mean(correlations)) if all(c is not None for c in correlations) else None,
            'top_8_overlap': overlaps, 'mean_reference_score_top_8': float(np.mean(reference_scores)),
            'reference_scores_top_8': reference_scores}
    a, b = result.get('16'), result.get('32')
    proceed = bool(a and b and a['mean_correlation'] is not None and b['mean_correlation'] is not None
                   and b['mean_correlation'] - a['mean_correlation'] >= .10
                   and b['mean_reference_score_top_8'] > a['mean_reference_score_top_8'])
    return {'pairs': result, 'verdict': 'proceed' if proceed else 'do not proceed'}


def diagnostic_schedule(population_size, pairs, config, hall, reigning, stream):
    rng = np.random.default_rng(932874 + stream)
    schedule = league_schedule(rng, 1, population_size, pairs, 0, config[2], hall, reigning, config[14])
    for p, item in enumerate(schedule):
        item['seed'] = 2 * (10**17 + stream * 1000000 + p) + 1
    return schedule


def diagnose(checkpoint, pairs=(16, 32), splits=3, reference_pairs=64, workers=12):
    with np.load(checkpoint, allow_pickle=False) as data:
        population, hall, reigning = data['population'], list(data['hall']), data['reigning']
        state = json.loads(str(data['state']))
    if state.get('encoder', 'v1') != 'v1':
        raise ValueError('This diagnostic is v1-only')
    config = state['config']
    stream = 0
    def measure(n):
        nonlocal stream
        schedule = diagnostic_schedule(len(population), n, config, hall, reigning, stream)
        stream += 1
        return league_scores(population, schedule, hall, reigning, workers, config[13], config[11], config[12])[0]
    reference = measure(reference_pairs)
    measurements = {p: [(measure(p), measure(p)) for _ in range(splits)] for p in pairs}
    report = summarize(measurements, reference)
    report.update(checkpoint=str(checkpoint), generation=state['generation'], splits=splits,
                  reference_pairs=reference_pairs, population_size=len(population), reference_scores=reference.tolist(),
                  seed_namespace='odd seeds 2*(10^17 + stream*1000000 + pair)+1')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pairs', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--splits', type=int, default=3)
    parser.add_argument('--reference-pairs', type=int, default=64)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if min(args.pairs + [args.splits, args.reference_pairs, args.workers]) < 1 or len(set(args.pairs)) != len(args.pairs) or max(args.pairs + [args.reference_pairs]) >= 1000000:
        parser.error('counts must be positive, pair counts unique and below 1000000')
    report = diagnose(args.checkpoint, args.pairs, args.splits, args.reference_pairs, args.workers)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
