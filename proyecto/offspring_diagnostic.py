"""Compare offspring against parents on a fixed mirrored schedule."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np
from phase1_trainer import _evaluate

SEED_BASE = 10**18

def variants(a, b, rng):
    cross = np.where(rng.random(len(a)) < .5, a, b)
    mutated = cross + rng.normal(0, .05, len(a)) * (rng.random(len(a)) < .1)
    result = {'parent': a.copy(), 'crossover_mutation': mutated, 'crossover_only': cross}
    for sigma in (.025, .05, .10):
        result[f'mutation_{sigma:.3f}'] = a + rng.normal(0, sigma, len(a)) * (rng.random(len(a)) < .1)
    return result

def summary(values):
    a = np.asarray(values)
    margin = 1.96 * float(np.std(a, ddof=1)) / math.sqrt(len(a))
    return {'mean': float(np.mean(a)), 'ci95': [float(np.mean(a)-margin), float(np.mean(a)+margin)], 'fraction_beating_parent': float(np.mean(a > 0)), 'spread': float(np.std(a, ddof=1))}

def diagnose(checkpoint, k=32, pairs=32, workers=12):
    start = perf_counter()
    with np.load(checkpoint, allow_pickle=False) as data:
        population, hall, reigning = data['population'], list(data['hall']), data['reigning']
        state = json.loads(str(data['state']))
    if len(population) < 6 or k < 2 or pairs < 2 or workers < 1:
        raise ValueError('population must have at least six members, k and pairs at least two, workers positive')
    rng = np.random.default_rng(720394)
    reserved = rng.choice(len(population), 4, replace=False)
    candidates = np.array([i for i in range(len(population)) if i not in reserved])
    indices = rng.choice(candidates, (k, 2), replace=True)
    groups = [variants(population[a], population[b], rng) for a, b in indices]
    opponents = [reigning] + hall + [population[i] for i in reserved]
    jobs = []
    for group in groups:
        for genome in group.values():
            for p in range(pairs):
                seed = 2 * (SEED_BASE + p) + 1
                jobs.append((genome, opponents[p % len(opponents)], seed, 15 if p % 2 == 0 else 30, (0, 0), None))
    if workers == 1:
        results = list(map(_evaluate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_evaluate, jobs, chunksize=8))
    rates = np.array([sum(r.winner == seat for r, seat in result)/2 for result in results]).reshape(k, len(groups[0]), pairs).mean(axis=2)
    names = list(groups[0]); deltas = rates[:, 1:] - rates[:, [0]]
    summaries = {name: summary(deltas[:, i]) for i, name in enumerate(names[1:])}
    difference = deltas[:, names[1:].index('crossover_mutation')] - deltas[:, names[1:].index('mutation_0.050')]
    return {'checkpoint': str(checkpoint), 'generation': state['generation'], 'k': k, 'pairs': pairs, 'parent_sampling': 'uniform with replacement; checkpoint has no scores', 'parent_indices': indices.tolist(), 'reserved_opponent_indices': reserved.tolist(), 'seed_namespace': 'odd 2*(10^18 + pair)+1', 'variants': summaries, 'crossover_minus_mutation': summary(difference), 'pair_rates': rates.tolist(), 'seconds': perf_counter()-start}

def interpretation(offspring, behavior):
    m = offspring['variants']['mutation_0.050']
    small = offspring['variants']['mutation_0.025']
    diff = offspring['crossover_minus_mutation']
    agreement = behavior['variants']['mutation_0.050']['mean_argmax_agreement']
    pop = behavior['population']['mean_pairwise_tv']
    panel = behavior['panel']['mean_pairwise_tv']
    scores = {}
    def ratio(value, width):
        return value / width if width else float('inf')
    if diff['ci95'][1] < 0:
        scores['crossover damage: proposal 4'] = ratio(-diff['mean'], (diff['ci95'][1]-diff['ci95'][0])/2)
    if agreement > .98 and m['ci95'][0] <= 0 <= m['ci95'][1] and m['spread'] < .044 * math.sqrt(64/offspring['pairs']):
        scores['mutation too weak: larger sigma'] = (agreement-.98)/.02
    if m['ci95'][1] < 0 and m['fraction_beating_parent'] < .4 and small['mean'] >= m['mean']:
        scores['mutation too destructive: smaller sigma'] = ratio(-m['mean'], (m['ci95'][1]-m['ci95'][0])/2)
    if panel > 0 and pop < panel/3:
        scores['low behavioral diversity: proposal 6'] = 1-pop/(panel/3)
    return {'fired_rules': scores, 'recommendation': max(scores, key=scores.get) if scores else 'no operator diagnosis; consider inputs (proposal 7)'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint'); parser.add_argument('--output', required=True)
    parser.add_argument('--k', type=int, default=32); parser.add_argument('--pairs', type=int, default=32); parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if min(args.k, args.pairs) < 2 or args.workers < 1 or args.pairs >= 1000000: parser.error('k and pairs must be at least two, pairs below 1000000, workers positive')
    Path(args.output).write_text(json.dumps(diagnose(args.checkpoint, args.k, args.pairs, args.workers), indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__': main()
