"""Minimal phase 1 genetic training using generation peers only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from phase1_network import GENOME_SIZE, NetworkPolicy
from symmetric_engine import play_match

def fitness(results):
    """Score (result, seat) pairs using match wins and normalized point difference."""
    return sum((result.winner == seat) + .1 * (result.points[seat] - result.points[1-seat]) / max(result.points) for result, seat in results) / len(results)

def _evaluate(job):
    genome, opponent, seed, target, starting_score = job
    a = NetworkPolicy(genome, seed=seed * 2)
    b = NetworkPolicy(opponent, seed=seed * 2 + 1)
    first = play_match(a, b, seed=seed, target=target, starting_score=starting_score)
    second = play_match(b, a, seed=seed, target=target, starting_score=starting_score)
    return [(first, 0), (second, 1)]

def _behavior(results):
    own = [d for result, seat in results for d in result.decisions if d['player'] == seat]
    hands = len({(j, d['hand_index']) for j, (result, seat) in enumerate(results) for d in result.decisions if d['player'] == seat})
    calls = [d for d in own if d['bluff'] is not None]
    opportunities = [d for d in own if d['observation']['pending'] and d['observation']['pending']['kind'] == 'truco']
    return {'envido': sum(d['action'] in ('envido', 'real_envido', 'falta_envido') for d in own) / max(hands, 1), 'bluff': sum(d['bluff'] for d in calls) / max(len(calls), 1), 'truco_acceptance': sum(d['action'] in ('quiero', 'retruco', 'vale_cuatro') for d in opportunities) / max(len(opportunities), 1)}

def train(directory, population_size=4, generations=1, pairs=1, workers=2, target=None, seed=0):
    if min(population_size - 1, generations, pairs, workers) < 1 or (target is not None and target < 1):
        raise ValueError('Invalid training configuration')
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    population = rng.normal(0, .1, (population_size, GENOME_SIZE))
    with (output / 'generations.jsonl').open('w', encoding='utf-8') as log:
        for generation in range(1, generations + 1):
            start = perf_counter()
            jobs = []
            for i in range(population_size):
                for p in range(pairs):
                    opponent = (i + 1 + p % (population_size - 1)) % population_size
                    match_target = target if target is not None else int(rng.choice((15, 30)))
                    starting_score = tuple(int(x) for x in rng.integers(match_target, size=2)) if rng.random() < .3 else (0, 0)
                    jobs.append((population[i], population[opponent], seed + generation * 100000 + p, match_target, starting_score))
            if workers == 1:
                evaluated = list(map(_evaluate, jobs))
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    evaluated = list(pool.map(_evaluate, jobs))
            results = [sum(evaluated[i*pairs:(i+1)*pairs], []) for i in range(population_size)]
            scores = np.array([fitness(group) for group in results])
            champion = int(np.argmax(scores))
            name = f'champion_{generation:04d}.npy'
            NetworkPolicy(population[champion]).save(output / name)
            group = results[champion]
            first, seat = group[0]
            hand = [d for d in first.decisions if d['hand_index'] == 0]
            entry = {'generation': generation, 'phase': 1, 'fitness': {'max': float(scores.max()), 'average': float(scores.mean()), 'min': float(scores.min())}, 'behavior': _behavior(group), 'diversity': float(np.std(population, axis=0).mean()), 'example_hand': hand, 'champion': name}
            log.write(json.dumps(entry) + '\n')
            log.flush()
            print(f'gen {generation} · best {scores.max():.3f} · avg {scores.mean():.3f} · {perf_counter()-start:.2f} s', flush=True)
            next_population = [population[champion].copy()]
            while len(next_population) < population_size:
                parents = []
                for _ in range(2):
                    candidates = rng.integers(population_size, size=3)
                    parents.append(population[candidates[np.argmax(scores[candidates])]])
                child = np.where(rng.random(GENOME_SIZE) < .5, parents[0], parents[1])
                child += rng.normal(0, .05, GENOME_SIZE) * (rng.random(GENOME_SIZE) < .1)
                next_population.append(child)
            population = np.array(next_population)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='training_output/phase1')
    parser.add_argument('--population', type=int, default=4)
    parser.add_argument('--generations', type=int, default=1)
    parser.add_argument('--pairs', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--target', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train(args.output, args.population, args.generations, args.pairs, args.workers, args.target, args.seed)

if __name__ == '__main__':
    main()



