"""Phase 1 genetic training with a styled league and promotion checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import shutil
from pathlib import Path
from time import perf_counter
import numpy as np
from phase1_network import GENOME_SIZE, NetworkPolicy
from symmetric_engine import play_match
from validation_schedule import validation_schedule

def fitness(results):
    """Score (result, seat) pairs using match wins and normalized point difference."""
    return sum((result.winner == seat) + .1 * (result.points[seat] - result.points[1-seat]) / max(result.points) for result, seat in results) / len(results)

def _evaluate(job):
    genome, opponent, seed, target, starting_score, style = job
    a = NetworkPolicy(genome, seed=seed * 2)
    b = NetworkPolicy(opponent, seed=seed * 2 + 1, style=style)
    first = play_match(a, b, seed=seed, target=target, starting_score=starting_score, deal_partition=seed % 2)
    second = play_match(b, a, seed=seed, target=target, starting_score=starting_score[::-1], deal_partition=seed % 2)
    return [(first, 0), (second, 1)]

def _behavior(results):
    own = [d for result, seat in results for d in result.decisions if d['player'] == seat]
    hands = len({(j, d['hand_index']) for j, (result, seat) in enumerate(results) for d in result.decisions if d['player'] == seat})
    calls = [d for d in own if d['bluff'] is not None]
    opportunities = [d for d in own if d['observation']['pending'] and d['observation']['pending']['kind'] == 'truco']
    return {'envido': sum(d['action'] in ('envido', 'real_envido', 'falta_envido') for d in own) / max(hands, 1), 'bluff': sum(d['bluff'] for d in calls) / max(len(calls), 1), 'truco_acceptance': sum(d['action'] in ('quiero', 'retruco', 'vale_cuatro') for d in opportunities) / max(len(opportunities), 1)}

def train(directory, population_size=64, generations=1, pairs=1, workers=16, target=None, seed=0, validation_pairs=1000, promotion_threshold=.55, patience=20, resume=None):
    if min(population_size - 1, generations, pairs, workers, validation_pairs, patience) < 1 or (target is not None and target < 1):
        raise ValueError('Invalid training configuration')
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_config.json').write_text(json.dumps({'seed': seed, 'target': target}), encoding='utf-8')
    rng = np.random.default_rng(seed)
    population = rng.normal(0, .1, (population_size, GENOME_SIZE))
    hall, reigning, reigning_generation, stale, first_generation = [], None, None, 0, 1
    if resume:
        with np.load(resume, allow_pickle=False) as data:
            population, reigning, hall = data['population'], data['reigning'], list(data['hall'])
            state = json.loads(str(data['state']))
        if state['config'] != [population_size, pairs, target, seed, validation_pairs, promotion_threshold, patience]:
            raise ValueError('Checkpoint configuration does not match')
        rng.bit_generator.state = state['rng']
        reigning_generation, stale, first_generation = state['reigning_generation'], state['stale'], state['generation'] + 1
        log_path = output / 'generations.jsonl'
        if log_path.exists():
            lines = log_path.read_text(encoding='utf-8').splitlines()
            kept = [line for line in lines if json.loads(line)['generation'] <= state['generation']]
            log_path.write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
        if state['ended']:
            return
    with (output / 'generations.jsonl').open('a' if resume else 'w', encoding='utf-8') as log:
        for generation in range(first_generation, generations + 1):
            start = perf_counter()
            jobs = []
            styles = []
            for i in range(population_size):
                for p in range(pairs):
                    style = (None, None, None, None, 'bluffer', 'cautious', 'aggressive', 'calculating')[(i * pairs + p + generation - 1) % 8]
                    opponent_index = (i + 1 + int(rng.integers(population_size - 1))) % population_size
                    opponent = population[opponent_index]
                    if style is None and hall and rng.random() < .5:
                        opponent = hall[int(rng.integers(len(hall)))]
                    styles.append(style or 'unbiased')
                    match_target = target if target is not None else int(rng.choice((15, 30)))
                    starting_score = tuple(int(x) for x in rng.integers(match_target, size=2)) if rng.random() < .3 else (0, 0)
                    jobs.append((population[i], opponent, 2 * (seed * 1000000000 + generation * population_size * pairs + i * pairs + p) + 1, match_target, starting_score, style))
            if workers == 1:
                evaluated = list(map(_evaluate, jobs))
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    evaluated = list(pool.map(_evaluate, jobs))
            league_seconds = perf_counter() - start
            results = [sum(evaluated[i*pairs:(i+1)*pairs], []) for i in range(population_size)]
            scores = np.array([fitness(group) for group in results])
            champion = int(np.argmax(scores))
            name = f'champion_{generation:04d}.npy'
            NetworkPolicy(population[champion]).save(output / name)
            genome = population[champion].copy()
            check, promoted = None, False
            check_seconds = 0.0
            if reigning is None:
                reigning, reigning_generation = genome.copy(), generation
            else:
                check_start = perf_counter()
                jobs = [(genome, reigning, deal_seed, match_target, (0, 0), None) for deal_seed, match_target in validation_schedule(seed, target, validation_pairs)]
                if workers == 1:
                    checks = list(map(_evaluate, jobs))
                else:
                    with ProcessPoolExecutor(max_workers=workers) as pool:
                        checks = list(pool.map(_evaluate, jobs))
                check_seconds = perf_counter() - check_start
                pair_rates = [sum(result.winner == seat for result, seat in pair) / 2 for pair in checks]
                rate = float(np.mean(pair_rates))
                error = 1.96 * float(np.std(pair_rates, ddof=1)) / math.sqrt(validation_pairs) if validation_pairs > 1 else 0.0
                check = {'win_rate': rate, 'margin_of_error': error, 'pairs': validation_pairs}
                promoted = rate >= promotion_threshold
                if promoted:
                    reigning, reigning_generation, stale = genome.copy(), generation, 0
                    hall.append(genome.copy())
                else:
                    stale += 1
            ended = stale >= patience
            NetworkPolicy(reigning).save(output / 'reigning_champion.npy')
            group = results[champion]
            first, seat = group[0]
            hand = [d for d in first.decisions if d['hand_index'] == 0]
            entry = {'generation': generation, 'phase': 1, 'fitness': {'max': float(scores.max()), 'average': float(scores.mean()), 'min': float(scores.min())}, 'behavior': _behavior(group), 'diversity': float(np.std(population, axis=0).mean()), 'example_hand': hand, 'champion': name}
            entry.update(promotion_check=check, promoted=promoted, reigning_champion=reigning_generation, phase_ended=ended, league=styles)
            log.write(json.dumps(entry) + '\n')
            log.flush()
            status = 'first champion' if check is None else f"{'promoted' if promoted else 'not promoted'} {check['win_rate']:.0%} +/-{check['margin_of_error']*100:.0f}"
            print(f'gen {generation} | best {scores.max():.3f} | avg {scores.mean():.3f} | {status} | {perf_counter()-start:.2f} s | league {league_seconds:.2f} s | check {check_seconds:.2f} s', flush=True)
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
            if generation % 10 == 0:
                state = dict(generation=generation, config=[population_size, pairs, target, seed, validation_pairs, promotion_threshold, patience], rng=rng.bit_generator.state, reigning_generation=reigning_generation, stale=stale, ended=ended)
                np.savez_compressed(output / f'checkpoint_{generation:04d}.npz', population=population, reigning=reigning, hall=np.asarray(hall).reshape(-1, GENOME_SIZE), state=json.dumps(state))
            if ended:
                break

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='training_output/phase1')
    parser.add_argument('--population', type=int, default=64)
    parser.add_argument('--generations', type=int, default=1)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--target', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validation-pairs', type=int, default=1000)
    parser.add_argument('--promotion-threshold', type=float, default=.55)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--resume')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.publish:
        destination = Path(__file__).resolve().parent.parent / 'champions' / 'phase1'
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('reigning_champion.npy', 'generations.jsonl', 'reference_evaluation.jsonl'):
            source = Path(args.output) / name
            if name != 'reference_evaluation.jsonl' or source.exists():
                shutil.copy2(source, destination / name)
    else:
        train(args.output, args.population, args.generations, args.pairs, args.workers, args.target, args.seed, args.validation_pairs, args.promotion_threshold, args.patience, args.resume)

if __name__ == '__main__':
    main()



