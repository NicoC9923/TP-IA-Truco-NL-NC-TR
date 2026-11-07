"""Phase 1 genetic training with a styled league and promotion checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import shutil
from pathlib import Path
from time import perf_counter
import numpy as np
from phase1_network import GENOME_SIZE, INPUTS, HIDDEN, NetworkPolicy
from symmetric_engine import play_match
from validation_schedule import validation_schedule

def fitness(results, mode='classic', points_weight=.1):
    """Score wins plus point difference, or a weighted mixture of the two."""
    win_weight = 1 - points_weight if mode == 'mixed' else 1
    return sum(win_weight * (result.winner == seat) + points_weight * (result.points[seat] - result.points[1-seat]) / max(result.points) for result, seat in results) / len(results)

def _evaluate(job):
    genome, opponent, seed, target, starting_score, style = job[:6]
    strength = job[6] if len(job) > 6 else 1.0
    # Legacy short jobs represent v1 policies on both sides.
    encoder_a, encoder_b = job[7:9] if len(job) >= 9 else ('v1', 'v1')
    a = NetworkPolicy(genome, seed=seed * 2, encoder=encoder_a)
    b = NetworkPolicy(opponent, seed=seed * 2 + 1, style=style, style_strength=strength, encoder=encoder_b)
    first = play_match(a, b, seed=seed, target=target, starting_score=starting_score, deal_partition=seed % 2)
    second = play_match(b, a, seed=seed, target=target, starting_score=starting_score[::-1], deal_partition=seed % 2)
    return [(first, 0), (second, 1)]

def _behavior(results):
    own = [d for result, seat in results for d in result.decisions if d['player'] == seat]
    hands = len({(j, d['hand_index']) for j, (result, seat) in enumerate(results) for d in result.decisions if d['player'] == seat})
    calls = [d for d in own if d['bluff'] is not None]
    opportunities = [d for d in own if d['observation']['pending'] and d['observation']['pending']['kind'] == 'truco']
    return {'envido': sum(d['action'] in ('envido', 'real_envido', 'falta_envido') for d in own) / max(hands, 1), 'bluff': sum(d['bluff'] for d in calls) / max(len(calls), 1), 'truco_acceptance': sum(d['action'] in ('quiero', 'retruco', 'vale_cuatro') for d in opportunities) / max(len(opportunities), 1)}

def league_schedule(rng, generation, population_size, pairs, seed, target, hall, reigning, styles_after_first_promotion):
    schedule = []
    for p in range(pairs):
        style = None if p % 2 == 0 else ('bluffer', 'cautious', 'aggressive', 'calculating')[(p // 2 + 2 * (generation - 1)) % 4]
        if styles_after_first_promotion and not hall:
            style = None
        opponent_index = int(rng.integers(population_size))
        opponent_kind = 'population'
        if p == 0 and reigning is not None:
            opponent_kind = 'reigning'
            style = None
        elif style is None and hall and rng.random() < .5:
            opponent_kind = 'hall'
            opponent_index = int(rng.integers(len(hall)))
        match_target = target if target is not None else int(rng.choice((15, 30)))
        starting_score = tuple(int(x) for x in rng.integers(match_target, size=2)) if rng.random() < .3 else (0, 0)
        deal_seed = 2 * (seed * 1000000000 + generation * population_size * pairs + p) + 1
        schedule.append(dict(opponent=opponent_kind, index=opponent_index, style=style,
                             seed=deal_seed, target=match_target, starting_score=starting_score))
    return schedule

def league_scores(population, schedule, hall, reigning, workers, style_strength, fitness_mode, points_weight, encoder='v1'):
    population_size = len(population)
    pairs = len(schedule)
    jobs = []
    for i in range(population_size):
        for item in schedule:
            if item['opponent'] == 'reigning':
                opponent = reigning
            elif item['opponent'] == 'hall':
                opponent = hall[item['index']]
            else:
                index = item['index']
                opponent = population[(index + 1) % population_size if index == i else index]
            jobs.append((population[i], opponent, item['seed'], item['target'], item['starting_score'], item['style'], style_strength, encoder, encoder))
    if workers == 1:
        evaluated = list(map(_evaluate, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evaluated = list(pool.map(_evaluate, jobs))
    results = [sum(evaluated[i*pairs:(i+1)*pairs], []) for i in range(population_size)]
    scores = np.array([fitness(group, fitness_mode, points_weight) for group in results])
    return scores, results

def train(directory, population_size=64, generations=1, pairs=1, workers=16, target=None, seed=0, validation_pairs=1000, promotion_threshold=.55, patience=50, resume=None, check_batch=200, finalists=8, finalist_pairs_multiplier=4, output_init_sigma=.3, fitness_mode="classic", points_weight=.1, style_strength=.5, styles_after_first_promotion=True, early_stop=None, extend_patience=None, encoder='v1'):
    if encoder not in ('v1', 'v2'):
        raise ValueError('Unknown encoder')
    if extend_patience is not None and not resume:
        raise ValueError("--extend-patience requires --resume")
    extension_from = None
    if early_stop not in (None, 'reject-only', 'both'):
        raise ValueError('Invalid early-stop mode')
    if not math.isfinite(style_strength) or style_strength < 0 or fitness_mode not in ('classic', 'mixed') or not math.isfinite(points_weight) or points_weight < 0 or (fitness_mode == 'mixed' and points_weight > 1) or output_init_sigma < 0 or finalists < 0 or finalist_pairs_multiplier < 1 or check_batch < 0 or min(population_size - 1, generations, pairs, workers, validation_pairs, patience) < 1 or (target is not None and target < 1):
        raise ValueError('Invalid training configuration')
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    population = rng.normal(0, .1, (population_size, GENOME_SIZE))
    population[:, INPUTS * HIDDEN + HIDDEN:] = rng.normal(0, output_init_sigma, (population_size, GENOME_SIZE - INPUTS * HIDDEN - HIDDEN))
    hall, reigning, reigning_generation, stale, first_generation = [], None, None, 0, 1
    if resume:
        with np.load(resume, allow_pickle=False) as data:
            population, reigning, hall = data['population'], data['reigning'], list(data['hall'])
            state = json.loads(str(data['state']))
        if state.get('encoder', 'v1') != encoder:
            raise ValueError('Checkpoint encoder does not match')
        saved_config = state['config']
        saved_mode = saved_config[15] if len(saved_config) > 15 else 'both'
        early_stop = saved_mode if early_stop is None else early_stop
        saved_patience = saved_config[6]
        if extend_patience is not None:
            if extend_patience < saved_patience or (extend_patience == saved_patience and not state.get('patience_extended_from')):
                raise ValueError('Checkpoint configuration does not match')
            if patience not in (saved_patience, extend_patience, 50):
                raise ValueError('Checkpoint configuration does not match')
            extension_from = state.get('patience_extended_from') if extend_patience == saved_patience else saved_patience
            patience = extend_patience
        expected_config = [population_size, pairs, target, seed, validation_pairs, promotion_threshold, patience, check_batch, finalists, finalist_pairs_multiplier, output_init_sigma, fitness_mode, points_weight, style_strength, styles_after_first_promotion]
        comparable_config = saved_config[:15].copy()
        if extend_patience is not None:
            comparable_config[6] = patience
        if comparable_config != expected_config or early_stop != saved_mode:
            raise ValueError('Checkpoint configuration does not match')
        rng.bit_generator.state = state['rng']
        reigning_generation, stale, first_generation = state['reigning_generation'], state['stale'], state['generation'] + 1
        log_path = output / 'generations.jsonl'
        if log_path.exists():
            lines = log_path.read_text(encoding='utf-8').splitlines()
            kept = [line for line in lines if json.loads(line)['generation'] <= state['generation']]
            log_path.write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
        if state['ended'] and (extend_patience is None or stale >= patience):
            return
    (output / 'run_config.json').write_text(json.dumps({'seed': seed, 'target': target, 'encoder': encoder}), encoding='utf-8')
    early_stop = early_stop or 'reject-only'
    with (output / 'generations.jsonl').open('a' if resume else 'w', encoding='utf-8') as log:
        for generation in range(first_generation, generations + 1):
            start = perf_counter()
            schedule = league_schedule(rng, generation, population_size, pairs, seed, target, hall, reigning, styles_after_first_promotion)
            styles = [item['style'] or 'unbiased' for i in range(population_size) for item in schedule]
            scores, results = league_scores(population, schedule, hall, reigning, workers, style_strength, fitness_mode, points_weight, encoder)
            league_seconds = perf_counter() - start
            champion = int(np.argmax(scores))
            finalist_scores = []
            retest_pairs = pairs * finalist_pairs_multiplier if finalists else 0
            if finalists:
                candidates = np.argsort(-scores, kind='stable')[:min(finalists, population_size)]
                retest_schedule = []
                for p in range(retest_pairs):
                    item = dict(schedule[p % pairs])
                    item['seed'] = 2 * (10**15 + seed * 1000000000 + generation * 1000000 + p) + 1
                    item['target'] = target if target is not None else int(rng.choice((15, 30)))
                    item['starting_score'] = tuple(int(x) for x in rng.integers(item['target'], size=2)) if rng.random() < .3 else (0, 0)
                    retest_schedule.append(item)
                retest_jobs = []
                for i in candidates:
                    for item in retest_schedule:
                        if item['opponent'] == 'reigning':
                            opponent = reigning
                        elif item['opponent'] == 'hall':
                            opponent = hall[item['index']]
                        else:
                            index = item['index']
                            opponent = population[(index + 1) % population_size if index == i else index]
                        retest_jobs.append((population[i], opponent, item['seed'], item['target'], item['starting_score'], item['style'], style_strength, encoder, encoder))
                if workers == 1:
                    retested = list(map(_evaluate, retest_jobs))
                else:
                    with ProcessPoolExecutor(max_workers=workers) as pool:
                        retested = list(pool.map(_evaluate, retest_jobs))
                finalist_scores = [dict(individual=int(i), score=float(fitness(sum(retested[n * retest_pairs:(n + 1) * retest_pairs], []), fitness_mode, points_weight))) for n, i in enumerate(candidates)]
                champion = max(finalist_scores, key=lambda row: row['score'])['individual']
            name = f'champion_{generation:04d}.npy'
            NetworkPolicy(population[champion], encoder=encoder).save(output / name)
            genome = population[champion].copy()
            check, promoted = None, False
            check_seconds = 0.0
            if reigning is None:
                reigning, reigning_generation = genome.copy(), generation
            else:
                check_start = perf_counter()
                jobs = [(genome, reigning, deal_seed, match_target, (0, 0), None, 1.0, encoder, encoder) for deal_seed, match_target in validation_schedule(seed, target, validation_pairs)]
                pair_rates = []
                batch_size = check_batch or validation_pairs
                pool = ProcessPoolExecutor(max_workers=workers) if workers != 1 else None
                try:
                    for offset in range(0, validation_pairs, batch_size):
                        batch = jobs[offset:offset + batch_size]
                        checks = list(pool.map(_evaluate, batch) if pool else map(_evaluate, batch))
                        pair_rates.extend(sum(result.winner == seat for result, seat in pair) / 2 for pair in checks)
                        n = len(pair_rates)
                        rate = float(np.mean(pair_rates))
                        error = 1.96 * float(np.std(pair_rates, ddof=1)) / math.sqrt(n) if n > 1 else 0.0
                        if check_batch and (rate + error < promotion_threshold or (early_stop == 'both' and rate - error >= promotion_threshold)):
                            break
                finally:
                    if pool:
                        pool.shutdown()
                check_seconds = perf_counter() - check_start
                check = {'win_rate': rate, 'margin_of_error': error, 'pairs': n, 'max_pairs': validation_pairs, 'stopped_early': n < validation_pairs, 'early_stop': early_stop}
                promoted = rate >= promotion_threshold
                if promoted:
                    reigning, reigning_generation, stale = genome.copy(), generation, 0
                    hall.append(genome.copy())
                else:
                    stale += 1
            ended = stale >= patience
            NetworkPolicy(reigning, encoder=encoder).save(output / 'reigning_champion.npy')
            group = results[champion]
            first, seat = group[0]
            hand = [d for d in first.decisions if d['hand_index'] == 0]
            entry = {'generation': generation, 'phase': 1, 'fitness': {'max': float(scores.max()), 'average': float(scores.mean()), 'min': float(scores.min())}, 'behavior': _behavior(group), 'diversity': float(np.std(population, axis=0).mean()), 'example_hand': hand, 'champion': name}
            entry.update(promotion_check=check, promoted=promoted, reigning_champion=reigning_generation, phase_ended=ended, league=styles, schedule=schedule, finalists=finalist_scores, finalist_pairs=retest_pairs, selected_individual=champion)
            if encoder != 'v1':
                entry['encoder'] = encoder
            if extension_from is not None and generation == first_generation:
                entry.update(patience_extended_from=extension_from, patience=patience)
            log.write(json.dumps(entry) + '\n')
            log.flush()
            status = 'first champion' if check is None else f"{'promoted' if promoted else 'not promoted'} {check['win_rate']:.0%} +/-{check['margin_of_error']*100:.0f} ({check['pairs']}/{check['max_pairs']} pairs)"
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
                state = dict(generation=generation, config=[population_size, pairs, target, seed, validation_pairs, promotion_threshold, patience, check_batch, finalists, finalist_pairs_multiplier, output_init_sigma, fitness_mode, points_weight, style_strength, styles_after_first_promotion, early_stop], patience_extended_from=extension_from, rng=rng.bit_generator.state, reigning_generation=reigning_generation, stale=stale, ended=ended)
                state['encoder'] = encoder
                np.savez_compressed(output / f'checkpoint_{generation:04d}.npz', population=population, reigning=reigning, hall=np.asarray(hall).reshape(-1, GENOME_SIZE), state=json.dumps(state))
            if ended:
                break

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='training_output/phase1')
    parser.add_argument('--encoder', choices=('v1', 'v2'), default='v1')
    parser.add_argument('--population', type=int, default=64)
    parser.add_argument('--generations', type=int, default=1)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--target', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validation-pairs', type=int, default=1000)
    parser.add_argument('--early-stop', choices=('reject-only', 'both'), default=None)
    parser.add_argument('--check-batch', type=int, default=200)
    parser.add_argument('--promotion-threshold', type=float, default=.55)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--finalists', type=int, default=8)
    parser.add_argument('--finalist-pairs-multiplier', type=int, default=4)
    parser.add_argument('--output-init-sigma', type=float, default=.3)
    parser.add_argument('--fitness-mode', choices=('classic', 'mixed'), default='classic')
    parser.add_argument('--points-weight', type=float, default=.1)
    parser.add_argument('--style-strength', type=float, default=.5)
    parser.add_argument('--styles-after-first-promotion', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume')
    parser.add_argument('--extend-patience', type=int)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.extend_patience is not None and not args.resume:
        parser.error('--extend-patience requires --resume')
    if args.publish:
        destination = Path(__file__).resolve().parent.parent / 'champions' / 'phase1'
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('reigning_champion.npy', 'generations.jsonl', 'reference_evaluation.jsonl'):
            source = Path(args.output) / name
            if name != 'reference_evaluation.jsonl' or source.exists():
                shutil.copy2(source, destination / name)
        metadata = Path(args.output) / 'reigning_champion.encoder.json'
        target_metadata = destination / metadata.name
        if metadata.exists():
            shutil.copy2(metadata, target_metadata)
        else:
            target_metadata.unlink(missing_ok=True)
    else:
        train(args.output, args.population, args.generations, args.pairs, args.workers, args.target, args.seed, args.validation_pairs, args.promotion_threshold, args.patience, args.resume, args.check_batch, args.finalists, args.finalist_pairs_multiplier, args.output_init_sigma, args.fitness_mode, args.points_weight, args.style_strength, args.styles_after_first_promotion, args.early_stop, args.extend_patience, args.encoder)

if __name__ == '__main__':
    main()



