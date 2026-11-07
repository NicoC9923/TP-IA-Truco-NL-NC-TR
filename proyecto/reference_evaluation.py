"""Evaluate saved phase 1 champions against fixed reference opponents."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import time
from pathlib import Path
import numpy as np
from engine_policies import ReactivePolicy, AdaptiveUtilityPolicy
from phase1_network import NetworkPolicy
from symmetric_engine import play_match
from validation_schedule import validation_schedule


def _evaluate_pair(job):
    genome, name, deal_seed, match_target = job[:4]
    encoder = job[4] if len(job) > 4 else 'v1'
    factory = ReactivePolicy if name == 'reactive' else AdaptiveUtilityPolicy
    network = NetworkPolicy(genome, seed=deal_seed * 2, encoder=encoder)
    reference = factory(deal_seed * 2 + 1)
    first = play_match(network, reference, seed=deal_seed, target=match_target, deal_partition=0)
    second = play_match(reference, network, seed=deal_seed, target=match_target, deal_partition=0)
    return ((first.winner == 0) + (second.winner == 1)) / 2


def evaluate(directory, *, pairs=1000, seed=None, target=None, only_new=False, workers=1, every=1, promoted_only=False, encoder=None):
    if pairs < 1 or workers < 1 or every < 1:
        raise ValueError('pairs must be positive')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / 'run_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    seed = config.get('seed', 0) if seed is None else seed
    target = config.get('target') if target is None else target
    log = root / 'reference_evaluation.jsonl'
    done = set()
    if only_new and log.exists():
        for line in log.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            done.add((row['generation'], row['reference_opponent']))
    generation_log = root / 'generations.jsonl'
    generations = []
    if generation_log.exists():
        for line in generation_log.read_text(encoding='utf-8').splitlines():
            try:
                generations.append(json.loads(line))
            except json.JSONDecodeError:
                if not only_new:
                    raise
    promoted = {row['generation'] for row in generations if row.get('promoted')}
    final = generations[-1]['generation'] if generations and generations[-1].get('phase_ended') else None
    for champion in sorted(root.glob('champion_[0-9][0-9][0-9][0-9].npy')):
        generation = int(champion.stem.split('_')[1])
        if generation not in (1, final) and (generation % every != 0 or (promoted_only and generation not in promoted)):
            continue
        try:
            selected_encoder = encoder
            if selected_encoder is None and not champion.with_suffix('.encoder.json').exists():
                selected_encoder = config.get('encoder')
            policy = NetworkPolicy.load(champion, **({'encoder': selected_encoder} if selected_encoder is not None else {}))
            genome = policy.genome
        except (OSError, ValueError, EOFError):
            if only_new:
                # The trainer may still be writing this champion; retry next scan.
                continue
            raise
        for name, factory in (('reactive', ReactivePolicy), ('adaptive_utility', AdaptiveUtilityPolicy)):
            if (generation, name) in done:
                continue
            started = time.perf_counter()
            jobs = [(genome, name, deal_seed, match_target, policy.encoder) for deal_seed, match_target in validation_schedule(seed, target, pairs)]
            if workers == 1:
                rates = list(map(_evaluate_pair, jobs))
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    rates = list(pool.map(_evaluate_pair, jobs))
            row = dict(phase=1, generation=generation, reference_opponent=name,
                       win_rate=float(np.mean(rates)), margin_of_error=1.96 * float(np.std(rates, ddof=1)) / math.sqrt(pairs) if pairs > 1 else 0.0)
            with log.open('a', encoding='utf-8') as output:
                output.write(json.dumps(row) + '\n')
            print(f'generation {generation} | {name} | {time.perf_counter() - started:.2f} s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='training_output/phase1')
    parser.add_argument('--encoder', choices=('v1', 'v2'))
    parser.add_argument('--pairs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--target', type=int, default=None)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--every', type=int, default=1)
    parser.add_argument('--promoted-only', action='store_true')
    parser.add_argument('--watch', type=float, default=0, metavar='SECONDS')
    args = parser.parse_args()
    while True:
        evaluate(args.directory, pairs=args.pairs, seed=args.seed, target=args.target, only_new=bool(args.watch), workers=args.workers, every=args.every, promoted_only=args.promoted_only, encoder=args.encoder)
        if not args.watch:
            break
        time.sleep(args.watch)

if __name__ == '__main__':
    main()
