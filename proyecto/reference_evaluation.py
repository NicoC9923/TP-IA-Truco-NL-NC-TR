"""Evaluate saved phase 1 champions against fixed reference opponents."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from engine_policies import ReactivePolicy, AdaptiveUtilityPolicy
from phase1_network import NetworkPolicy
from symmetric_engine import play_match
from validation_schedule import validation_schedule


def evaluate(directory, *, pairs=1000, seed=None, target=None, only_new=False):
    if pairs < 1:
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
    for champion in sorted(root.glob('champion_[0-9][0-9][0-9][0-9].npy')):
        generation = int(champion.stem.split('_')[1])
        try:
            genome = np.load(champion, allow_pickle=False)
            # Validate the shape before starting any matches.
            NetworkPolicy(genome)
        except (OSError, ValueError, EOFError):
            if only_new:
                # The trainer may still be writing this champion; retry next scan.
                continue
            raise
        for name, factory in (('reactive', ReactivePolicy), ('adaptive_utility', AdaptiveUtilityPolicy)):
            if (generation, name) in done:
                continue
            rates = []
            for deal_seed, match_target in validation_schedule(seed, target, pairs):
                network = NetworkPolicy(genome, seed=deal_seed * 2)
                reference = factory(deal_seed * 2 + 1)
                first = play_match(network, reference, seed=deal_seed, target=match_target, deal_partition=0)
                second = play_match(reference, network, seed=deal_seed, target=match_target, deal_partition=0)
                rates.append(((first.winner == 0) + (second.winner == 1)) / 2)
            row = dict(phase=1, generation=generation, reference_opponent=name,
                       win_rate=float(np.mean(rates)), margin_of_error=1.96 * float(np.std(rates, ddof=1)) / math.sqrt(pairs) if pairs > 1 else 0.0)
            with log.open('a', encoding='utf-8') as output:
                output.write(json.dumps(row) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='training_output/phase1')
    parser.add_argument('--pairs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--target', type=int, default=None)
    parser.add_argument('--watch', type=float, default=0, metavar='SECONDS')
    args = parser.parse_args()
    while True:
        evaluate(args.directory, pairs=args.pairs, seed=args.seed, target=args.target, only_new=bool(args.watch))
        if not args.watch:
            break
        time.sleep(args.watch)

if __name__ == '__main__':
    main()
