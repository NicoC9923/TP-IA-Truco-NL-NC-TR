"""Run a separate tournament of reference policies and optional policy factories."""
import argparse
import importlib
import itertools
import json
import math

from engine_policies import AdaptiveUtilityPolicy, ReactivePolicy
from symmetric_engine import play_mirror_pair


def tournament(policies, *, pairs=100, seed=0, target=15):
    """Return pairwise results; factories accept an integer seed and return policies."""
    if pairs < 1:
        raise ValueError('pairs must be positive')
    results = []
    for (name_a, factory_a), (name_b, factory_b) in itertools.combinations(policies.items(), 2):
        wins = [0, 0]
        calls = [0, 0]
        bluffs = [0, 0]
        for index in range(pairs):
            deal_seed = seed + index
            a = factory_a(deal_seed * 2)
            b = factory_b(deal_seed * 2 + 1)
            matches = play_mirror_pair(a, b, seed=deal_seed, target=target)
            for seat_swap, match in enumerate(matches):
                wins[match.winner ^ seat_swap] += 1
                for decision in match.decisions:
                    if decision['bluff'] is not None:
                        player = decision['player'] ^ seat_swap
                        calls[player] += 1
                        bluffs[player] += int(decision['bluff'])
        n = sum(wins)
        rate = wins[0] / n
        z = 1.959963984540054
        denom = 1 + z*z/n
        center = (rate + z*z/(2*n)) / denom
        margin = z * math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / denom
        results.append({'opponents': [name_a, name_b], 'matches': n,
                        'win_rate': rate, 'win_rates': {name_a: rate, name_b: 1-rate},
                        'margin_of_error': margin,
                        'confidence_interval': [center-margin, center+margin],
                        'bluff_rates': {name: bluffs[i]/calls[i] if calls[i] else None
                                        for i, name in enumerate((name_a, name_b))}})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', type=int, default=15)
    parser.add_argument('--policy', action='append', default=[], metavar='NAME=MODULE:FACTORY',
                        help='additional seed-accepting policy factory (repeatable)')
    args = parser.parse_args()
    policies = {'reactive': ReactivePolicy, 'adaptive_utility': AdaptiveUtilityPolicy}
    for entry in args.policy:
        try:
            name, location = entry.split('=', 1)
            module, attr = location.split(':', 1)
            if not name or name in policies:
                raise ValueError('policy names must be unique and nonempty')
            policies[name] = getattr(importlib.import_module(module), attr)
        except (ValueError, ImportError, AttributeError) as error:
            parser.error(str(error))
    for result in tournament(policies, pairs=args.pairs, seed=args.seed, target=args.target):
        print(json.dumps(result))


if __name__ == '__main__':
    main()
