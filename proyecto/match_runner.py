"""Run reference matches and measure throughput."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from engine_policies import RandomPolicy, ReactivePolicy
from symmetric_engine import play_match, play_mirror_pair

def run_one(args):
    seed, target, score, mirror, names = args
    a, b = [(ReactivePolicy if name == 'reactive' else RandomPolicy)(seed * 2 + i) for i, name in enumerate(names)]
    if mirror:
        return play_mirror_pair(a, b, seed=seed, target=target, starting_score=score)
    return (play_match(a, b, seed=seed, target=target, starting_score=score),)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', type=int, default=15)
    parser.add_argument('--score', type=int, nargs=2, default=(0, 0))
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--policy-a', choices=('reactive', 'random'), default='reactive')
    parser.add_argument('--policy-b', choices=('reactive', 'random'), default='reactive')
    args = parser.parse_args()
    if args.matches < 1:
        parser.error('--matches must be positive')
    jobs = [(args.seed+i, args.target, tuple(args.score), args.mirror, (args.policy_a, args.policy_b)) for i in range(args.matches)]
    for threads in (1, 16):
        start = perf_counter()
        with ProcessPoolExecutor(max_workers=threads) as executor:
            results = list(executor.map(run_one, jobs))
        elapsed = perf_counter() - start
        wins = [0, 0]
        for pair in results:
            for index, result in enumerate(pair):
                wins[result.winner if index == 0 else 1-result.winner] += 1
        total = sum(wins)
        print(f'{threads} threads: {total} matches, wins {wins}, {total / elapsed:.2f} matches/second; mirror={args.mirror}')

if __name__ == '__main__':
    main()
