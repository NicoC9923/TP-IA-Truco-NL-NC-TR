"""Network-only head-to-head of two champions on fresh held-out validation-partition deals."""
import sys, math, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, 'proyecto')
from phase1_trainer import _evaluate
from validation_schedule import validation_schedule

def main():
    a = np.load(sys.argv[1], allow_pickle=False); b = np.load(sys.argv[2], allow_pickle=False)
    seed, pairs = int(sys.argv[3]), int(sys.argv[4])
    jobs = [(a, b, s, t, (0, 0), None) for s, t in validation_schedule(seed, None, pairs)]
    with ProcessPoolExecutor(max_workers=12) as pool:
        res = list(pool.map(_evaluate, jobs, chunksize=8))
    rates = [sum(r.winner == seat for r, seat in pair) / 2 for pair in res]
    m = float(np.mean(rates)); e = 1.96 * float(np.std(rates, ddof=1)) / math.sqrt(pairs)
    print(json.dumps({'a': sys.argv[1], 'b': sys.argv[2], 'seed': seed, 'pairs': pairs, 'a_win_rate': m, 'margin': e}))

if __name__ == '__main__':
    main()
