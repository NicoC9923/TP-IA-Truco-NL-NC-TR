"""Short pool/serialization probe, without training output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'proyecto'))
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import pickle
import numpy as np
from phase1_network import GENOME_SIZE
from phase1_trainer import _evaluate

def identity(value):
    return value

if __name__ == '__main__':
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=12) as pool:
        list(pool.map(identity, range(24)))
        print('cold pool + 24 trivial jobs', perf_counter()-start)
        start = perf_counter()
        list(pool.map(identity, range(24)))
        print('warm pool + 24 trivial jobs', perf_counter()-start)
    genome = np.random.default_rng(0).normal(0, .1, GENOME_SIZE)
    job = (genome, genome.copy(), 1, 15, (0, 0), None, 1., 'v3', 'v3')
    result = _evaluate(job)
    for name, value in [('job', job), ('full result', result)]:
        data = pickle.dumps(value)
        start = perf_counter()
        for _ in range(1000):
            pickle.loads(pickle.dumps(value))
        print(name, 'bytes', len(data), '1000 roundtrips seconds', perf_counter()-start)
