import sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'proyecto'))
import phase1_trainer as t

def main():
    encoder, label = sys.argv[1:]
    original = t.league_scores
    def league(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        print('PHASE league', time.perf_counter()-start, flush=True)
        return result
    t.league_scores = league
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        t.train(directory, population_size=64, generations=2, pairs=16,
                finalists=8, finalist_pairs_multiplier=4, validation_pairs=1000,
                check_batch=200, workers=12, encoder=encoder)
    print(label, encoder, 'TOTAL', time.perf_counter()-start, flush=True)
if __name__ == '__main__':
    main()
