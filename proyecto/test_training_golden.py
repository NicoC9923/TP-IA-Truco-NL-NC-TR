"""Byte-level training regression against the pre-optimization implementation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from phase1_trainer import train

FIXTURE = Path(__file__).with_name('phase1_training_golden.json')

def capture(encoder, workers):
    with tempfile.TemporaryDirectory() as directory:
        # Ten generations include the real checkpoint writer, without mocking it.
        train(directory, population_size=6, pairs=2, finalists=2,
              validation_pairs=8, check_batch=4, generations=10,
              workers=workers, target=1, encoder=encoder)
        result = {}
        for path in sorted(Path(directory).iterdir()):
            if path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as archive:
                    for key in archive.files:
                        value = archive[key]
                        result[path.name + ':' + key] = hashlib.sha256(value.tobytes()).hexdigest()
            else:
                result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

class TrainingGoldenTests(unittest.TestCase):
    def test_all_encoders_and_worker_counts(self):
        expected = json.loads(FIXTURE.read_text())
        for encoder in ('v1', 'v2', 'v3'):
            for workers in (1, 2):
                with self.subTest(encoder=encoder, workers=workers):
                    self.assertEqual(expected[encoder], capture(encoder, workers))

if __name__ == '__main__':
    import sys
    if '--record' in sys.argv:
        FIXTURE.write_text(json.dumps({e: capture(e, 1) for e in ('v1', 'v2', 'v3')}, indent=2) + '\n')
    else:
        unittest.main()
