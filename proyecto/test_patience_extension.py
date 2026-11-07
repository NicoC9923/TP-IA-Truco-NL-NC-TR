import unittest
import tempfile
import json
from pathlib import Path
from phase1_trainer import train

class PatienceExtensionTest(unittest.TestCase):
    def test_extension_continues_finished_run(self):
        with tempfile.TemporaryDirectory() as d:
            opts = dict(population_size=2, pairs=1, workers=1, target=1, seed=1, validation_pairs=1, promotion_threshold=2, patience=9, finalists=0)
            train(d, generations=10, **opts)
            path = Path(d, 'checkpoint_0010.npz')
            train(d, generations=11, resume=path, extend_patience=12, **opts)
            entries = [json.loads(x) for x in Path(d, 'generations.jsonl').read_text().splitlines()]
            self.assertEqual(entries[-1]['generation'], 11)
            self.assertEqual(entries[-1]['patience_extended_from'], 9)
            self.assertEqual(entries[-1]['patience'], 12)
    def test_invalid_extensions_and_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            opts = dict(population_size=2, pairs=1, workers=1, target=1, seed=1, validation_pairs=1, promotion_threshold=2, patience=9, finalists=0)
            train(d, generations=10, **opts)
            path = Path(d, 'checkpoint_0010.npz')
            for changes in ({'patience': 12}, {'extend_patience': 9}, {'extend_patience': 8}, {'extend_patience': 12, 'seed': 2}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'Checkpoint configuration does not match'):
                    train(d, generations=11, resume=path, **(opts | changes))
            with self.assertRaisesRegex(ValueError, 'requires --resume'):
                train(d, extend_patience=12, **opts)
