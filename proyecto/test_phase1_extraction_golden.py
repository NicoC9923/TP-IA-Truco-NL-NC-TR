import tempfile
import unittest
from pathlib import Path
from phase1_trainer import train


class TrainingGoldenTest(unittest.TestCase):
    def test_generation_log_matches_original_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            train(directory, population_size=4, pairs=2, generations=3, workers=1,
                  seed=42, validation_pairs=2, finalists=0, target=2)
            actual = (Path(directory) / 'generations.jsonl').read_bytes()
        expected = (Path(__file__).parent / 'test_fixtures/phase1_golden.jsonl').read_bytes()
        self.assertEqual(actual, expected)
