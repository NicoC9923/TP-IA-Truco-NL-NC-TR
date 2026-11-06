import json, tempfile, unittest
from pathlib import Path
from phase1_trainer import train

class EarlyStoppingTests(unittest.TestCase):
    def test_decisive_check_stops_after_first_batch(self):
        with tempfile.TemporaryDirectory() as d:
            train(d, population_size=2, generations=2, pairs=1, workers=1, target=1, validation_pairs=4, check_batch=2, promotion_threshold=2)
            row = json.loads(Path(d, 'generations.jsonl').read_text().splitlines()[1])
            self.assertEqual(row['promotion_check']['pairs'], 2)
            self.assertTrue(row['promotion_check']['stopped_early'])
            self.assertFalse(row['promoted'])
    def test_promotion_close_and_disabled(self):
        for threshold, batch, expected_pairs, promoted in [(-1, 2, 2, True), (.5, 0, 4, False)]:
            with self.subTest(threshold=threshold), tempfile.TemporaryDirectory() as d:
                train(d, population_size=2, generations=2, pairs=1, workers=1, target=1, validation_pairs=4, check_batch=batch, promotion_threshold=threshold)
                row = json.loads(Path(d, 'generations.jsonl').read_text().splitlines()[1])
                self.assertEqual(row['promotion_check']['pairs'], expected_pairs)
                if batch:
                    self.assertEqual(row['promoted'], promoted)
    def test_close_check_uses_all_pairs_and_disabled_matches_golden(self):
        for batch in (2, 0):
            with tempfile.TemporaryDirectory() as d:
                train(d, population_size=2, generations=2, pairs=1, workers=1, target=1, seed=0, validation_pairs=4, check_batch=batch, promotion_threshold=.5)
                check = json.loads(Path(d, 'generations.jsonl').read_text().splitlines()[1])['promotion_check']
                self.assertEqual(check['pairs'], 4)
                self.assertFalse(check['stopped_early'])
                self.assertEqual(check['win_rate'], .375)
                self.assertAlmostEqual(check['margin_of_error'], 0.245)
