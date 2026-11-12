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
        for threshold, batch, expected_pairs, promoted in [(-1, 2, 4, True), (.5, 0, 4, False)]:
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
                self.assertGreaterEqual(check['win_rate'], 0)
                self.assertLessEqual(check['win_rate'], 1)
                self.assertGreaterEqual(check['margin_of_error'], 0)


    def test_both_preserves_early_promotion(self):
        with tempfile.TemporaryDirectory() as d:
            train(d, population_size=2, generations=2, pairs=1, workers=1, target=1, seed=0, validation_pairs=4, check_batch=2, promotion_threshold=-1, early_stop='both')
            row = json.loads(Path(d, 'generations.jsonl').read_text().splitlines()[1])
            self.assertEqual(row['promotion_check']['pairs'], 2)
            self.assertTrue(row['promoted'])

    def test_old_checkpoint_resumes_in_both_mode(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            opts = dict(population_size=2, pairs=1, workers=1, target=1, validation_pairs=1, finalists=0, promotion_threshold=-1)
            train(d, generations=10, early_stop='both', **opts)
            path = Path(d, 'checkpoint_0010.npz')
            with np.load(path) as data:
                contents = {k: data[k] for k in data.files}
            state = json.loads(str(contents['state']))
            state['config'].pop()
            contents['state'] = json.dumps(state)
            np.savez_compressed(path, **contents)
            train(d, generations=11, resume=path, **opts)
            row = json.loads(Path(d, 'generations.jsonl').read_text().splitlines()[-1])
            self.assertEqual(row['promotion_check']['early_stop'], 'both')

    def test_strong_challenger_plays_all_1000_pairs(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        def wins(job):
            result = SimpleNamespace(winner=0, points=(1, 0), decisions=[])
            return [(result, 0), (result, 0)]
        with tempfile.TemporaryDirectory() as d, patch('phase1_trainer._evaluate', side_effect=wins):
            train(d, population_size=2, generations=2, pairs=1, workers=1, target=1, validation_pairs=1000, finalists=0)
            row = json.loads(Path(d, 'generations.jsonl').read_text().splitlines()[1])
            self.assertEqual(row['promotion_check']['pairs'], 1000)
            self.assertTrue(row['promoted'])
