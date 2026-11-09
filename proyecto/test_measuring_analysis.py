import unittest
import json
import tempfile
from pathlib import Path
from measuring_analysis import generation_size, prior_metrics, temperature_metrics, segment

class MeasuringAnalysisTests(unittest.TestCase):
    def test_temperature(self):
        result = temperature_metrics([([0, 1], 0), ([0, 1], 1)])
        self.assertEqual(result['temperature'], .2)
        self.assertEqual(result['tactical_mistake_rate'], .5)
        self.assertEqual(temperature_metrics([([0, 0, 0], 0)])['temperature'], .05)
        self.assertEqual(temperature_metrics([([0, 0], 0)])['temperature'], .1)

    def test_prior_and_ties(self):
        self.assertEqual(prior_metrics([[('fold', True), ('fold', True)]])['chosen_prior'], 5)
        self.assertEqual(prior_metrics([[('fold', True), ('fold', False), ('fold', True)]])['scored_opportunities'], 2)

    def test_generation(self):
        result = generation_size(10, 20)
        self.assertGreaterEqual(result['estimated_seconds'], 300)
        self.assertLessEqual(result['estimated_seconds'], 600)
        self.assertLessEqual(result['overhead_fraction'], .15)
        self.assertIsNone(generation_size(1, 100)['matches'])

    def test_head_off_segment_uses_pre_check_elapsed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'progress.json').write_text(json.dumps(dict(matches=480, samples=960, elapsed=50, matches_per_s=9, samples_per_s=18)))
            rows = [dict(matches=240, check={'gain': 0}, elapsed_seconds=25) for _ in range(2)]
            (root / 'generations.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            bench = root / 'bench.json'
            bench.write_text(json.dumps(dict(check_seconds=[5], validation_seconds=[], learner_seconds_per_update=.01)))
            result = segment(root, bench)
            self.assertEqual(result['training_matches_per_s'], 9.6)
            self.assertEqual(result['generation_seconds'], [30, 30])
            self.assertEqual(result['seconds_per_check'], 5)
            self.assertEqual(result['learner_ms_per_update'], 10)
