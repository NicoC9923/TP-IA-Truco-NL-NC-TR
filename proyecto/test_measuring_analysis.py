import unittest
from measuring_analysis import generation_size, prior_metrics, temperature_metrics

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
