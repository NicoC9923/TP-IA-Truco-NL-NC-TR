import unittest
import numpy as np
from offspring_diagnostic import variants

class OffspringTests(unittest.TestCase):
    def test_operators_match_trainer(self):
        a, b = np.zeros(1860), np.ones(1860)
        actual = variants(a, b, np.random.default_rng(42))
        rng = np.random.default_rng(42)
        cross = np.where(rng.random(1860) < .5, a, b)
        expected = cross + rng.normal(0, .05, 1860) * (rng.random(1860) < .1)
        np.testing.assert_array_equal(actual['crossover_mutation'], expected)
        np.testing.assert_array_equal(actual['crossover_only'], cross)

class SummaryTests(unittest.TestCase):
    def test_summary_and_namespace(self):
        from offspring_diagnostic import summary, SEED_BASE
        from behavior_diversity import BANK_SEED
        result = summary([-.2, .2])
        self.assertEqual(result['mean'], 0)
        self.assertEqual(result['fraction_beating_parent'], .5)
        self.assertLess(result['ci95'][0], 0)
        self.assertGreater(result['ci95'][1], 0)
        self.assertGreater(SEED_BASE, 10**17 + 10**6)
        self.assertGreater(BANK_SEED, SEED_BASE + 10**6)
