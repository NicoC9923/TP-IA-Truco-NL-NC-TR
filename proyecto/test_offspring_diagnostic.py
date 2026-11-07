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
