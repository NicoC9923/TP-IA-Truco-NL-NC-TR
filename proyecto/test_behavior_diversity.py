import unittest
import numpy as np
from behavior_diversity import compare

class BehaviorTests(unittest.TestCase):
    def test_known_distributions(self):
        distance, agreement = compare(np.array([[.5, .5], [1., 0.]]), np.array([[1., 0.], [0., 1.]]))
        self.assertEqual(distance, .75)
        self.assertEqual(agreement, .5)
