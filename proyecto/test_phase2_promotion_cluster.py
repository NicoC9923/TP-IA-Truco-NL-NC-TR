import math
import unittest
from phase2_promotion import promotion_decision


class ClusteredPromotionTests(unittest.TestCase):
    def test_identical_correlated_panel_members_are_not_promoted(self):
        challenger = {str(i): [1.0] * 56 + [0.0] * 44 for i in range(4)}
        champion = {str(i): [0.0] * 56 + [1.0] * 44 for i in range(4)}
        result = promotion_decision(challenger, champion, [.5] * 200)
        self.assertAlmostEqual(result['gain'], .12)
        self.assertAlmostEqual(result['lower_bound'], -0.075564, places=5)
        self.assertFalse(result['promoted'])

    def test_independent_member_patterns_match_closed_form(self):
        # Balanced orthogonal member patterns have zero sample covariance.
        challenger = {str(bit): [float(bool(i & (1 << bit))) for i in range(8)]
                      for bit in range(3)}
        champion = {name: [.5] * 8 for name in challenger}
        result = promotion_decision(challenger, champion, [.5])
        # Each member's sample variance is 2/7; averaging three independent
        # members and then eight deal clusters gives variance (2/7)/3/8.
        margin = 1.96 * math.sqrt(1 / 84)
        self.assertAlmostEqual(result['gain'], 0)
        self.assertAlmostEqual(result['lower_bound'], -margin)
        self.assertAlmostEqual(result['upper_bound'], margin)

    def test_unequal_member_schedules_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'matching deals'):
            promotion_decision({'a': [1, 0], 'b': [1, 0, 1]},
                               {'a': [0, 0], 'b': [0, 0, 0]}, [.5])
