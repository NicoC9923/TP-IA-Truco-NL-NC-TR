"""Truco strength percentile table (ADR 0062): built only from the rules module."""
import json
import unittest
from pathlib import Path
import truco_reglas as rules
import truco_strength as strength

BEST = [(1, 'Espada'), (1, 'Basto'), (7, 'Espada')]
WEAKEST_THREE = [(4, 'Espada'), (4, 'Basto'), (4, 'Oro')]


class TrucoStrengthTest(unittest.TestCase):
    def test_single_card_percentiles_are_exact_midranks(self):
        # Rank 13 is one card of 40: 39 cards below it, none tied.
        self.assertEqual(strength.percentile([(1, 'Espada')]), (39 + 0.5) / 40)
        # Rank 0 is the four 4s: none below, three tied besides itself.
        self.assertEqual(strength.percentile([(4, 'Copa')]), (0 + 0.5 * 4) / 40)

    def test_no_cards_is_zero(self):
        self.assertEqual(strength.percentile([]), 0.0)

    def test_order_and_suit_of_equal_ranks_do_not_matter(self):
        self.assertEqual(strength.percentile([(3, 'Oro'), (1, 'Espada')]),
                         strength.percentile([(1, 'Espada'), (3, 'Copa')]))

    def test_percentile_follows_the_rules_strength_measure(self):
        hands = [BEST, WEAKEST_THREE, [(3, 'Oro'), (2, 'Copa'), (12, 'Basto')], [(5, 'Oro'), (6, 'Copa'), (7, 'Copa')]]
        by_measure = sorted(hands, key=lambda h: rules.poder_mano_truco([{'valor': v, 'palo': p} for v, p in h]))
        values = [strength.percentile(h) for h in by_measure]
        self.assertEqual(values, sorted(values))
        self.assertGreater(strength.percentile(BEST), 0.999)
        self.assertLess(strength.percentile(WEAKEST_THREE), 0.01)

    def test_three_card_table_is_a_proper_distribution(self):
        from itertools import combinations
        deck = [(v, p) for p in rules.PALOS for v in rules.VALORES]
        values = [strength.percentile(list(h)) for h in combinations(deck, 3)]
        self.assertEqual(len(values), 9880)
        self.assertAlmostEqual(sum(values) / len(values), 0.5, places=9)

    def test_committed_table_regenerates_identically(self):
        stored = json.loads(Path(strength.TABLE_PATH).read_text(encoding='utf-8'))
        self.assertEqual(stored, json.loads(json.dumps(strength.generate())))

    def test_module_does_not_use_adaptive_utility_code(self):
        source = Path(strength.__file__).read_text(encoding='utf-8')
        for forbidden in ('truco_ia', 'engine_policies', 'AdaptiveUtility'):
            self.assertNotIn(forbidden, source)


if __name__ == '__main__':
    unittest.main()
