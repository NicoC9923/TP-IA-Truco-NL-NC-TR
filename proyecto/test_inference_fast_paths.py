import unittest
from unittest.mock import patch

from opponent_inference import Inference


class InferenceFastPathTests(unittest.TestCase):
    def test_played_and_excluded_cards_match_ordered_reference(self):
        cards = ((1, 'Espada'), (2, 'Copa'), (3, 'Oro'))
        for excluded in ((), ((6, 'Copa'),)):
            for played in (((7, 'Oro'),), ((7, 'Oro'), (5, 'Basto'))):
                with self.subTest(excluded=excluded, played=played):
                    belief = Inference(cards)
                    belief.truco_call(.3)
                    expected = {h: w for h, w in belief.weights.items()
                                if not set(excluded).intersection(h) and set(played).issubset(h)}
                    belief.seen_cards(excluded=excluded, opponent_played=played)
                    self.assertEqual(list(belief.weights.items()), list(expected.items()))
                    expected_distribution = {h: w / sum(expected.values()) for h, w in expected.items()}
                    self.assertEqual(belief.distribution(), expected_distribution)

    def test_neutral_call_preserves_weights_validation_and_cache_invalidation(self):
        cards = ((1, 'Espada'), (2, 'Copa'), (3, 'Oro'))
        belief = Inference(cards)
        belief.truco_call(.17)
        before = list(belief.weights.items())
        summary = belief.summaries(cards[0])
        with patch('opponent_inference._strength', side_effect=AssertionError('neutral call calculated strength')):
            belief.truco_call(1.0)
        self.assertEqual(list(belief.weights.items()), before)
        self.assertIsNone(belief._summary_cache)
        self.assertEqual(belief.summaries(cards[0]), summary)
        for invalid in (-.01, 1.01, float('nan')):
            with self.assertRaises(ValueError):
                belief.truco_call(invalid)
