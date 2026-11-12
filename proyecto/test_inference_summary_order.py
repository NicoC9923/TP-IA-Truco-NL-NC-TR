"""Ordered floating-point reference for vectorized posterior summaries."""
import random
import unittest
from opponent_inference import Inference, _strength
import truco_reglas as rules

class SummaryOrderTests(unittest.TestCase):
    def test_exact_reference_with_weighted_and_filtered_posteriors(self):
        rng = random.Random(74)
        deck = [(v, p) for p in rules.PALOS for v in rules.VALORES]
        bravas = {(1, 'Espada'), (1, 'Basto'), (7, 'Espada'), (7, 'Oro')}
        for step in range(8):
            cards = rng.sample(deck, 6)
            belief = Inference(cards[:3])
            belief.truco_call(.17)
            belief.truco_call(.71)
            belief.seen_cards(opponent_played=cards[3:3 + step % 4])
            expected = [0., 0., 0., 0., 0.]
            legacy = 0.
            ranks = [rules.valor_carta_truco(*c) for c in cards[:3]]
            for hand, weight in belief.distribution().items():
                remaining = [c for c in hand if c not in belief.played]
                expected[0] += weight * bool(bravas.intersection(remaining))
                expected[1] += weight * _strength(tuple(remaining))
                if remaining:
                    others = [rules.valor_carta_truco(*c) for c in remaining]
                    for i, rank in enumerate(ranks):
                        expected[i+2] += weight * sum((rank > other) - (rank < other) for other in others) / len(remaining)
                    legacy += weight * sum(ranks[0] >= other for other in others) / len(remaining)
            actual = belief.v3_summaries(cards[:3], (0, 1, 2), estimate_outcomes=True)
            self.assertEqual(expected, [actual[0], actual[1], *actual[2]])
            self.assertEqual(dict(brava_probability=expected[0], expected_strength=expected[1], next_trick_win_probability=legacy), belief.summaries(cards[0], ties_win=True))
