"""Repeated public evidence must preserve the ordered, weighted posterior."""
import unittest
from unittest.mock import patch

from opponent_inference import Inference, IncrementalInference


class ScalarInference(Inference):
    """Pre-optimization filter, independent of the fast-path condition."""
    def seen_cards(self, excluded=(), opponent_played=()):
        excluded, played = set(excluded), set(opponent_played)
        self.played.update(played)
        self.weights = {h: w for h, w in self.weights.items()
                        if not excluded.intersection(h) and played.issubset(h)}


class EvidenceFilterTests(unittest.TestCase):
    def test_incremental_resets_match_original_filter_in_order(self):
        base = dict(hand=((1, 'Espada'), (2, 'Copa'), (3, 'Oro')),
                    player=0, mano=0, table=(), history=(), envido_reveal=None)
        played = ((1, (7, 'Oro')),)
        reveal = lambda value: {'declarations': ((0, 27), (1, value))}
        called = ((1, 'truco'),)
        observations = [
            (base, .17), (base, .17),
            ({**base, 'history': called}, .17),
            ({**base, 'history': called, 'table': played}, .17),
            ({**base, 'history': called, 'table': played}, .17),
            ({**base, 'history': called, 'table': played, 'envido_reveal': reveal(33)}, .17),
            ({**base, 'history': called, 'table': played, 'envido_reveal': reveal(33)}, .71),
            ({**base, 'table': played, 'envido_reveal': reveal(33)}, .71),
            ({**base, 'table': played, 'envido_reveal': reveal(32)}, .71),
            ({**base, 'table': played, 'envido_reveal': reveal('son buenas')}, .71),
            ({**base, 'table': played}, .71),
            ({**base, 'hand': ((1, 'Basto'), (2, 'Copa'), (3, 'Oro'))}, .71),
        ]
        fast, reference = IncrementalInference(), IncrementalInference()
        for obs, rate in observations:
            with self.subTest(observation=obs, rate=rate):
                actual = fast.update(obs, rate)
                with patch('opponent_inference.Inference', ScalarInference):
                    expected = reference.update(obs, rate)
                self.assertEqual(list(actual.weights.items()), list(expected.weights.items()))
                self.assertEqual(actual.played, expected.played)
                self.assertEqual(actual.distribution(), expected.distribution())
        fast.reset()
        reference.reset()
        actual = fast.update(base, .17)
        with patch('opponent_inference.Inference', ScalarInference):
            expected = reference.update(base, .17)
        self.assertEqual(list(actual.weights.items()), list(expected.weights.items()))

    def test_exclusions_still_apply_to_already_played_cards(self):
        own = ((1, 'Espada'), (2, 'Copa'), (3, 'Oro'))
        fast, reference = Inference(own), ScalarInference(own)
        for evidence in [dict(), dict(opponent_played=[(7, 'Oro')]),
                         dict(opponent_played=[(7, 'Oro')], excluded=[(6, 'Oro')]),
                         dict(opponent_played=[(7, 'Oro'), (5, 'Copa')]),
                         dict(opponent_played=[]),
                         dict(opponent_played=[(7, 'Oro')], excluded=[(7, 'Oro')])]:
            fast.seen_cards(**evidence)
            reference.seen_cards(**evidence)
            self.assertEqual(list(fast.weights.items()), list(reference.weights.items()))
            self.assertEqual(fast.played, reference.played)
