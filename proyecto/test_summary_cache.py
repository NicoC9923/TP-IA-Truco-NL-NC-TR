"""Cached arrays must match rebuilding after every posterior mutation."""
import unittest
from unittest.mock import patch
import numpy as np
from opponent_inference import Inference


class UncachedInference(Inference):
    def _summary_arrays(self):
        self._summary_cache = None
        return super()._summary_arrays()


class SummaryCacheTests(unittest.TestCase):
    def test_every_mutator_matches_fresh_arrays_and_scalar_summaries(self):
        own = ((1, 'Espada'), (2, 'Copa'), (3, 'Oro'))
        cached, reference = Inference(own), UncachedInference(own)
        operations = [
            lambda b: None,
            lambda b: b.seen_cards(),
            lambda b: b.truco_call(.17),
            lambda b: b.declared_tantos(33),
            lambda b: b.seen_cards(opponent_played=[(7, 'Oro')]),
            lambda b: b.seen_cards(opponent_played=[(7, 'Oro')]),
            lambda b: b.son_buenas(33, opponent_is_mano=False),
            lambda b: b.seen_cards(excluded=[(6, 'Copa')]),
            lambda b: b.seen_cards(opponent_played=[(7, 'Oro'), (6, 'Oro')]),
            lambda b: b.truco_call(.71),
            lambda b: b.seen_cards(opponent_played=[(7, 'Oro'), (6, 'Oro'), (5, 'Basto')]),
            lambda b: b.seen_cards(excluded=[(7, 'Oro')]),
        ]
        for index, operation in enumerate(operations):
            with self.subTest(step=index):
                operation(cached)
                operation(reference)
                for actual, expected in zip(cached._summary_arrays(), reference._summary_arrays()):
                    self.assertEqual(actual.tobytes(), expected.tobytes())
                self.assertEqual(cached.summaries(own[0]), reference.summaries(own[0]))
                self.assertEqual(cached.v3_summaries(own, (0, 1, 2), estimate_outcomes=True),
                                 reference.v3_summaries(own, (0, 1, 2), estimate_outcomes=True))
                with patch.object(np, 'fromiter', side_effect=AssertionError('Rebuilt an unchanged posterior')):
                    cached._summary_arrays()
