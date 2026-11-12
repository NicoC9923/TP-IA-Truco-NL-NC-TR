"""Bit-level references for auxiliary score and evidence optimizations."""
import unittest
import numpy as np
import torch
from opponent_inference import _IDS, _HANDS, _hand_tantos, compatible_mask
from phase2_torch import combination_scores


class CombinationScoresTests(unittest.TestCase):
    def test_scores_and_duplicate_card_gradient_match_original(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(35)
        for dtype, batch in ((torch.float32, 1), (torch.float32, 32),
                             (torch.float32, 512), (torch.float64, 7)):
            for hand_ids in (torch.from_numpy(_IDS), torch.from_numpy(_IDS)[::13]):
                with self.subTest(dtype=dtype, batch=batch, hands=len(hand_ids)):
                    source = torch.randn(batch, 40, dtype=dtype, generator=generator)
                    source[0].fill_(-0.0)
                    original, optimized = source.clone().requires_grad_(), source.clone().requires_grad_()
                    expected = original[:, hand_ids].sum(dim=2)
                    actual = combination_scores(optimized, hand_ids)
                    self.assertEqual(actual.detach().numpy().tobytes(), expected.detach().numpy().tobytes())
                    gradient = torch.randn(expected.shape, dtype=dtype, generator=generator)
                    expected.backward(gradient)
                    actual.backward(gradient)
                    self.assertEqual(optimized.grad.numpy().tobytes(), original.grad.numpy().tobytes())

    def test_membership_masks_match_original_for_varied_evidence(self):
        rng = np.random.default_rng(35)
        evidence = []
        for i in range(512):
            cards = rng.choice(40, size=6, replace=False)
            evidence.append((tuple(cards[:3]), tuple(cards[3:3 + i % 4]),
                             (-1, 0, 20, 33)[i % 4], (-1, 0, 20, 27, 33, 34)[i % 6]))
        expected = np.ones((len(evidence), len(_HANDS)), dtype=bool)
        tantos = _hand_tantos()
        for row, (own, played, declared, upper) in zip(expected, evidence):
            for card in own:
                row &= ~(_IDS == card).any(axis=1)
            for card in played:
                row &= (_IDS == card).any(axis=1)
            if declared >= 0:
                row &= tantos == declared
            row &= tantos <= upper
        np.testing.assert_array_equal(compatible_mask(evidence), expected)
        self.assertEqual(compatible_mask([]).shape, (0, len(_HANDS)))
