import unittest
import torch
from phase2_torch import Phase2Network, card_distribution

class AuxiliaryTest(unittest.TestCase):
    def test_exact_distribution(self):
        obs = dict(hand=((1, 'Espada'), (2, 'Oro'), (3, 'Basto')), player=0, mano=0,
                   table=[(1, (7, 'Copa'))], history=[],
                   envido_reveal={'declarations': [(0, 30), (1, 33)]})
        hands, probabilities, marginals = card_distribution(torch.zeros(40), obs)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(marginals.sum()), 2.0, places=5)
        self.assertTrue(all((7, 'Copa') in hand for hand in hands))
        self.assertTrue(all(any((7, suit) in hand and (6, suit) in hand for suit in ('Espada', 'Basto', 'Oro', 'Copa')) for hand in hands))
        self.assertEqual(float(marginals[0]), 0.0)

    def test_son_buenas_excludes_tied_mano(self):
        from opponent_inference import _tantos
        obs = dict(hand=((1, 'Espada'), (2, 'Oro'), (3, 'Basto')), player=0, mano=1,
                   table=[], history=[], envido_reveal={'declarations': [(0, 20), (1, 'son buenas')]})
        hands, probabilities, marginals = card_distribution(torch.zeros(40), obs)
        self.assertTrue(all(_tantos(hand) < 20 for hand in hands))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(marginals.sum()), 3.0, places=4)

    def test_head_off_preserves_output(self):
        model = Phase2Network(seed=4)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 451468)
        self.assertEqual(model(torch.zeros(1628)).shape, (12,))
