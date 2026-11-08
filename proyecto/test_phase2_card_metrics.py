import math
import unittest
from phase2_card_metrics import prediction_metrics

class CardPredictionTest(unittest.TestCase):
    def test_exposed_cards_are_excluded_from_top_n_accuracy(self):
        hands = [((1, 'Espada'), (2, 'Oro'), (3, 'Basto')),
                 ((1, 'Espada'), (4, 'Oro'), (5, 'Basto'))]
        result = prediction_metrics(hands, [.75, .25], hands[0], [(1, 'Espada')])
        self.assertAlmostEqual(result['log_loss'], -math.log(.75))
        self.assertEqual(result['hidden_card_accuracy'], 1.0)

    def test_top_n_is_hidden_card_count(self):
        hands = [((1, 'Espada'), (2, 'Oro'), (3, 'Basto')),
                 ((1, 'Espada'), (4, 'Oro'), (5, 'Basto'))]
        result = prediction_metrics(hands, [.25, .75], hands[0], [(1, 'Espada')])
        self.assertEqual(result['hidden_card_accuracy'], 0.0)
        self.assertAlmostEqual(result['log_loss'], -math.log(.25))

    def test_validation_predictions_are_deterministic(self):
        from phase2_card_metrics import validation_predictions, _validation_situations
        from phase2_torch import Phase2Network
        model = Phase2Network(seed=5, auxiliary=True)
        first = validation_predictions(model, schedule_seed=10000, target=1, temperature=.1)
        _validation_situations.cache_clear()
        second = validation_predictions(model, schedule_seed=10000, target=1, temperature=.1)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {'exact', 'weighted', 'learned'})
        for metrics in first.values():
            self.assertGreater(metrics['situations'], 0)
            self.assertGreaterEqual(metrics['log_loss'], 0)
            self.assertTrue(0 <= metrics['hidden_card_accuracy'] <= 1)
