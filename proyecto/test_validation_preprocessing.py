"""Compare cached validation with the original per-call prediction path."""
import math
import statistics
import unittest
from unittest.mock import patch

import torch
import phase2_card_metrics as metrics
from phase2_torch import Phase2Network, card_distribution
from opponent_inference import exact_inference_from_observation, inference_from_observation


def original_predictions(model, config):
    """Reference from before fixed preprocessing, including learned distribution."""
    rows = {name: [] for name in ('exact', 'weighted', 'learned')}
    for obs, inputs, rate, true_hand, exposed in metrics._validation_situations(**config):
        exact = exact_inference_from_observation(obs).distribution()
        weighted = inference_from_observation(obs, rate).distribution()
        with torch.no_grad():
            hands, probabilities, _ = card_distribution(model.card_logits(torch.from_numpy(inputs)).double(), obs)
        for name, candidate_hands, weights in (
                ('exact', tuple(exact), tuple(exact.values())),
                ('weighted', tuple(weighted), tuple(weighted.values())),
                ('learned', hands, probabilities.tolist())):
            rows[name].append(metrics.prediction_metrics(candidate_hands, weights, true_hand, exposed))
    result = {}
    for name, values in rows.items():
        accuracies = [row['hidden_card_accuracy'] for row in values]
        result[name] = {}
        for key in ('log_loss', 'hidden_card_accuracy'):
            total = 0.0
            # Preserve the original += sequence; Python 3.12 sum compensates.
            for row in values:
                total += row[key]
            result[name][key] = total / len(values)
        result[name]['situations'] = len(values)
        result[name]['accuracy_standard_error'] = statistics.stdev(accuracies) / math.sqrt(len(values)) if len(values) > 1 else None
    return result


class ValidationPreprocessingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        metrics._validation_situations.cache_clear()
        metrics._validation_preprocessing.cache_clear()
        self.model = Phase2Network(seed=35, auxiliary=True)
        self.config = dict(schedule_seed=10000, target='mixed', temperature=.1, situation_count=12)

    def tearDown(self):
        metrics._validation_preprocessing.cache_clear()
        metrics._validation_situations.cache_clear()

    def test_cold_warm_and_changed_model_match_original_exactly(self):
        expected = original_predictions(self.model, self.config)
        cold = metrics.validation_predictions(self.model, **self.config)
        self.assertEqual(cold, expected)
        cold['exact']['log_loss'] = -1  # Returned reports cannot mutate the cache.
        with patch.object(self.model, 'card_logits', wraps=self.model.card_logits) as predict:
            warm = metrics.validation_predictions(self.model, **self.config)
            self.assertEqual(predict.call_count, self.config['situation_count'])
        self.assertEqual(warm, expected)
        with torch.no_grad():
            self.model.card_head[-1].bias.add_(torch.linspace(-3, 3, 40))
        changed = metrics.validation_predictions(self.model, **self.config)
        self.assertEqual(changed, original_predictions(self.model, self.config))
        self.assertNotEqual(changed['learned']['log_loss'], expected['learned']['log_loss'])
        self.assertEqual(changed['exact'], expected['exact'])
        self.assertEqual(changed['weighted'], expected['weighted'])
        metrics._validation_preprocessing.cache_clear()
        metrics._validation_situations.cache_clear()
        self.assertEqual(metrics.validation_predictions(self.model, **self.config), changed)

    def test_protocol_changes_match_original(self):
        base = {**self.config, 'target': 1, 'situation_count': 4}
        for config in (base, {**base, 'schedule_seed': 10001}, {**base, 'target': 2},
                       {**base, 'temperature': .2}, {**base, 'situation_count': 5}, base):
            with self.subTest(config=config):
                self.assertEqual(metrics.validation_predictions(self.model, **config),
                                 original_predictions(self.model, config))
