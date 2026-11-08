"""Metrics for distributions over an opponent's original hand."""
import math
import numpy as np
from opponent_inference import _CARD_IDS


def prediction_metrics(hands, probabilities, true_hand, exposed):
    """Score the original hand and the top-N still-hidden cards."""
    true = frozenset(true_hand)
    hidden = true.difference(exposed)
    if not hidden:
        raise ValueError('A prediction situation must have hidden cards')
    marginals = np.zeros(40, dtype=np.float64)
    probability = 0.0
    for hand, weight in zip(hands, probabilities):
        weight = float(weight)
        if frozenset(hand) == true:
            probability += weight
        for card in hand:
            if card not in exposed:
                marginals[_CARD_IDS[card]] += weight
    ranking = sorted(range(40), key=lambda card: (-marginals[card], card))
    top = set(ranking[:len(hidden)])
    return {'log_loss': -math.log(probability) if probability > 0 else float('inf'),
            'hidden_card_accuracy': sum(_CARD_IDS[card] in top for card in hidden) / len(hidden)}

from functools import lru_cache
import torch
from opponent_inference import exact_inference_from_observation, inference_from_observation
from phase2_torch import Phase2Network, Phase2Policy, card_distribution
from symmetric_engine import play_match
from validation_schedule import validation_schedule


class _ValidationPolicy(Phase2Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.situations = []

    def __call__(self, observation, legal_actions):
        self.current_observation = observation
        self.current_bluff_rate = self.profile.rates()['bluff']
        return super().__call__(observation, legal_actions)

    def record_decision(self, inputs, action):
        self.situations.append((self.current_observation, inputs.copy(), self.current_bluff_rate))


@lru_cache(maxsize=16)
def _validation_situations(schedule_seed, target, temperature):
    """Replay one fixed validation deal using fixed policies; sample at most eight decisions."""
    deal_seed, match_target = validation_schedule(schedule_seed, None if target == 'mixed' else target, 1)[0]
    model = Phase2Network(seed=0)
    policies = [_ValidationPolicy(model, seed=201 + seat, temperature=temperature) for seat in range(2)]
    result = play_match(*policies, seed=deal_seed, target=match_target)
    offsets = [0, 0]
    situations = []
    for decision in result.decisions:
        seat = decision['player']
        obs, inputs, bluff_rate = policies[seat].situations[offsets[seat]]
        offsets[seat] += 1
        exposed = tuple(card for player, card in obs['table'] if player == 1 - seat)
        hand = result.deals[decision['hand_index']][1 - seat]
        if len(exposed) < 3:
            situations.append((obs, inputs, bluff_rate, hand, exposed))
    if len(situations) > 8:
        situations = [situations[i] for i in np.linspace(0, len(situations) - 1, 8, dtype=int)]
    return tuple(situations)


def validation_predictions(model, *, schedule_seed, target, temperature):
    """Compare predictors on fixed replayed decisions from the first validation deal."""
    totals = {name: {'log_loss': 0.0, 'hidden_card_accuracy': 0.0, 'situations': 0}
              for name in ('exact', 'weighted', 'learned')}
    for obs, inputs, bluff_rate, true_hand, exposed in _validation_situations(schedule_seed, target, temperature):
        exact = exact_inference_from_observation(obs).distribution()
        weighted = inference_from_observation(obs, bluff_rate).distribution()
        with torch.no_grad():
            logits = model.card_logits(torch.from_numpy(inputs)).double()
            hands, probabilities, _ = card_distribution(logits, obs)
        distributions = {'exact': (tuple(exact), tuple(exact.values())),
                         'weighted': (tuple(weighted), tuple(weighted.values())),
                         'learned': (hands, probabilities.tolist())}
        for name, (candidate_hands, candidate_probabilities) in distributions.items():
            result = prediction_metrics(candidate_hands, candidate_probabilities, true_hand, exposed)
            for metric in ('log_loss', 'hidden_card_accuracy'):
                totals[name][metric] += result[metric]
            totals[name]['situations'] += 1
    for result in totals.values():
        if result['situations']:
            for metric in ('log_loss', 'hidden_card_accuracy'):
                result[metric] /= result['situations']
    return totals
