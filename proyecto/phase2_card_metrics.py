"""Metrics for distributions over an opponent's original hand."""
import math
import statistics
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
from phase2_torch import Phase2Network, Phase2Policy
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
def _validation_situations(schedule_seed, target, temperature, situation_count, profile_prior=20):
    """Replay fixed validation deals, sampling up to eight decisions per match."""
    model = Phase2Network(seed=0)
    situations = []
    # Every match has at least one decision, so this schedule is sufficient.
    schedule = validation_schedule(schedule_seed, None if target == 'mixed' else target, situation_count)
    for match_index, (deal_seed, match_target) in enumerate(schedule):
        policies = [_ValidationPolicy(model, seed=201 + 2 * match_index + seat,
                                      temperature=temperature, profile_prior=profile_prior) for seat in range(2)]
        result = play_match(*policies, seed=deal_seed, target=match_target, deal_partition=0)
        offsets = [0, 0]
        candidates = []
        for decision in result.decisions:
            seat = decision['player']
            obs, inputs, bluff_rate = policies[seat].situations[offsets[seat]]
            offsets[seat] += 1
            exposed = tuple(card for player, card in obs['table'] if player == 1 - seat)
            hand = result.deals[decision['hand_index']][1 - seat]
            if len(exposed) < 3:
                candidates.append((obs, inputs, bluff_rate, hand, exposed))
        count = min(8, len(candidates), situation_count - len(situations))
        situations.extend(candidates[i] for i in np.linspace(0, len(candidates) - 1, count, dtype=int))
        if len(situations) == situation_count:
            break
    return tuple(situations)


@lru_cache(maxsize=1)
def _validation_preprocessing(schedule_seed, target, temperature, situation_count, profile_prior=20):
    """Cache only fixed evidence, baseline scores and ordered hand indices.

    Keep one protocol to bound the hand-index tensors retained in a long run.
    Learned logits, probabilities and scores never enter this cache.
    """
    prepared = []
    for obs, inputs, bluff_rate, true_hand, exposed in _validation_situations(schedule_seed, target, temperature, situation_count, profile_prior):
        exact = exact_inference_from_observation(obs).distribution()
        weighted = inference_from_observation(obs, bluff_rate).distribution()
        hands = tuple(exact)
        ids = torch.tensor([[_CARD_IDS[c] for c in hand] for hand in hands])
        baseline = {name: prediction_metrics(tuple(distribution), tuple(distribution.values()), true_hand, exposed)
                    for name, distribution in (('exact', exact), ('weighted', weighted))}
        prepared.append((inputs, true_hand, exposed, hands, ids, baseline))
    return tuple(prepared)


def validation_predictions(model, *, schedule_seed, target, temperature, situation_count=400, profile_prior=20):
    """Compare predictors on fixed replayed decisions from the first validation deal."""
    if situation_count < 1:
        raise ValueError('situation_count must be positive')
    accuracies = {name: [] for name in ('exact', 'weighted', 'learned')}
    totals = {name: {'log_loss': 0.0, 'hidden_card_accuracy': 0.0, 'situations': 0}
              for name in ('exact', 'weighted', 'learned')}
    for inputs, true_hand, exposed, hands, ids, baseline in _validation_preprocessing(schedule_seed, target, temperature, situation_count, profile_prior):
        with torch.no_grad():
            logits = model.card_logits(torch.from_numpy(inputs)).double()
            # Same gather, ordered three-card sum and softmax as card_distribution;
            # its unused marginals need not be constructed for validation.
            probabilities = torch.softmax(logits[ids.to(logits.device)].sum(dim=1), dim=0)
        results = dict(baseline, learned=prediction_metrics(hands, probabilities.tolist(), true_hand, exposed))
        for name, result in results.items():
            for metric in ('log_loss', 'hidden_card_accuracy'):
                totals[name][metric] += result[metric]
            accuracies[name].append(result['hidden_card_accuracy'])
            totals[name]['situations'] += 1
    for name, result in totals.items():
        if result['situations']:
            for metric in ('log_loss', 'hidden_card_accuracy'):
                result[metric] /= result['situations']
        values = accuracies[name]
        result['accuracy_standard_error'] = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return totals
