"""Reusing exact v3 summaries for an identical weighted posterior is byte-exact."""
import copy
import random
import unittest
import numpy as np
from opponent_inference import IncrementalInference
from opponent_profile import OpponentProfile
from phase1_network import ACTIONS, encode_observation as encode_v3
import phase2_network
from phase2_network import INPUT_SIZE, INDEX, MAX_EVENTS, _card_id, CATEGORIES, _FixedInference
from opponent_inference import inference_from_observation
from symmetric_engine import play_match


def reference_encode(obs, profile, legal_actions, *, weighted_beliefs=True, exact_inference=None, weighted_inference=None):
    """Frozen copy of the encoder before summary reuse."""
    if len(obs['history']) > MAX_EVENTS:
        raise ValueError('Current hand event capacity exceeded')
    exact = (exact_inference.update(obs, 1.0) if exact_inference is not None
             else inference_from_observation(obs, 1.0))
    summary = encode_v3(obs, profile, _FixedInference(exact), encoder='v3')
    result = np.zeros(INPUT_SIZE, dtype=np.float32)
    result[INDEX['summary']] = summary
    result[INDEX['exact_belief']] = summary[50:55]
    if weighted_beliefs:
        weighted = encode_v3(obs, profile, weighted_inference, encoder='v3')
        result[INDEX['weighted_belief']] = weighted[50:55]
    result[INDEX['confidence']] = [n / (n + profile.prior_strength) for n in profile.confidence().values()]
    player = obs['player']
    result[INDEX['buenas']] = [obs['target'] != 30 or obs['score'][player] >= 15,
                               obs['target'] != 30 or obs['score'][1-player] >= 15]
    for slot, card in enumerate(obs['hand']):
        result[70 + slot * 40 + _card_id(card)] = 1
    for action in legal_actions:
        result[190 + ACTIONS.index(action)] = 1
    rows = result[INDEX['events']].reshape(MAX_EVENTS, 59)
    cards_played = 0
    pending_envido = False
    for row, (actor, action) in zip(rows, obs['history']):
        row[0] = 1
        row[1] = 1 if actor == player else -1
        category = 'card' if action.startswith('card_') else action
        row[2 + CATEGORIES.index(category)] = 1
        row[52 + min(cards_played // 2, 2)] = 1
        if category == 'card':
            table_actor, card = obs['table'][cards_played]
            row[12 + _card_id(card)] = 1
            cards_played += 1
        if action in ('envido', 'real_envido', 'falta_envido'):
            pending_envido = True
        elif pending_envido and action in ('quiero', 'no_quiero', 'fold'):
            if action == 'quiero' and obs.get('envido_reveal'):
                for seat, declaration in obs['envido_reveal']['declarations']:
                    relative = 0 if seat == player else 1
                    if declaration == 'son buenas':
                        row[57 + relative] = 1
                    else:
                        row[55 + relative] = declaration / 33
            pending_envido = False
    return result


class Comparing:
    """Random seat that encodes every decision both ways with separate posteriors."""
    def __init__(self, test, seed, prior, weighted=True, incremental=True):
        self.test, self.rng = test, random.Random(seed)
        self.profile = OpponentProfile(prior_strength=prior)
        self.weighted, self.incremental = weighted, incremental
        self.states = [(IncrementalInference(), IncrementalInference()) for _ in range(2)]
        self.count = 0

    def __call__(self, obs, legal):
        outputs = []
        for function, (exact, weighted) in zip((phase2_network.encode_observation, reference_encode), self.states):
            kwargs = dict(weighted_beliefs=self.weighted)
            if self.incremental:
                kwargs.update(exact_inference=exact, weighted_inference=weighted)
            outputs.append(function(copy.deepcopy(obs), self.profile, legal, **kwargs))
        self.test.assertEqual(outputs[0].tobytes(), outputs[1].tobytes())
        self.count += 1
        return self.rng.choice(legal)

    def observe_hand(self, decisions, opponent):
        self.profile.observe(decisions, opponent, semantics='phase2')
        for exact, weighted in self.states:
            exact.reset()
            weighted.reset()


class EncoderReuseTests(unittest.TestCase):
    def test_matches_with_truco_calls_and_reveals_encode_identically(self):
        total = 0
        for seed in range(30):
            options = dict(prior=(5, 20, 1)[seed % 3], weighted=seed % 7 != 6, incremental=seed % 5 != 4)
            a, b = Comparing(self, seed, **options), Comparing(self, seed + 1000, **options)
            play_match(a, b, seed=seed, target=(15, 30)[seed % 2])
            total += a.count + b.count
        self.assertGreater(total, 1000)


if __name__ == '__main__':
    unittest.main()
