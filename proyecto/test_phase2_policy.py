import os
import tempfile
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from phase1_network import NetworkPolicy, GENOME_SIZE
if torch is not None:
    from phase2_torch import Phase2Network, Phase2Policy
from symmetric_engine import play_match, play_mirror_pair

@unittest.skipUnless(torch is not None, 'PyTorch is optional')
class Phase2PolicyTests(unittest.TestCase):
    def test_random_policies_play_mirror_matches(self):
        for opponent in (Phase2Policy(seed=2), NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v1'), NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v3')):
            pair = play_mirror_pair(Phase2Policy(seed=1), opponent, seed=44, target=3)
            for match in pair:
                self.assertTrue(match.decisions)
                self.assertTrue(all(d['action'] in d['legal_actions'] for d in match.decisions))

    def test_temperature_on_fixed_values(self):
        model = Phase2Network()
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
            model.layers[-1].bias[0] = 1
        obs = {'player': 0, 'hand': ((1, 'Espada'), (2, 'Oro'), (3, 'Copa')), 'available': (0, 1, 2), 'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1, 'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}
        low = Phase2Policy(model=model, temperature=.01)
        high = Phase2Policy(model=model, temperature=100)
        low(obs, ('card_0', 'card_1')); high(obs, ('card_0', 'card_1'))
        self.assertGreater(low.last_probabilities['card_0'], .999)
        self.assertAlmostEqual(high.last_probabilities['card_0'], .5, delta=.01)

    def test_weights_round_trip(self):
        model = Phase2Network()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'weights.pt')
            model.save(path)
            raw = torch.load(path, map_location='cpu', weights_only=True)
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in raw.values()))
            other = Phase2Network.load(path)
            for a, b in zip(model.parameters(), other.parameters()):
                self.assertTrue(torch.equal(a, b))

    def test_incremental_matches_fresh_encoding(self):
        from phase2_network import encode_observation
        for target in (15, 30):
            for weighted in (True, False):
                policy = Phase2Policy(seed=3, weighted_beliefs=weighted)
                cases = set()
                checked_hands = set()
                class CheckingPolicy:
                    def __call__(self, obs, legal):
                        expected = encode_observation(obs, policy.profile, legal, weighted_beliefs=weighted)
                        actual = policy.encode(obs, legal)
                        np.testing.assert_array_equal(actual, expected)
                        checked_hands.add(obs['hand'])
                        if obs['envido_reveal']:
                            cases.add('reveal')
                            if any(value == 'son buenas' for _, value in obs['envido_reveal']['declarations']):
                                cases.add('son buenas')
                        return policy(obs, legal)
                    def observe_hand(self, decisions, opponent):
                        if any(d['action'] == 'fold' for d in decisions):
                            cases.add('fold')
                        policy.observe_hand(decisions, opponent)
                for seed in range(12):
                    play_match(CheckingPolicy(), Phase2Policy(seed=seed + 50), seed=seed, target=target)
                    if len(checked_hands) >= 3 and cases == {'reveal', 'son buenas', 'fold'}:
                        break
                self.assertGreaterEqual(len(checked_hands), 3)
                self.assertEqual(cases, {'reveal', 'son buenas', 'fold'})

    def test_same_seed_initializes_same_weights_and_actions(self):
        first, second = Phase2Policy(seed=123), Phase2Policy(seed=123)
        for a, b in zip(first.model.parameters(), second.model.parameters()):
            self.assertTrue(torch.equal(a, b))
        obs = {'player': 0, 'hand': ((1, 'Espada'), (2, 'Oro'), (3, 'Copa')), 'available': (0, 1, 2), 'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1, 'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}
        for _ in range(10):
            self.assertEqual(first(obs, ('card_0', 'card_1')), second(obs, ('card_0', 'card_1')))

    def test_style_offset_and_legal_mask(self):
        model = Phase2Network()
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
        obs = {'player': 0, 'hand': ((1, 'Espada'), (2, 'Oro'), (3, 'Copa')), 'available': (0, 1, 2), 'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1, 'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}
        policy = Phase2Policy(model=model, style='aggressive', style_strength=.1, temperature=.1)
        policy(obs, ('card_0', 'real_envido'))
        self.assertAlmostEqual(policy.last_probabilities['real_envido'] / policy.last_probabilities['card_0'], np.e, places=6)
        self.assertNotIn('card_1', policy.last_probabilities)


_GREEDY_OBS = {'player': 0, 'hand': ((1, 'Espada'), (2, 'Oro'), (3, 'Copa')), 'available': (0, 1, 2), 'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1, 'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}


def _zero_model(**biases):
    from phase1_network import ACTIONS
    model = Phase2Network()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for action, value in biases.items():
            model.layers[-1].bias[ACTIONS.index(action)] = value
    return model


@unittest.skipUnless(torch is not None, 'PyTorch is optional')
class Phase2GreedyTests(unittest.TestCase):
    def test_greedy_picks_highest_legal_q_even_when_a_higher_illegal_one_exists(self):
        policy = Phase2Policy(model=_zero_model(card_1=1.0, card_2=3.0, envido=0.5), greedy=True)
        for _ in range(5):
            self.assertEqual(policy(_GREEDY_OBS, ('card_0', 'card_1', 'envido')), 'card_1')

    def test_greedy_ties_go_to_lowest_action_index_whatever_the_legal_order(self):
        policy = Phase2Policy(model=_zero_model(), greedy=True)
        self.assertEqual(policy(_GREEDY_OBS, ('envido', 'card_2', 'card_1')), 'card_1')
        self.assertEqual(policy(_GREEDY_OBS, ('card_2', 'envido', 'card_1')), 'card_1')

    def test_greedy_does_not_consume_the_rng_and_records_decisions(self):
        policy = Phase2Policy(model=_zero_model(card_0=1.0), seed=7, greedy=True)
        recorded = []
        policy.record_decision = lambda inputs, action: recorded.append(action)
        state = policy.rng.bit_generator.state
        self.assertEqual(policy(_GREEDY_OBS, ('card_0', 'card_1')), 'card_0')
        self.assertEqual(policy.rng.bit_generator.state, state)
        self.assertEqual(recorded, ['card_0'])
        self.assertEqual(policy.last_probabilities, {'card_0': 1.0, 'card_1': 0.0})

    def test_sampling_still_uses_one_rng_choice_per_decision(self):
        policy = Phase2Policy(model=_zero_model(card_0=1.0), seed=7, temperature=.5)
        rng = np.random.default_rng(7)
        weights = np.exp(np.array([1.0, 0.0]) / .5)
        probabilities = weights / weights.sum()
        for _ in range(6):
            expected = ('card_0', 'card_1')[rng.choice(2, p=probabilities)]
            self.assertEqual(policy(_GREEDY_OBS, ('card_0', 'card_1')), expected)

    def test_greedy_still_validates_temperature_when_not_greedy(self):
        with self.assertRaises(ValueError):
            Phase2Policy(temperature=0)
        Phase2Policy(greedy=True)  # greedy does not need a temperature
