import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from phase1_network import NetworkPolicy, GENOME_SIZE
from phase2_torch import Phase2Network, Phase2Policy
from symmetric_engine import play_match, play_mirror_pair

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
        for weighted in (True, False):
            policy = Phase2Policy(seed=3, weighted_beliefs=weighted)
            checked_hands = set()
            class CheckingPolicy:
                def __call__(self, obs, legal):
                    expected = encode_observation(obs, policy.profile, legal, weighted_beliefs=weighted)
                    actual = policy.encode(obs, legal)
                    np.testing.assert_array_equal(actual, expected)
                    checked_hands.add(obs['hand'])
                    return policy(obs, legal)
                def observe_hand(self, decisions, opponent):
                    policy.observe_hand(decisions, opponent)
            for seed in (15, 28, 43):
                play_match(CheckingPolicy(), Phase2Policy(seed=8), seed=seed, target=5)
            self.assertGreaterEqual(len(checked_hands), 3)

    def test_torch_optional(self):
        script = "import sys; sys.modules['torch']=None; import symmetric_engine, phase1_network, phase1_trainer, phase2_network, partida_truco"
        result = subprocess.run([sys.executable, '-c', script], cwd=os.path.dirname(__file__), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
