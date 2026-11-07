import unittest
import numpy as np
from phase1_network import NetworkPolicy, GENOME_SIZE

class NetworkExplanationTests(unittest.TestCase):
    def test_probabilities_cover_only_legal_actions(self):
        policy = NetworkPolicy(np.zeros(GENOME_SIZE), seed=1, explain=True)
        obs = {'player': 1, 'hand': ((1, 'Espada'), (2, 'Oro'), (3, 'Copa')), 'available': (0, 1, 2), 'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1, 'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}
        action = policy(obs, ('card_0', 'card_1', 'truco'))
        self.assertIn(action, policy.last_probabilities)
        self.assertEqual(set(policy.last_probabilities), {'card_0', 'card_1', 'truco'})
        self.assertAlmostEqual(sum(policy.last_probabilities.values()), 1)
        self.assertIn('brava_probability', policy.last_belief)
