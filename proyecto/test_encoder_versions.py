import hashlib
import unittest
import numpy as np
import truco_reglas as rules
from opponent_profile import OpponentProfile
from phase1_network import encode_observation, NetworkPolicy, GENOME_SIZE


def observation():
    return {'player': 1, 'hand': ((1, 'Espada'), (2, 'Oro'), (3, 'Copa')), 'available': (0, 1, 2), 'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1, 'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}


class EncoderTests(unittest.TestCase):
    def test_v1_golden(self):
        obs = observation()
        for available, expected in zip(((0, 1, 2), (1, 2), ()), ('c14a4fedf50845f6c8403d5e1199477217ce2d30ec158caeddd0cd5aa005901f', '75d9e506d8cf38ac32ca0838e7b08b0d45592e8183ef2e3189074986e58460d2', 'a3e93030bf647f72bf58405d7c6e6c79e862438c083990a7bac089350be5a911')):
            obs['available'] = available
            self.assertEqual(hashlib.sha256(encode_observation(obs, OpponentProfile(), encoder='v1').tobytes()).hexdigest(), expected)

    def test_all_cards_use_rule_strength_and_only_held_features_change(self):
        obs = observation()
        for value in rules.VALORES:
            for suit in rules.PALOS:
                obs['hand'] = ((value, suit), (2, 'Oro'), (3, 'Copa'))
                a = encode_observation(obs, OpponentProfile(), encoder='v1')
                b = encode_observation(obs, OpponentProfile(), encoder='v2')
                self.assertEqual(b[0], rules.valor_carta_truco(value, suit) / 13)
                np.testing.assert_array_equal(np.delete(a, [0, 4, 8]), np.delete(b, [0, 4, 8]))

    def test_unknown_encoder_rejected(self):
        with self.assertRaises(ValueError):
            encode_observation(observation(), OpponentProfile(), encoder='v3')
        with self.assertRaises(ValueError):
            NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v3')
