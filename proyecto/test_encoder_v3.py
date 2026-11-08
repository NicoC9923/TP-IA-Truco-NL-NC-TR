import hashlib
import unittest
import numpy as np
from opponent_profile import OpponentProfile
from phase1_network import encode_observation, NetworkPolicy, GENOME_SIZE
from test_encoder_versions import observation

class EncoderV3Tests(unittest.TestCase):
    def encode(self, obs):
        return encode_observation(obs, OpponentProfile(), encoder='v3')

    def test_index_map_and_empty_table(self):
        x = self.encode(observation())
        self.assertEqual(len(x), 64)
        self.assertEqual(x[0], 1)
        np.testing.assert_array_equal(x[12:30], np.zeros(18))
        self.assertEqual(x[59], 0)
        np.testing.assert_array_equal(x[60:], np.zeros(4))
        obs = observation()
        obs['table'] = ((0, (4, 'Copa')),)
        x = self.encode(obs)
        np.testing.assert_array_equal(x[12:15], [0, 1, -1])
        np.testing.assert_array_equal(x[15:18], [0, 0, 0])

    def test_exact_results_and_parda_for_either_mano(self):
        for mano in (0, 1):
            obs = observation()
            obs.update(hand=((1, 'Espada'), (3, 'Copa'), (4, 'Basto')), mano=mano,
                       table=((0, (3, 'Oro')),))
            np.testing.assert_array_equal(self.encode(obs)[52:55], [1, 0, -1])
            obs.update(tricks=(None,), table=((1, (2, 'Oro')), (0, (2, 'Copa')), (0, (3, 'Oro'))))
            x = self.encode(obs)
            self.assertEqual(x[30], 0)
            self.assertEqual(x[59], 0.5)
            np.testing.assert_array_equal(x[52:55], [1, 0, -1])

    def test_expectation_and_unavailable_slots(self):
        obs = observation()
        obs.update(hand=((1, 'Espada'), (4, 'Oro'), (3, 'Copa')), available=(0, 1))
        x = self.encode(obs)
        self.assertAlmostEqual(x[52], 1)
        # Three remaining fours tie, all 34 other unseen cards beat this four.
        self.assertAlmostEqual(x[53], -34 / 37)
        self.assertEqual(x[54], 0)
        self.assertEqual(x[10], 0)

    def test_hidden_cards_do_not_affect_encoding(self):
        obs = observation()
        obs['opponent_hand'] = ((1, 'Basto'), (4, 'Oro'), (5, 'Copa'))
        a = self.encode(obs)
        obs['opponent_hand'] = ((7, 'Oro'), (6, 'Basto'), (12, 'Oro'))
        np.testing.assert_array_equal(a, self.encode(obs))

    def test_v2_golden(self):
        obs = observation()
        for available, expected in zip(((0, 1, 2), (1, 2), ()), ('46a6c557314ef30e20c2ad6d4976245ee389921cb0c1c8d7d3f77325ebba37df', 'fc111a786573f8842ab96afedd0a196d5113e53c12fd43d40d1a708e0df1cf16', 'a0560e49e8ea89439d53b8924ba9967927335f37f41f100e9cf68f0b0c6e041d')):
            obs['available'] = available
            self.assertEqual(hashlib.sha256(encode_observation(obs, OpponentProfile(), encoder='v2').tobytes()).hexdigest(), expected)

    def test_policy_accepts_v3(self):
        self.assertEqual(NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v3').encoder, 'v3')
