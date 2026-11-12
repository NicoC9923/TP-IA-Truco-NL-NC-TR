import unittest
import numpy as np
from opponent_profile import OpponentProfile
from phase2_network import encode_observation, INDEX, INPUT_SIZE, EVENT_SIZE

class Phase2EncoderTests(unittest.TestCase):
    def setUp(self):
        self.obs = {'player': 0, 'hand': ((1, 'Espada'), (2, 'Oro'), (3, 'Copa')), 'available': (0, 1, 2), 'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1, 'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}
        self.profile = OpponentProfile()

    def encode(self, obs, **kwargs):
        return encode_observation(obs, self.profile, ('card_0', 'truco'), **kwargs)

    def test_shape_slots_and_legal_actions(self):
        x = self.encode(self.obs)
        self.assertEqual(x.shape, (INPUT_SIZE,))
        self.assertEqual(x[INDEX['own_cards']].sum(), 3)
        self.assertEqual(x[INDEX['legal_actions']].sum(), 2)

    def test_events_order_and_private_opponent_slot(self):
        a = dict(self.obs, history=((1, 'card_0'), (0, 'truco')), table=((1, (7, 'Oro')),))
        b = dict(a, history=((1, 'card_2'), (0, 'truco')), opponent_hand=((4, 'Oro'),), style='bluffer', identity='x')
        np.testing.assert_array_equal(self.encode(a), self.encode(b))
        c = dict(a, history=((0, 'truco'), (1, 'card_0')))
        self.assertFalse(np.array_equal(self.encode(a)[INDEX['events']], self.encode(c)[INDEX['events']]))
        self.assertEqual(len(self.encode(a)[INDEX['events']]), 24 * EVENT_SIZE)

    def test_card_slots_and_public_card_identity(self):
        x = self.encode(self.obs)
        swapped = dict(self.obs, hand=(self.obs['hand'][1], self.obs['hand'][0], self.obs['hand'][2]))
        y = self.encode(swapped)
        np.testing.assert_array_equal(x[70:110], y[110:150])
        obs = dict(self.obs, history=((1, 'card_2'),), table=((1, (7, 'Oro')),))
        row = self.encode(obs)[INDEX['events']][:EVENT_SIZE]
        self.assertEqual(row[12:52].sum(), 1)
        self.assertEqual(row[1], -1)

    def test_envido_declarations_are_on_resolution_event(self):
        obs = dict(self.obs, history=((0, 'envido'), (1, 'quiero')),
                   envido_reveal={'declarations': ((0, 30), (1, 'son buenas'))})
        row = self.encode(obs)[INDEX['events']].reshape(24, EVENT_SIZE)[1]
        self.assertAlmostEqual(row[55], 30 / 33)
        self.assertEqual(row[58], 1)

    def test_buenas_flags_distinguish_match_lengths(self):
        short = dict(self.obs, score=(5, 5), target=15)
        long = dict(self.obs, score=(20, 20), target=30)
        np.testing.assert_array_equal(self.encode(short)[INDEX['buenas']], [1, 1])
        np.testing.assert_array_equal(self.encode(long)[INDEX['buenas']], [1, 1])
        long_malas = dict(self.obs, score=(5, 10), target=30)
        np.testing.assert_array_equal(self.encode(long_malas)[INDEX['buenas']], [0, 0])

    def test_overflow_rejected(self):
        obs = dict(self.obs, history=((0, 'truco'),) * 25)
        with self.assertRaises(ValueError): self.encode(obs)

    def test_weighted_switch(self):
        obs = dict(self.obs, history=((1, 'truco'),))
        enabled = self.encode(obs)
        disabled = self.encode(obs, weighted_beliefs=False)
        self.assertTrue(np.any(enabled[INDEX['weighted_belief']] != 0))
        self.assertTrue(np.all(disabled[INDEX['weighted_belief']] == 0))
        np.testing.assert_array_equal(enabled[INDEX['exact_belief']], disabled[INDEX['exact_belief']])

    def test_confidence_is_normalized_by_prior(self):
        self.profile.counts['fold'] = [2, 20]
        x = self.encode(self.obs)
        self.assertAlmostEqual(x[INDEX['confidence']][1], .5)

    def test_weighted_belief_equals_exact_at_bluff_one(self):
        class CertainProfile(OpponentProfile):
            def rates(self, adapt=True):
                rates = super().rates(adapt)
                rates['bluff'] = 1.0
                return rates
        obs = dict(self.obs, history=((1, 'truco'),))
        x = encode_observation(obs, CertainProfile(), ('card_0',))
        np.testing.assert_array_equal(x[INDEX['exact_belief']], x[INDEX['weighted_belief']])
