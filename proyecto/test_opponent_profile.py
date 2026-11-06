import unittest
from opponent_profile import OpponentProfile

class ProfileTests(unittest.TestCase):
    def test_neutral_prior_and_observed_rates(self):
        profile = OpponentProfile()
        self.assertEqual(profile.rates(), dict(bluff=0.5, fold=0.5, truco_accepted=0.5, envido_called=0.5))
        decision = {'player': 1, 'action': 'fold', 'observation': {'pending': None}, 'revealed': False, 'bluff': None}
        profile.observe([decision], 1)
        self.assertLess(profile.rates()['fold'], 0.53)
        profile.observe([decision] * 1000, 1)
        self.assertGreater(profile.rates()['fold'], 0.98)
        self.assertEqual(profile.rates(adapt=False)['fold'], 0.5)

    def test_bluffs_only_count_revealed_hands(self):
        profile = OpponentProfile()
        call = {'player': 1, 'action': 'truco', 'observation': {'pending': None}, 'revealed': False, 'bluff': True}
        profile.observe([call] * 100, 1)
        self.assertEqual(profile.rates()['bluff'], 0.5)
        profile.observe([dict(call, revealed=True)] * 100, 1)
        self.assertGreater(profile.rates()['bluff'], 0.9)
    def test_multiple_calls_in_one_revealed_hand_count_once(self):
        profile = OpponentProfile()
        calls = [dict(player=1, action=action, observation={'pending': None}, revealed=True, bluff=True, hand_index=0) for action in ('truco', 'retruco')]
        profile.observe(calls, 1)
        self.assertAlmostEqual(profile.rates()['bluff'], 11 / 21)
