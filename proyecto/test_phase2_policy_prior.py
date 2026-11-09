import unittest
import numpy as np
from phase2_torch import Phase2Policy
from test_encoder_versions import observation


class PolicyPriorTests(unittest.TestCase):
    def test_rates_and_confidence_follow_prior(self):
        decision = {'player': 0, 'action': 'fold', 'observation': {'pending': {'kind': 'truco'}}, 'legal_actions': ['fold', 'quiero']}
        for prior, rate, confidence in ((5, 3.5 / 6, 1 / 6), (20, 11 / 21, 1 / 21)):
            with self.subTest(prior=prior):
                policy = Phase2Policy(seed=1, profile_prior=prior)
                before = policy.encode(observation(), ('card_0',))
                np.testing.assert_array_equal(before[55:59], [0.5] * 4)
                np.testing.assert_array_equal(before[64:68], [0] * 4)
                policy.observe_hand([decision], 0)
                after = policy.encode(observation(), ('card_0',))
                self.assertAlmostEqual(after[56], rate)
                self.assertAlmostEqual(after[65], confidence)

    def test_invalid_priors_are_rejected(self):
        from opponent_profile import OpponentProfile
        for prior in (float('nan'), float('inf'), float('-inf'), 0, -1):
            with self.subTest(prior=prior):
                with self.assertRaises(ValueError):
                    Phase2Policy(profile_prior=prior)
                with self.assertRaises(ValueError):
                    OpponentProfile(prior_strength=prior)

    def test_adaptation_off_is_identical_with_evidence(self):
        decision = {'player': 0, 'action': 'fold', 'observation': {'pending': {'kind': 'truco'}}, 'legal_actions': ['fold', 'quiero']}
        inputs = []
        for prior in (5, 20):
            policy = Phase2Policy(seed=1, profile_prior=prior, adapt=False)
            policy.observe_hand([decision], 0)
            inputs.append(policy.encode(observation(), ('card_0',)))
        np.testing.assert_array_equal(inputs[0], inputs[1])
        np.testing.assert_array_equal(inputs[0][55:59], [0.5] * 4)
        np.testing.assert_array_equal(inputs[0][64:68], [0] * 4)

    def test_existing_profile_and_explicit_prior_must_agree(self):
        from opponent_profile import OpponentProfile
        profile = OpponentProfile(prior_strength=5)
        self.assertIs(Phase2Policy(profile=profile).profile, profile)
        self.assertIs(Phase2Policy(profile=profile, profile_prior=5).profile, profile)
        with self.assertRaises(ValueError):
            Phase2Policy(profile=profile, profile_prior=20)

    def test_default_encoding_matches_original_encoder(self):
        from opponent_profile import OpponentProfile
        from phase2_network import encode_observation
        profile = OpponentProfile()
        decision = {'player': 0, 'action': 'fold', 'observation': {'pending': {'kind': 'truco'}}, 'legal_actions': ['fold', 'quiero']}
        profile.observe([decision], 0, semantics='phase2')
        policy = Phase2Policy(seed=1)
        policy.observe_hand([decision], 0)
        expected = encode_observation(observation(), profile, ('card_0',))
        np.testing.assert_array_equal(policy.encode(observation(), ('card_0',)), expected)
