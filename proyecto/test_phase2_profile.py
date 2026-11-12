import unittest
from opponent_profile import OpponentProfile

class Phase2ProfileTests(unittest.TestCase):
    def test_no_raise_has_no_fold_evidence(self):
        profile = OpponentProfile()
        profile.observe([{'player': 1, 'action': 'card_0', 'observation': {'pending': None}, 'legal_actions': ['card_0', 'fold', 'envido']}], 1, semantics='phase2')
        self.assertEqual(profile.confidence()['fold'], 0)
        self.assertEqual(profile.rates()['fold'], 0.5)

    def test_envido_counted_once_per_hand(self):
        profile = OpponentProfile()
        decisions = [
            {'player': 1, 'action': 'card_0', 'hand_index': 0, 'observation': {'pending': None}, 'legal_actions': ['card_0', 'envido']},
            {'player': 1, 'action': 'quiero', 'hand_index': 0, 'observation': {'pending': {'kind': 'truco'}}, 'legal_actions': ['quiero', 'envido']},
        ]
        profile.observe(decisions, 1, semantics='phase2')
        self.assertEqual(profile.confidence()['envido_called'], 1)
        self.assertAlmostEqual(profile.rates()['envido_called'], 10 / 21)

    def test_opportunities_and_configurable_prior(self):
        profile = OpponentProfile(prior_strength=2)
        decision = {'player': 1, 'action': 'no_quiero', 'observation': {'pending': {'kind': 'truco'}}, 'legal_actions': ['quiero', 'no_quiero', 'fold']}
        profile.observe([decision], 1, semantics='phase2')
        self.assertEqual(profile.confidence()['fold'], 1)
        self.assertEqual(profile.confidence()['truco_accepted'], 1)
        self.assertAlmostEqual(profile.rates()['truco_accepted'], 1/3)
        profile.observe([decision], 1, semantics='phase2')
        self.assertEqual(profile.confidence()['fold'], 2)

    def test_response_to_envido_is_not_opening_opportunity(self):
        profile = OpponentProfile()
        response = {'player': 1, 'action': 'real_envido', 'hand_index': 0,
                    'observation': {'pending': {'kind': 'envido'}},
                    'legal_actions': ['quiero', 'real_envido']}
        profile.observe([response], 1, semantics='phase2')
        self.assertEqual(profile.confidence()['envido_called'], 0)
        opening = dict(response, hand_index=1, observation={'pending': None})
        profile.observe([opening], 1, semantics='phase2')
        self.assertEqual(profile.confidence()['envido_called'], 1)
        self.assertAlmostEqual(profile.rates()['envido_called'], 11/21)

    def test_no_quiero_is_non_acceptance_but_not_mazo(self):
        profile = OpponentProfile(prior_strength=2)
        response = {'player': 1, 'action': 'no_quiero', 'observation': {'pending': {'kind': 'truco'}}, 'legal_actions': ['quiero', 'no_quiero', 'fold']}
        profile.observe([response], 1, semantics='phase2')
        self.assertAlmostEqual(profile.rates()['fold'], 1/3)
        self.assertAlmostEqual(profile.rates()['truco_accepted'], 1/3)
        response['action'] = 'fold'
        profile.observe([response], 1, semantics='phase2')
        self.assertAlmostEqual(profile.rates()['fold'], 1/2)
        self.assertAlmostEqual(profile.rates()['truco_accepted'], 1/4)

if __name__ == '__main__': unittest.main()
