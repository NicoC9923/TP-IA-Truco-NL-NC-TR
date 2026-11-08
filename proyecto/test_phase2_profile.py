import unittest
from opponent_profile import OpponentProfile

class Phase2ProfileTests(unittest.TestCase):
    def test_no_raise_has_no_fold_evidence(self):
        profile = OpponentProfile()
        profile.observe([{'player': 1, 'action': 'card_0', 'observation': {'pending': None}, 'legal_actions': ['card_0', 'fold', 'envido']}], 1, semantics='phase2')
        self.assertEqual(profile.confidence()['fold'], 0)
        self.assertEqual(profile.rates()['fold'], 0.5)

    def test_opportunities_and_configurable_prior(self):
        profile = OpponentProfile(prior_strength=2)
        decision = {'player': 1, 'action': 'no_quiero', 'observation': {'pending': {'kind': 'truco'}}, 'legal_actions': ['quiero', 'no_quiero', 'fold']}
        profile.observe([decision], 1, semantics='phase2')
        self.assertEqual(profile.confidence()['fold'], 1)
        self.assertEqual(profile.confidence()['truco_accepted'], 1)
        self.assertAlmostEqual(profile.rates()['truco_accepted'], 1/3)
        profile.observe([decision], 1, semantics='phase2')
        self.assertEqual(profile.confidence()['fold'], 2)

if __name__ == '__main__': unittest.main()
