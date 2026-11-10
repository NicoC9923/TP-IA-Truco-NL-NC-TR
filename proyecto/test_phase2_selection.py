import json
import tempfile
import unittest
from pathlib import Path

from phase2_game_policy import GamePhase2Policy
from phase2_loader import load_phase2_metadata, load_phase2_policy
from phase2_torch import Phase2Network


class SidecarSelectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / 'weights.pt'
        Phase2Network().save(self.path)

    def write(self, **extra):
        (self.dir / 'reigning_champion.json').write_text(json.dumps({'temperature': .05, 'profile_prior': 5, **extra}))

    def test_old_sidecar_without_selection_stays_sampled(self):
        self.write()
        self.assertFalse(load_phase2_policy(self.path, seed=1).greedy)

    def test_greedy_sidecar_loads_a_greedy_policy(self):
        self.write(selection='greedy')
        policy = load_phase2_policy(self.path, seed=1)
        self.assertTrue(policy.greedy)
        self.assertEqual(policy.profile.prior_strength, 5)

    def test_sampled_selection_is_not_greedy(self):
        self.write(selection='sampled')
        self.assertFalse(load_phase2_policy(self.path, seed=1).greedy)

    def test_unknown_selection_is_rejected(self):
        for bad in ('argmax', 1, None, True):
            with self.assertRaises(ValueError):
                load_phase2_metadata(self.path, {'temperature': .05, 'profile_prior': 5, 'selection': bad})

    def test_game_policy_keeps_greedy_and_reports_one_hot_choice(self):
        self.write(selection='greedy')
        policy = GamePhase2Policy.load(self.path, seed=3)
        self.assertTrue(policy.greedy)
        obs = {'player': 1, 'hand': ((1, 'Espada'), (7, 'Oro'), (4, 'Copa')), 'available': (0, 1, 2),
               'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1,
               'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}
        action = policy(obs, ('card_0', 'card_1', 'card_2'))
        self.assertEqual(policy.last_probabilities[action], 1.0)


if __name__ == '__main__':
    unittest.main()
