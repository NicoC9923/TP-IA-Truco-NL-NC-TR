import tempfile
import unittest
from pathlib import Path
from opponent_profile import OpponentProfile


class PersistentProfileTests(unittest.TestCase):
    def test_observations_survive_reopen_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            profile = OpponentProfile(path)
            event = {'player': 0, 'action': 'fold', 'observation': {'pending': None}, 'revealed': False, 'bluff': None}
            profile.observe([event], 0)
            self.assertGreater(OpponentProfile(path).rates()['fold'], 0.5)
            profile.reset()
            self.assertEqual(OpponentProfile(path).rates()['fold'], 0.5)
