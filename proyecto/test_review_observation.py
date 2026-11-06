import unittest
from symmetric_engine import play_match

class ObservationReviewTests(unittest.TestCase):
    def test_post_hand_observer_cannot_read_hidden_cards_or_bluff(self):
        class Probe:
            def __init__(self): self.records = []
            def __call__(self, obs, actions):
                return 'envido' if 'envido' in actions and not obs['pending'] else 'fold'
            def observe_hand(self, decisions, opponent): self.records.extend(decisions)
        a, b = Probe(), Probe()
        play_match(a, b, seed=2, target=1)
        self.assertTrue(a.records)
        for d in a.records:
            self.assertNotIn('hand', d['observation'])
            if not d['revealed']: self.assertIsNone(d['bluff'])
