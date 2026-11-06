import unittest

from symmetric_engine import play_match


class EngineTests(unittest.TestCase):
    def test_match_returns_winner_score_and_decisions(self):
        def fold(observation, actions):
            return 'fold'
        result = play_match(fold, fold, seed=7, target=2)
        self.assertEqual(result.winner, 1)
        self.assertEqual(result.points, (0, 2))
        self.assertEqual(result.decisions[0]['action'], 'fold')

    def test_card_play_is_seeded_and_hides_opponent_cards(self):
        def first_card(observation, actions):
            self.assertNotIn('opponent_hand', observation)
            return next(a for a in actions if a.startswith('card_'))
        one = play_match(first_card, first_card, seed=4, target=3)
        two = play_match(first_card, first_card, seed=4, target=3)
        self.assertEqual(one, two)
        self.assertTrue(any(d['action'].startswith('card_') for d in one.decisions))

    def test_rejects_illegal_action(self):
        with self.assertRaises(ValueError):
            play_match(lambda o, a: 'cheat', lambda o, a: 'fold')

    def test_starting_score_and_mirror_deals(self):
        from symmetric_engine import play_mirror_pair
        fold = lambda o, a: 'fold'
        first, second = play_mirror_pair(fold, fold, seed=8, target=15, starting_score=(13, 14))
        self.assertEqual(first.points, (13, 16))
        self.assertEqual(first.deals[0], second.deals[0][::-1])

    def test_truco_raise_and_rejection_scores_prior_level(self):
        def scripted(observation, actions):
            pending = observation['pending']
            if pending is None:
                return 'truco' if 'truco' in actions else 'card_0'
            return 'retruco' if 'retruco' in actions else 'no_quiero'
        result = play_match(scripted, scripted, seed=1, target=2)
        self.assertEqual(result.points, (0, 2))

    def test_runner_reports_throughput_and_mirrors(self):
        import subprocess
        import sys
        from pathlib import Path
        output = subprocess.check_output([sys.executable, str(Path(__file__).with_name('match_runner.py')), '--matches', '2', '--mirror'], text=True)
        self.assertIn('matches/second', output)
        self.assertIn('16 threads', output)
        self.assertIn('mirror', output)

    def test_mirror_swaps_mano_as_well_as_cards(self):
        from symmetric_engine import play_mirror_pair
        fold = lambda o, a: 'fold'
        first, second = play_mirror_pair(fold, fold, seed=9, target=2)
        self.assertEqual(first.decisions[0]['player'], 0)
        self.assertEqual(second.decisions[0]['player'], 1)
