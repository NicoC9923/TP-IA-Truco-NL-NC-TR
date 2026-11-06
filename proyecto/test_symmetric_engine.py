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
