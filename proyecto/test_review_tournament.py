import unittest
from types import SimpleNamespace
from unittest.mock import patch
from reference_tournament import tournament
class TournamentReviewTests(unittest.TestCase):
    def test_margin_uses_mirror_pair_variation(self):
        matches = [(SimpleNamespace(winner=0, decisions=[]), SimpleNamespace(winner=1, decisions=[])), (SimpleNamespace(winner=1, decisions=[]), SimpleNamespace(winner=0, decisions=[]))]
        with patch('reference_tournament.play_mirror_pair', side_effect=matches):
            row = tournament({'a': lambda s: None, 'b': lambda s: None}, pairs=2)[0]
        self.assertAlmostEqual(row['win_rate'], .5)
        self.assertAlmostEqual(row['margin_of_error'], .98)
        self.assertEqual(row['confidence_interval'], [-.48, 1.48])
