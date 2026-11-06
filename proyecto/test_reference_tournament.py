import tempfile
import unittest
from pathlib import Path

from engine_policies import AdaptiveUtilityPolicy, ReactivePolicy
from symmetric_engine import play_match


class ReferencePolicyTests(unittest.TestCase):
    def test_fixed_seed_repeats_full_match_without_touching_history(self):
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / 'history.json'
            history.write_text('{"parametros_ia": {"prob_farol_envido": 0.99}, "partidas": []}', encoding='utf-8')
            before = history.read_bytes()
            def run():
                return play_match(AdaptiveUtilityPolicy(12, history_path=history), ReactivePolicy(13), seed=14, target=5)
            first, second = run(), run()
            self.assertEqual(first, second)
            self.assertEqual(history.read_bytes(), before)
            self.assertIn(first.winner, (0, 1))

class TournamentTests(unittest.TestCase):
    def test_mirror_tournament_reports_each_pair_and_objective_bluffs(self):
        from reference_tournament import tournament
        from engine_policies import RandomPolicy
        results = tournament({'reactive': ReactivePolicy, 'adaptive': AdaptiveUtilityPolicy, 'random': RandomPolicy}, pairs=2, seed=42, target=5)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result['matches'], 4)
            self.assertGreaterEqual(result['win_rate'], 0)
            self.assertLessEqual(result['win_rate'], 1)
            self.assertGreater(result['margin_of_error'], 0)
            self.assertEqual(set(result['bluff_rates']), set(result['opponents']))
            for rate in result['bluff_rates'].values():
                self.assertTrue(rate is None or 0 <= rate <= 1)

    def test_rates_count_engine_bluff_flags_and_swapped_seats(self):
        from reference_tournament import tournament
        from symmetric_engine import play_mirror_pair
        class Caller:
            def __init__(self, seed):
                pass
            def __call__(self, observation, actions):
                if observation['pending']:
                    return 'no_quiero'
                if 'envido' in actions:
                    return 'envido'
                return next(action for action in actions if action.startswith('card_'))
        expected = [[], []]
        for swapped, match in enumerate(play_mirror_pair(Caller(0), Caller(1), seed=0, target=5)):
            for decision in match.decisions:
                if decision['bluff'] is not None:
                    expected[decision['player'] ^ swapped].append(decision['bluff'])
        report = tournament({'a': Caller, 'b': Caller}, pairs=1, seed=0, target=5)[0]
        for index, name in enumerate(('a', 'b')):
            self.assertEqual(report['bluff_rates'][name], sum(expected[index]) / len(expected[index]) if expected[index] else None)
