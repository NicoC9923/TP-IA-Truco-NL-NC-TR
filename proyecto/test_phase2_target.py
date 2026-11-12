"""Run 3 blended per-hand target (ADR 0062): result and match equity after the decision's hand."""
import random
import unittest
from types import SimpleNamespace
import numpy as np
import phase2_equity as equity
import phase2_target as target_module
from symmetric_engine import play_match


def toy_table():
    """equity(own, opp, target, m) = clip(0.5 + (own - opp) / 40 + 0.02 * m): easy to compute by hand."""
    tables = {}
    for target in (15, 30):
        grid = np.zeros((2, target, target))
        for m in (0, 1):
            for own in range(target):
                for opp in range(target):
                    grid[m][own][opp] = min(1, max(0, 0.5 + (own - opp) / 40 + 0.02 * m))
        tables[target] = grid
    return equity.EquityTable(tables)


def decision(player, hand, score, mano, target=15):
    return {'player': player, 'hand_index': hand,
            'observation': {'score': score, 'mano': mano, 'target': target}}


def fake_result(decisions, winner):
    return SimpleNamespace(decisions=decisions, winner=winner)


def expected(blend, result, eq):
    return (1 - blend) * result + blend * (2 * eq - 1)


class RandomPolicy:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def __call__(self, observation, legal):
        return self.rng.choice(legal)


class TargetTest(unittest.TestCase):
    table = toy_table()

    def three_hand_match(self, start=(0, 0), winner=0):
        # Hand 0: mano 0, ends 3-1. Hand 1: mano 1, ends 3-6. Hand 2 (last): mano 0.
        a, b = start
        return fake_result([
            decision(0, 0, (a, b), 0), decision(1, 0, (a, b), 0), decision(0, 0, (a + 3, b), 0),
            decision(1, 1, (a + 3, b + 1), 1), decision(0, 1, (a + 3, b + 1), 1),
            decision(0, 2, (a + 3, b + 6), 0), decision(1, 2, (a + 3, b + 6), 0)], winner)

    def test_hand_boundaries_read_the_score_after_the_hand_with_the_next_mano(self):
        result = self.three_hand_match()
        seat0 = target_module.decision_targets(result, 0, self.table, 0.5)
        seat1 = target_module.decision_targets(result, 1, self.table, 0.5)
        # Hand 0 -> next hand starts 3-1 with seat 1 as mano.
        self.assertAlmostEqual(seat0[0], expected(.5, 1, self.table.equity(3, 1, 15, False)))
        self.assertAlmostEqual(seat0[1], seat0[0])  # both seat 0 decisions of hand 0 share one target
        self.assertAlmostEqual(seat1[0], expected(.5, -1, self.table.equity(1, 3, 15, True)))
        # Hand 1 -> next hand starts 3-6 with seat 0 as mano.
        self.assertAlmostEqual(seat0[2], expected(.5, 1, self.table.equity(3, 6, 15, True)))
        self.assertAlmostEqual(seat1[1], expected(.5, -1, self.table.equity(6, 3, 15, False)))

    def test_final_hand_is_the_terminal_value(self):
        result = self.three_hand_match()
        self.assertEqual(target_module.decision_targets(result, 0, self.table, 0.5)[-1], 1.0)
        self.assertEqual(target_module.decision_targets(result, 1, self.table, 0.5)[-1], -1.0)
        self.assertEqual(target_module.decision_targets(result, 0, self.table, 1.0)[-1], 1.0)

    def test_blend_zero_is_the_old_match_result_target(self):
        result = self.three_hand_match(winner=1)
        self.assertEqual(target_module.decision_targets(result, 0, self.table, 0.0), [-1.0] * 4)
        self.assertEqual(target_module.decision_targets(result, 1, self.table, 0.0), [1.0] * 3)

    def test_pure_equity_blend(self):
        result = self.three_hand_match()
        first = target_module.decision_targets(result, 0, self.table, 1.0)[0]
        self.assertAlmostEqual(first, 2 * self.table.equity(3, 1, 15, False) - 1)

    def test_falta_envido_ending_the_match_inside_a_hand_is_terminal(self):
        result = fake_result([decision(0, 0, (0, 0), 0), decision(1, 0, (0, 0), 0),
                              decision(1, 1, (4, 2), 1), decision(0, 1, (4, 2), 1), decision(1, 1, (4, 2), 1)], 1)
        self.assertEqual(target_module.decision_targets(result, 1, self.table, 0.5)[1:], [1.0, 1.0])
        self.assertEqual(target_module.decision_targets(result, 0, self.table, 0.5)[1], -1.0)

    def test_envido_points_scored_mid_hand_are_not_counted_twice(self):
        # Envido moved the score during hand 0 (the second decision shows 5-0), but the target only
        # reads the score at the start of hand 1.
        result = fake_result([decision(0, 0, (0, 0), 0), decision(1, 0, (5, 0), 0),
                              decision(0, 1, (6, 0), 1), decision(1, 1, (6, 2), 1)], 0)
        self.assertAlmostEqual(target_module.decision_targets(result, 0, self.table, 0.5)[0],
                               expected(.5, 1, self.table.equity(6, 0, 15, False)))

    def test_random_start_and_thirty_point_matches(self):
        result = fake_result([decision(0, 0, (7, 9), 0, 30), decision(1, 0, (7, 9), 0, 30),
                              decision(0, 1, (7, 12), 1, 30), decision(1, 1, (7, 12), 1, 30)], 1)
        self.assertAlmostEqual(target_module.decision_targets(result, 0, self.table, 0.5)[0],
                               expected(.5, -1, self.table.equity(7, 12, 30, False)))
        self.assertAlmostEqual(target_module.decision_targets(result, 1, self.table, 0.5)[0],
                               expected(.5, 1, self.table.equity(12, 7, 30, True)))

    def test_invalid_blend_is_rejected(self):
        with self.assertRaises(ValueError):
            target_module.decision_targets(self.three_hand_match(), 0, self.table, 1.5)

    def test_real_matches_give_bounded_targets_and_terminal_last_hand(self):
        for seed in range(6):
            start = (2, 11) if seed % 2 else (0, 0)
            result = play_match(RandomPolicy(seed), RandomPolicy(seed + 50), seed=seed, target=15, starting_score=start)
            last_hand = result.decisions[-1]['hand_index']
            for seat in (0, 1):
                values = target_module.decision_targets(result, seat, self.table, 0.5)
                mine = [d for d in result.decisions if d['player'] == seat]
                self.assertEqual(len(values), len(mine))
                self.assertTrue(all(-1 <= v <= 1 for v in values))
                final = 1.0 if result.winner == seat else -1.0
                self.assertTrue(all(v == final for v, d in zip(values, mine) if d['hand_index'] == last_hand))


if __name__ == '__main__':
    unittest.main()
