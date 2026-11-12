"""Match equity table (ADR 0062), tested on synthetic chains and tiny seeded self-play only."""
import json
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import phase2_equity as equity
from phase2_torch import Phase2Network
from symmetric_engine import play_match


def exact_chain(target, mano_edge=0.1):
    """Independent truth: each hand one player scores 1-3 points; the mano wins it with 0.5 + mano_edge/2.
    Returns win[own_is_mano][own][opp] solved by backward induction (own = player to be mano or not)."""
    points = {1: .5, 2: .3, 3: .2}
    win = np.zeros((2, target + 1, target + 1))
    for own in range(target, -1, -1):
        for opp in range(target, -1, -1):
            for m in (0, 1):
                if own >= target:
                    win[m][own][opp] = 1
                elif opp >= target:
                    win[m][own][opp] = 0
                else:
                    p_own = .5 + (mano_edge / 2 if m else -mano_edge / 2)
                    value = 0.0
                    for gain, probability in points.items():
                        up = win[1 - m][min(target, own + gain)][opp]
                        down = win[1 - m][own][min(target, opp + gain)]
                        value += probability * (p_own * up + (1 - p_own) * down)
                    win[m][own][opp] = value
    return win


def synthetic_matches(count, target, seed, mano_edge=0.1):
    """Observation lists (one per match) from the chain above, from seat 0's point of view, plus the winner."""
    rng = random.Random(seed)
    matches = []
    for _ in range(count):
        scores, mano, states = [0, 0], 0, []
        while max(scores) < target:
            states.append((target, scores[0], scores[1], mano))
            p0 = .5 + (mano_edge / 2 if mano == 0 else -mano_edge / 2)
            gain = rng.choices((1, 2, 3), (.5, .3, .2))[0]
            scores[0 if rng.random() < p0 else 1] += gain
            mano = 1 - mano
        matches.append((states, 0 if scores[0] > scores[1] else 1))
    return matches


def to_observations(matches):
    rows = []
    for states, winner in matches:
        for target, s0, s1, mano in states:
            rows.append(equity.Observation(target, s0, s1, mano, winner))
    return rows


class TableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = to_observations(synthetic_matches(4000, 15, 1))
        cls.table = equity.fit_equity_table(cls.train)

    def test_terminal_states_are_exact(self):
        for target in (15, 30):
            table = equity.fit_equity_table([equity.Observation(target, 0, 0, 0, 0)])
            for m in (False, True):
                self.assertEqual(table.equity(target, 3, target, m), 1.0)
                self.assertEqual(table.equity(3, target, target, m), 0.0)
                self.assertEqual(table.equity(target + 2, target - 1, target, m), 1.0)

    def test_monotone_in_both_scores_and_inside_zero_one(self):
        for m in (False, True):
            grid = np.array([[self.table.equity(own, opp, 15, m) for opp in range(15)] for own in range(15)])
            self.assertTrue((np.diff(grid, axis=0) >= -1e-12).all())  # up with the own score
            self.assertTrue((np.diff(grid, axis=1) <= 1e-12).all())   # down with the opponent's
            self.assertTrue(((grid > 0) & (grid < 1)).all())

    def test_symmetry_between_the_two_players(self):
        for own in range(15):
            for opp in range(15):
                for m in (False, True):
                    total = self.table.equity(own, opp, 15, m) + self.table.equity(opp, own, 15, not m)
                    self.assertAlmostEqual(total, 1.0, places=9)

    def test_even_state_favours_the_next_mano_when_data_say_so(self):
        self.assertGreater(self.table.equity(7, 7, 15, True), self.table.equity(7, 7, 15, False))

    def test_estimates_follow_the_true_chain_where_data_are_plentiful(self):
        truth = exact_chain(15)
        for own, opp in ((0, 0), (3, 3), (6, 4), (4, 6), (9, 9), (10, 3)):
            for m in (0, 1):
                self.assertAlmostEqual(self.table.equity(own, opp, 15, bool(m)), truth[m][own][opp], delta=0.04)

    def test_tiny_samples_still_give_a_valid_table(self):
        table = equity.fit_equity_table(self.train[:12])
        grid = np.array([[table.equity(o, p, 15, True) for p in range(15)] for o in range(15)])
        self.assertTrue((np.diff(grid, axis=0) >= -1e-12).all() and (np.diff(grid, axis=1) <= 1e-12).all())
        self.assertTrue(((grid >= 0) & (grid <= 1)).all())

    def test_untouched_target_is_prior_only_and_still_valid(self):
        self.assertTrue(0 < self.table.equity(5, 5, 30, True) < 1)
        self.assertAlmostEqual(self.table.equity(5, 5, 30, True) + self.table.equity(5, 5, 30, False), 1.0, places=9)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'table.json'
            self.table.metadata = {'seed': 3, 'matches': 4000}
            self.table.save(path)
            loaded = equity.EquityTable.load(path)
            self.assertEqual(loaded.metadata, {'seed': 3, 'matches': 4000})
            self.assertEqual(loaded.equity(6, 4, 15, True), self.table.equity(6, 4, 15, True))
            self.assertEqual(json.loads(path.read_text())['version'], 1)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.table.equity(1, 1, 20, True)
        with self.assertRaises(ValueError):
            self.table.equity(-1, 1, 15, True)


class BrierTest(unittest.TestCase):
    def test_equity_beats_the_raw_score_difference_on_held_out_matches(self):
        table = equity.fit_equity_table(to_observations(synthetic_matches(4000, 15, 5)))
        held_out = to_observations(synthetic_matches(1500, 15, 6))
        result = equity.brier_comparison(table, held_out)
        self.assertEqual(result['n'], 2 * len(held_out))
        self.assertLess(result['equity'], result['score_difference'])
        self.assertTrue(result['equity_better'])

    def test_brier_value_is_the_mean_squared_error_of_each_player_view(self):
        table = equity.fit_equity_table([equity.Observation(15, 0, 0, 0, 0)])
        row = equity.Observation(15, 2, 1, 0, 0)  # seat 0 won, both views are scored
        p_own = table.equity(2, 1, 15, True)
        p_opp = table.equity(1, 2, 15, False)
        expected = ((p_own - 1) ** 2 + (p_opp - 0) ** 2) / 2
        self.assertAlmostEqual(equity.brier_comparison(table, [row])['equity'], expected, places=12)


class RandomPolicy:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def __call__(self, observation, legal_actions):
        return self.rng.choice(legal_actions)


class CollectTest(unittest.TestCase):
    def test_every_hand_start_is_one_observation_with_its_mano_and_score(self):
        result = play_match(RandomPolicy(1), RandomPolicy(2), seed=7, target=15, starting_score=(4, 9))
        rows = equity.hand_start_observations(result, 15)
        hands = len(result.deals)
        self.assertEqual(len(rows), hands)
        self.assertEqual(tuple(rows[0][1:4]), (4, 9, 0))
        self.assertEqual([r.mano for r in rows], [i % 2 for i in range(hands)])
        self.assertTrue(all(r.winner == result.winner for r in rows))
        scores = [(r.score0, r.score1) for r in rows]
        self.assertEqual(scores, sorted(scores, key=lambda s: (s[0] + s[1])))  # scores never go down


class BuilderTest(unittest.TestCase):
    def _weights(self, folder):
        path = Path(folder) / 'net.pt'
        model = Phase2Network(seed=0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn(parameter.shape, generator=torch.Generator().manual_seed(1)) * 0.05)
        model.save(path)
        return path

    def test_tiny_build_is_reproducible_and_independent_of_the_worker_count(self):
        with tempfile.TemporaryDirectory() as folder:
            weights = self._weights(folder)
            one = equity.build_equity_table(weights, matches=6, seed=31, workers=1)
            again = equity.build_equity_table(weights, matches=6, seed=31, workers=1)
            two = equity.build_equity_table(weights, matches=6, seed=31, workers=2)
            other = equity.build_equity_table(weights, matches=6, seed=32, workers=1)
            self.assertEqual(one.to_json(), again.to_json())
            self.assertEqual(one.to_json(), two.to_json())
            self.assertNotEqual(one.to_json(), other.to_json())
            self.assertEqual(one.metadata['seed'], 31)
            self.assertEqual(one.metadata['matches'], 6)
            self.assertEqual(len(one.metadata['weights_sha256']), 64)

    def test_worker_limit(self):
        with self.assertRaises(ValueError):
            equity.build_equity_table('x', matches=2, seed=1, workers=13)


if __name__ == '__main__':
    unittest.main()
