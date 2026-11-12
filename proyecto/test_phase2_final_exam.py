"""Synthetic tests for the phase 2 final exam tool. Never uses the real exam seeds or model files."""
import math
import statistics
import unittest

import phase2_final_exam as fx


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class StatisticsTests(unittest.TestCase):
    def test_pair_score_counts_evaluated_wins_over_both_seats(self):
        # game 0: evaluated sits in seat 0; game 1: evaluated sits in seat 1
        self.assertEqual(fx.pair_score([0, 1]), 1.0)
        self.assertEqual(fx.pair_score([1, 0]), 0.0)
        self.assertEqual(fx.pair_score([0, 0]), 0.5)
        self.assertEqual(fx.pair_score([1, 1]), 0.5)

    def test_paired_gain_uses_sample_sd_over_sqrt_n(self):
        stat = fx.paired_stat([0.5, 0.0, 1.0, 0.5])
        self.assertAlmostEqual(stat['mean'], 0.5)
        self.assertAlmostEqual(stat['se'], math.sqrt(1 / 6) / 2)  # sample variance 1/6
        self.assertAlmostEqual(stat['lower'], 0.5 - 1.96 * stat['se'])
        self.assertAlmostEqual(stat['upper'], 0.5 + 1.96 * stat['se'])
        self.assertEqual(stat['n'], 4)

    def test_one_sided_p_uses_normal_cdf(self):
        self.assertAlmostEqual(fx.one_sided_p(1.0, 0.5), 1 - phi(2.0))
        self.assertAlmostEqual(fx.one_sided_p(-1.0, 0.5), 1 - phi(-2.0))
        self.assertEqual(fx.one_sided_p(0.0, 0.0), 0.5)
        self.assertEqual(fx.one_sided_p(0.1, 0.0), 0.0)
        self.assertEqual(fx.one_sided_p(-0.1, 0.0), 1.0)

    def test_holm_smaller_p_at_half_alpha_larger_only_if_smaller_passed(self):
        self.assertEqual(fx.holm({'C1': 0.012, 'C2': 0.02}), {'C1': True, 'C2': True})
        self.assertEqual(fx.holm({'C1': 0.02, 'C2': 0.012}), {'C1': True, 'C2': True})
        self.assertEqual(fx.holm({'C1': 0.0125, 'C2': 0.0251}), {'C1': True, 'C2': False})
        # smaller fails the 0.0125 level: nothing is rejected, even though 0.02 < 0.025
        self.assertEqual(fx.holm({'C1': 0.013, 'C2': 0.02}), {'C1': False, 'C2': False})
        self.assertEqual(fx.holm({'C1': 0.5, 'C2': 0.001}), {'C1': False, 'C2': True})

    def test_guardrail_bound_equal_to_threshold_fails(self):
        self.assertFalse(fx.guardrail_pass(-0.02, 0.0))
        self.assertTrue(fx.guardrail_pass(-0.0199, 0.0))
        self.assertTrue(fx.guardrail_pass(0.0, 0.0102))   # bound -0.019992
        self.assertFalse(fx.guardrail_pass(0.0, 0.0103))  # bound -0.020188


N = 100


def noise(a=0.5):
    return [a if i % 2 == 0 else -a for i in range(N)]


def shifted(m, spread=0.5):
    """N paired differences with exact mean m and sample SD spread*sqrt(N/(N-1))."""
    return [m + x for x in noise(spread)]


def tight(m=0.0):
    """Mean m, tiny spread: one +.5 and one -.5 among N-2 values equal to m."""
    d = [m] * N
    d[0] += 0.5
    d[1] -= 0.5
    return d


def exam_scores(diffs):
    """diffs[name][k] -> scores where B scores 0.5 everywhere and candidate = B + diff."""
    base = [0.5] * N
    scores = {'B': [list(base) for _ in range(5)]}
    for name, per_block in diffs.items():
        scores[name] = [[b + d for b, d in zip(base, per_block[k])] for k in range(5)]
    return scores


def clean_pass():
    return [shifted(0.3)] + [tight(0.0) for _ in range(4)]


def clean_fail():
    return [shifted(-0.3)] + [tight(0.0) for _ in range(4)]


class ExamDecisionTests(unittest.TestCase):
    def test_first_ranked_passing_candidate_ships(self):
        result = fx.analyze_exam(exam_scores({'C1': clean_pass(), 'C2': clean_pass()}))
        self.assertTrue(result['candidates']['C1']['passes'])
        self.assertTrue(result['candidates']['C2']['passes'])
        self.assertEqual(result['decision'], 'C1')

    def test_second_ships_when_first_fails_a_guardrail(self):
        bad = clean_pass()
        bad[3] = tight(-0.1)
        result = fx.analyze_exam(exam_scores({'C1': bad, 'C2': clean_pass()}))
        c1 = result['candidates']['C1']
        self.assertTrue(c1['primary']['holm_rejected'])
        self.assertFalse(c1['guardrails_pass'])
        self.assertFalse(c1['guardrails'][2]['pass'])  # k = 3 is the third guardrail
        self.assertTrue(c1['guardrails'][0]['pass'])
        self.assertEqual(result['decision'], 'C2')

    def test_baseline_stays_when_nobody_passes(self):
        result = fx.analyze_exam(exam_scores({'C1': clean_fail(), 'C2': clean_fail()}))
        self.assertEqual(result['decision'], 'B')
        self.assertFalse(result['candidates']['C1']['primary']['holm_rejected'])

    def test_gain_is_paired_difference_against_baseline_with_se(self):
        result = fx.analyze_exam(exam_scores({'C1': clean_pass(), 'C2': clean_fail()}))
        gain = result['candidates']['C1']['gains'][0]
        sd = statistics.stdev(shifted(0.3))
        self.assertAlmostEqual(gain['gain'], 0.3)
        self.assertAlmostEqual(gain['se'], sd / math.sqrt(N))
        self.assertAlmostEqual(result['candidates']['C1']['primary']['z'], 0.3 / (sd / math.sqrt(N)))

    def test_holm_blocks_two_candidates_that_each_pass_at_025(self):
        sd = statistics.stdev(shifted(0.0))
        def p_of(m):
            return 1 - phi(m / (sd / math.sqrt(N)))
        a, b = 0.105, 0.11
        self.assertTrue(0.0125 < p_of(b) < p_of(a) < 0.025)  # independent check of the fixture
        rest = [tight(0.0) for _ in range(4)]
        result = fx.analyze_exam(exam_scores({'C1': [shifted(a)] + rest, 'C2': [shifted(b)] + rest}))
        for name in ('C1', 'C2'):
            self.assertFalse(result['candidates'][name]['primary']['holm_rejected'])
        self.assertEqual(result['decision'], 'B')

    def test_balance_must_be_strictly_positive(self):
        # Primary wins big, guardrails individually hold, but the equal-weight mean is negative.
        diffs = [[0.005] * N] + [tight(-0.004) for _ in range(4)]
        result = fx.analyze_exam(exam_scores({'C1': diffs, 'C2': clean_fail()}))
        c1 = result['candidates']['C1']
        self.assertTrue(c1['primary']['holm_rejected'])
        self.assertTrue(c1['guardrails_pass'])
        self.assertLess(c1['balance']['mean'], 0)
        self.assertFalse(c1['balance']['pass'])
        self.assertFalse(c1['passes'])
        self.assertEqual(result['decision'], 'B')

    def test_reports_win_rates_including_against_adaptive_utility(self):
        result = fx.analyze_exam(exam_scores({'C1': clean_pass(), 'C2': clean_fail()}))
        au = result['win_rates']['C1'][0]
        self.assertAlmostEqual(au['mean'], 0.8)
        self.assertTrue(au['above_half'])
        self.assertAlmostEqual(result['win_rates']['B'][0]['mean'], 0.5)
        self.assertFalse(result['win_rates']['B'][0]['above_half'])
        self.assertFalse(result['win_rates']['C2'][0]['above_half'])


def temp_scores(au_half_wins_002=0, au_half_wins_greedy=0, p1_half_wins_002=0, p1_half_wins_greedy=0, T=300):
    """Pair scores at 0.05 are all 0.5; the others get that many pairs raised by +0.5."""
    def block(extra):
        return [1.0 if i < extra else 0.5 for i in range(T)]
    return {'0.05': [block(0), block(0)],
            '0.02': [block(au_half_wins_002), block(p1_half_wins_002)],
            'greedy': [block(au_half_wins_greedy), block(p1_half_wins_greedy)]}


class TemperatureRuleTests(unittest.TestCase):
    def test_gain_of_exactly_two_points_in_equal_weight_mean_replaces_0_05(self):
        # +24 pairs of +0.5 against AU only: AU mean +0.04, m(T) +0.02 exactly
        result = fx.analyze_temperature(temp_scores(au_half_wins_002=24))
        self.assertEqual(result['tstar'], '0.02')
        self.assertTrue(result['changed'])

    def test_gain_just_below_two_points_keeps_0_05(self):
        result = fx.analyze_temperature(temp_scores(au_half_wins_002=23))
        self.assertEqual(result['tstar'], '0.05')
        self.assertFalse(result['changed'])

    def test_ties_break_toward_0_05_then_0_02(self):
        both = fx.analyze_temperature(temp_scores(au_half_wins_002=30, au_half_wins_greedy=30))
        self.assertEqual(both['tstar'], '0.02')
        greedy_only = fx.analyze_temperature(temp_scores(au_half_wins_greedy=30))
        self.assertEqual(greedy_only['tstar'], 'greedy')
        # 0.02 ties 0.05 as the best (no gain) while greedy is worse: stays at 0.05
        worse = temp_scores()
        worse['greedy'][0][0] = 0.0
        self.assertEqual(fx.analyze_temperature(worse)['tstar'], '0.05')

    def test_best_temperature_is_by_equal_weight_mean_of_the_two_opponents(self):
        result = fx.analyze_temperature(temp_scores(au_half_wins_002=10, p1_half_wins_002=60,
                                                    au_half_wins_greedy=50, p1_half_wins_greedy=30))
        # m gains: 0.02 -> (10+60)/2/600 = .0583..., greedy -> (50+30)/2/600 = .0667
        self.assertEqual(result['tstar'], 'greedy')
        self.assertAlmostEqual(result['m']['greedy'] - result['m']['0.05'], 80 / 2 / 600 * 1.0, places=9)

    def test_reports_all_three_means_with_paired_intervals(self):
        result = fx.analyze_temperature(temp_scores(au_half_wins_002=24))
        self.assertEqual(set(result['m']), {'0.05', '0.02', 'greedy'})
        self.assertAlmostEqual(result['m']['0.05'], 0.5)
        paired = result['paired_vs_0.05']['0.02']
        self.assertAlmostEqual(paired['m_diff']['mean'], 0.02)
        self.assertAlmostEqual(paired['per_opponent'][0]['mean'], 0.04)
        d = [0.5] * 24 + [0.0] * 276
        se_au = statistics.stdev(d) / math.sqrt(300)
        se_p1 = 0.0
        self.assertAlmostEqual(paired['m_diff']['se'], math.sqrt(se_au ** 2 + se_p1 ** 2) / 2)
        self.assertAlmostEqual(paired['m_diff']['lower'], 0.02 - 1.96 * paired['m_diff']['se'])


class SeedTests(unittest.TestCase):
    """The real plans are only inspected, never played, in these tests."""

    def test_part_t_seed_ranges_match_the_rule(self):
        deal, action = fx.seed_ranges(fx.PART_T_PLAN)
        self.assertEqual(deal, (180002000000000, 180004000000598))
        self.assertEqual(action, (2 ** 131, 2 ** 131 + 2399))

    def test_part_e_seed_ranges_match_the_rule(self):
        deal, action = fx.seed_ranges(fx.PART_E_PLAN)
        self.assertEqual(deal, (182002000000000, 182010000001998))
        self.assertEqual(action, (2 ** 130, 2 ** 130 + 19999))

    def test_seeds_follow_the_rule_formulas_at_sample_points(self):
        t, e = fx.PART_T_PLAN, fx.PART_E_PLAN
        self.assertEqual(fx.deal_seed(t, 1, 7), 2 * (90002 * 10 ** 9 + 7))
        self.assertEqual(fx.action_seed(t, 1, 7, 1, 0), 2 ** 131 + (((1 * 300 + 7) * 2 + 1) * 2 + 0))
        self.assertEqual(fx.deal_seed(e, 3, 5), 2 * (91004 * 10 ** 9 + 5))
        self.assertEqual(fx.action_seed(e, 3, 5, 0, 1), 2 ** 130 + (((3 * 1000 + 5) * 2 + 0) * 2 + 1))

    def test_every_seed_in_each_plan_is_unique_and_parts_are_disjoint(self):
        sets = []
        for plan in (fx.PART_T_PLAN, fx.PART_E_PLAN):
            deals, actions = fx.all_seeds(plan)
            self.assertEqual(len(deals), len(plan['opponents']) * plan['count'])
            self.assertEqual(len(actions), len(plan['opponents']) * plan['count'] * 4)
            sets.append((deals, actions))
        self.assertFalse(sets[0][0] & sets[1][0])
        self.assertFalse(sets[0][1] & sets[1][1])

    def test_disjoint_from_every_earlier_namespace(self):
        from phase2_promotion import RESERVED_SEEDS
        self.assertTrue(RESERVED_SEEDS <= fx.EARLIER_SCHEDULE_SEEDS)
        for plan in (fx.PART_T_PLAN, fx.PART_E_PLAN):
            (deal_lo, deal_hi), (action_lo, action_hi) = fx.seed_ranges(plan)
            for seed in fx.EARLIER_SCHEDULE_SEEDS:
                lo, hi = 2 * seed * 10 ** 9, 2 * (seed * 10 ** 9 + 10 ** 9 - 1)
                self.assertTrue(hi < deal_lo or lo > deal_hi, seed)
            for lo, hi in fx.EARLIER_ACTION_RANGES:
                self.assertTrue(hi < action_lo or lo > action_hi, (lo, hi))
        new_schedule_seeds = {90001, 90002, 91001, 91002, 91003, 91004, 91005}
        self.assertFalse(new_schedule_seeds & fx.EARLIER_SCHEDULE_SEEDS)

    def test_target_alternates_15_and_30(self):
        self.assertEqual([fx.pair_target(fx.PART_E_PLAN, i) for i in range(4)], [15, 30, 15, 30])


class RecordTests(unittest.TestCase):
    PLAN = {'part': 'X', 'deal_base': 7, 'action_exp': 140, 'count': 4, 'targets': [3, 5],
            'opponents': ['AU', 'R']}

    def test_expected_record_has_decimal_action_seeds_and_target(self):
        rec = fx.expected_record(self.PLAN, 1, 3)
        self.assertEqual(rec['block'], 1)
        self.assertEqual(rec['i'], 3)
        self.assertEqual(rec['deal_seed'], 2 * (8 * 10 ** 9 + 3))
        self.assertEqual(rec['target'], 5)
        base = 2 ** 140 + ((1 * 4 + 3) * 2) * 2
        self.assertEqual(rec['action_seeds'], [[str(base), str(base + 1)], [str(base + 2), str(base + 3)]])

    def test_validate_records_rejects_duplicates_bad_winners_and_missing_pairs(self):
        names = ['a', 'b']
        def rec(b, i, winners=None):
            r = fx.expected_record(self.PLAN, b, i)
            r['winners'] = winners or {'a': [0, 1], 'b': [1, 1]}
            return r
        full = [rec(b, i) for b in range(2) for i in range(4)]
        self.assertEqual(len(fx.validate_records(full, self.PLAN, names)), 8)
        with self.assertRaises(ValueError):
            fx.validate_records(full[:-1], self.PLAN, names)
        self.assertEqual(len(fx.validate_records(full[:-1], self.PLAN, names, complete=False)), 7)
        with self.assertRaises(ValueError):
            fx.validate_records(full + [full[0]], self.PLAN, names, complete=False)
        with self.assertRaises(ValueError):
            fx.validate_records([rec(0, 0, {'a': [0, 2], 'b': [0, 0]})], self.PLAN, names, complete=False)
        with self.assertRaises(ValueError):
            fx.validate_records([rec(0, 0, {'a': [0, 1]})], self.PLAN, names, complete=False)
        with self.assertRaises(ValueError):
            fx.validate_records([rec(2, 0)], self.PLAN, names, complete=False)
        tampered = rec(0, 1)
        tampered['deal_seed'] += 2
        with self.assertRaises(ValueError):
            fx.validate_records([tampered], self.PLAN, names, complete=False)


class PlayPairTests(unittest.TestCase):
    PLAN = RecordTests.PLAN

    def play(self, evaluated, winner_of=lambda calls: 0):
        from types import SimpleNamespace
        from unittest.mock import patch
        made, matches = [], []

        def factory(spec, seed):
            policy = SimpleNamespace(spec=spec, seed=seed)
            made.append(policy)
            return policy

        def play_match(a, b, **kwargs):
            matches.append((a, b, kwargs))
            return SimpleNamespace(winner=winner_of(matches))

        with patch('phase2_final_exam.play_match', play_match):
            record = fx.play_record(self.PLAN, 1, 3, evaluated, 'OPP', factory)
        return record, made, matches

    def test_same_deal_and_action_seeds_for_every_evaluated_network(self):
        record, made, matches = self.play({'a': 'NET_A', 'b': 'NET_B'})
        expected = fx.expected_record(self.PLAN, 1, 3)
        self.assertEqual({k: v for k, v in record.items() if k != 'winners'}, expected)
        self.assertEqual(len(matches), 4)
        seeds = [int(s) for pair in expected['action_seeds'] for s in pair]
        by_net = {'NET_A': [], 'NET_B': []}
        for game in matches:
            for seat, policy in enumerate(game[:2]):
                by_net.setdefault(policy.spec, []).append((seat, policy.seed))
        self.assertEqual([seed for _, seed in by_net['NET_A']], [seeds[0], seeds[3]])  # a: game0 seat0, game1 seat1
        self.assertEqual([seed for _, seed in by_net['NET_B']], [seeds[0], seeds[3]])
        self.assertEqual([seed for _, seed in by_net['OPP']], [seeds[1], seeds[2], seeds[1], seeds[2]])
        for _, _, kwargs in matches:
            self.assertEqual(kwargs, {'seed': expected['deal_seed'], 'target': expected['target'],
                                      'starting_score': (0, 0), 'deal_partition': 0})

    def test_evaluated_network_sits_in_seat_0_then_seat_1(self):
        _, _, matches = self.play({'a': 'NET_A'})
        self.assertEqual([m[0].spec for m in matches], ['NET_A', 'OPP'])
        self.assertEqual([m[1].spec for m in matches], ['OPP', 'NET_A'])

    def test_every_game_builds_new_policies(self):
        _, made, _ = self.play({'a': 'NET_A', 'b': 'NET_B'})
        self.assertEqual(len(made), 8)
        self.assertEqual(len({id(p) for p in made}), 8)

    def test_winners_are_recorded_per_game(self):
        record, _, _ = self.play({'a': 'NET_A'}, winner_of=lambda calls: 0)
        self.assertEqual(record['winners'], {'a': [0, 0]})
        record, _, _ = self.play({'a': 'NET_A'}, winner_of=lambda calls: 1)
        self.assertEqual(record['winners'], {'a': [1, 1]})
        self.assertEqual(fx.pair_score(record['winners']['a']), 0.5)


# ---------------------------------------------------------------- runner (synthetic models, tiny plans)

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    import numpy as np
    import torch
    from phase1_network import GENOME_SIZE
    from phase2_torch import Phase2Network
except ImportError:  # pragma: no cover
    torch = None

SYNTH_T = fx.make_plan('T', 7, 140, 4, ['AU', 'P1'], targets=(2, 3))
SYNTH_E = fx.make_plan('E', 11, 141, 2, ['AU', 'R', 'P1', 'H1', 'H2'], targets=(2, 3))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    subprocess.run(['git', '-C', str(root), '-c', 'user.email=t@example.com', '-c', 'user.name=t', *args],
                   check=True, capture_output=True)


@unittest.skipUnless(torch is not None, 'PyTorch is optional')
class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'repo'
        (self.root / 'proyecto').mkdir(parents=True)
        (self.root / 'proyecto/helper.py').write_text('X = 1\n')
        (self.root / 'RULE.txt').write_text('rule\n')
        git(self.root, 'init', '-q')
        git(self.root, 'add', '-A')
        git(self.root, 'commit', '-q', '-m', 'fixture')
        self.models = self.make_models(Path(self.tmp.name) / 'models')
        self.out = Path(self.tmp.name) / 'exam'

    def make_models(self, folder):
        folder.mkdir()
        for name, seed in (('c1', 1), ('c2', 2), ('b', 3)):
            Phase2Network(seed=seed).save(folder / f'{name}.pt')
        shutil.copy(folder / 'c1.pt', folder / 'c1_snapshot.pt')
        (folder / 'c1.json').write_text(json.dumps({'generation': 38, 'temperature': 0.05, 'profile_prior': 5.0}))
        (folder / 'b.json').write_text(json.dumps({'generation': 28, 'temperature': 0.05, 'profile_prior': 5.0}))
        for name, offset in (('p1', 0), ('h1', 1), ('h2', 2)):
            np.save(folder / f'{name}.npy', np.zeros(GENOME_SIZE) + offset / 100)

        def entry(kind, file=None, **extra):
            item = {'kind': kind, **extra}
            if file:
                item.update(path=str(folder / file), sha256=sha(folder / file))
            return item
        return {'C1': entry('phase2', 'c1.pt', profile_prior=5,
                            equals={'path': str(folder / 'c1_snapshot.pt'), 'sha256': sha(folder / 'c1_snapshot.pt')},
                            sidecar={'path': str(folder / 'c1.json'), 'sha256': sha(folder / 'c1.json'),
                                     'require': {'generation': 38, 'profile_prior': 5}}),
                'C2': entry('phase2', 'c2.pt', profile_prior=5),
                'B': entry('phase2', 'b.pt', profile_prior=5, temperature=0.05,
                           sidecar={'path': str(folder / 'b.json'), 'sha256': sha(folder / 'b.json'),
                                    'require': {'temperature': 0.05, 'profile_prior': 5}}),
                'AU': {'kind': 'au'}, 'R': {'kind': 'reactive'},
                'P1': entry('phase1', 'p1.npy'), 'H1': entry('phase1', 'h1.npy'), 'H2': entry('phase1', 'h2.npy')}

    def part_t(self, **kwargs):
        args = dict(root=self.root, output=self.out / 'part-t', workers=2, models=self.models, plan=SYNTH_T,
                    rule_path=self.root / 'RULE.txt')
        args.update(kwargs)
        return fx.run_part_t(**args)

    def part_e(self, **kwargs):
        args = dict(root=self.root, output=self.out / 'part-e', part_t_output=self.out / 'part-t', workers=2,
                    models=self.models, plan=SYNTH_E, plan_t=SYNTH_T, rule_path=self.root / 'RULE.txt')
        args.update(kwargs)
        return fx.run_part_e(**args)

    def lines(self, folder):
        return (folder / 'partial.jsonl').read_text(encoding='utf-8').splitlines()


class PartTRunTests(RunnerTestCase):
    def test_completes_records_config_and_validates_every_pair(self):
        result = self.part_t()
        out = self.out / 'part-t'
        for name in ('started.lock', 'config.json', 'partial.jsonl', 'results.json', 'complete.txt'):
            self.assertTrue((out / name).exists(), name)
        records = [json.loads(line) for line in self.lines(out)]
        self.assertEqual(len(records), 2 * 4)
        self.assertEqual(len(fx.validate_records(records, SYNTH_T, ['0.05', '0.02', 'greedy'])), 8)
        config = json.loads((out / 'config.json').read_text())
        head = subprocess.check_output(['git', '-C', str(self.root), 'rev-parse', 'HEAD'], text=True).strip()
        self.assertEqual(config['git_revision'], head)
        self.assertEqual(config['sha256']['proyecto/helper.py'], sha(self.root / 'proyecto/helper.py'))
        self.assertEqual(config['rule_sha256'], sha(self.root / 'RULE.txt'))
        self.assertEqual(config['models']['C1']['sha256'], self.models['C1']['sha256'])
        self.assertEqual(config['models']['C1']['sidecar']['sha256'], self.models['C1']['sidecar']['sha256'])
        self.assertEqual(config['plan'], SYNTH_T)
        self.assertIn(result['analysis']['tstar'], ('0.05', '0.02', 'greedy'))
        self.assertEqual(set(json.loads((out / 'results.json').read_text())['analysis']['m']),
                         {'0.05', '0.02', 'greedy'})

    def test_config_is_written_before_any_game(self):
        from unittest.mock import patch
        seen = {}

        def boom(*args, **kwargs):
            seen['config'] = (self.out / 'part-t/config.json').exists()
            seen['lock'] = (self.out / 'part-t/started.lock').exists()
            raise RuntimeError('stop')
        with patch('phase2_final_exam._play_jobs', boom):
            with self.assertRaises(RuntimeError):
                self.part_t()
        self.assertEqual(seen, {'config': True, 'lock': True})

    def test_refuses_existing_output(self):
        (self.out / 'part-t').mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.part_t()

    def test_second_run_without_resume_is_refused(self):
        self.part_t()
        with self.assertRaises(FileExistsError):
            self.part_t()

    def test_refuses_identity_mismatches_without_creating_output(self):
        def variant(edit):
            bad = json.loads(json.dumps(self.models))
            edit(bad)
            with self.assertRaises(ValueError):
                self.part_t(models=bad)
        variant(lambda m: m['C1'].update(sha256='0' * 64))
        # declared snapshot hash matches its file, but the tensors differ from C1
        variant(lambda m: m['C1'].update(equals={'path': m['C2']['path'], 'sha256': m['C2']['sha256']}))
        variant(lambda m: m['C1']['sidecar']['require'].update(generation=37))
        variant(lambda m: m['C1']['sidecar'].update(sha256='1' * 64))
        variant(lambda m: m['P1'].update(sha256='f' * 64))
        self.assertFalse((self.out / 'part-t').exists())

    def test_refuses_uncommitted_changes(self):
        (self.root / 'proyecto/helper.py').write_text('X = 2\n')
        with self.assertRaises(ValueError):
            self.part_t()
        self.assertFalse((self.out / 'part-t').exists())

    def test_refuses_untracked_project_python_and_modified_rule(self):
        (self.root / 'proyecto/new_module.py').write_text('Y = 1\n')
        with self.assertRaises(ValueError):
            self.part_t()
        (self.root / 'proyecto/new_module.py').unlink()
        (self.root / 'RULE.txt').write_text('changed\n')
        with self.assertRaises(ValueError):
            self.part_t()

    def test_refuses_invalid_worker_counts(self):
        with self.assertRaises(ValueError):
            self.part_t(workers=13)
        with self.assertRaises(ValueError):
            self.part_t(workers=0)

    def test_results_do_not_depend_on_worker_count(self):
        self.part_t(workers=1, output=self.out / 'one')
        self.part_t(workers=3, output=self.out / 'three')
        a = json.loads((self.out / 'one/results.json').read_text())
        b = json.loads((self.out / 'three/results.json').read_text())
        self.assertEqual(a['analysis'], b['analysis'])
        self.assertEqual(sorted(self.lines(self.out / 'one')), sorted(self.lines(self.out / 'three')))


class ResumeTests(RunnerTestCase):
    def interrupted(self, keep, tail=''):
        self.part_t()
        out = self.out / 'part-t'
        original = (out / 'results.json').read_bytes()
        lines = self.lines(out)
        (out / 'partial.jsonl').write_text(''.join(line + '\n' for line in lines[:keep]) + tail, encoding='utf-8')
        (out / 'results.json').unlink()
        (out / 'complete.txt').unlink()
        return out, original, lines

    def test_resume_only_plays_missing_pairs_and_reproduces_results(self):
        from unittest.mock import patch
        out, original, lines = self.interrupted(5)
        played = []
        real = fx._play_jobs

        def spy(config, jobs, workers):
            played.extend(jobs)
            return real(config, jobs, workers)
        with patch('phase2_final_exam._play_jobs', spy):
            self.part_t(resume=True)
        kept = {(json.loads(x)['block'], json.loads(x)['i']) for x in lines[:5]}
        self.assertEqual(len(played), 3)
        self.assertFalse(kept & set(map(tuple, played)))
        self.assertEqual(self.lines(out)[:5], lines[:5])
        self.assertEqual((out / 'results.json').read_bytes(), original)
        self.assertTrue((out / 'complete.txt').exists())

    def test_resume_drops_a_torn_last_line(self):
        out, original, _ = self.interrupted(4, tail='{"block": 1, "i"')
        self.part_t(resume=True)
        self.assertEqual((out / 'results.json').read_bytes(), original)
        self.assertEqual(len(self.lines(out)), 8)

    def test_resume_refuses_duplicate_records(self):
        out, _, lines = self.interrupted(3)
        (out / 'partial.jsonl').write_text('\n'.join(lines[:3] + [lines[0]]) + '\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.part_t(resume=True)

    def test_resume_refuses_changed_configuration(self):
        self.interrupted(3)
        (self.root / 'proyecto/helper.py').write_text('X = 5\n')
        git(self.root, 'commit', '-q', '-am', 'edit')
        with self.assertRaises(ValueError):
            self.part_t(resume=True)

    def test_resume_without_started_run_or_after_completion_is_refused(self):
        with self.assertRaises(ValueError):
            self.part_t(resume=True)
        self.part_t()
        with self.assertRaises(ValueError):
            self.part_t(resume=True)


class PartERunTests(RunnerTestCase):
    def test_part_e_uses_tstar_recomputed_from_part_t_and_records_it(self):
        t = self.part_t()
        result = self.part_e()
        config = json.loads((self.out / 'part-e/config.json').read_text())
        self.assertEqual(config['tstar'], t['analysis']['tstar'])
        self.assertEqual(config['part_t_raw_sha256'], t['raw_sha256'])
        self.assertEqual([e['name'] for e in config['evaluated']], ['C1', 'C2', 'B'])
        temperatures = {e['name']: e['temperature'] for e in config['evaluated']}
        self.assertEqual(temperatures['B'], 0.05)
        self.assertEqual(temperatures['C1'], temperatures['C2'])
        records = [json.loads(line) for line in self.lines(self.out / 'part-e')]
        self.assertEqual(len(records), 5 * 2)
        self.assertIn(result['analysis']['decision'], ('C1', 'C2', 'B'))
        self.assertEqual(len(result['analysis']['win_rates']['B']), 5)
        self.assertTrue((self.out / 'part-e/complete.txt').exists())

    def test_part_e_checks_the_baseline_identity_and_sidecar(self):
        self.part_t()
        for edit in (lambda m: m['B'].update(sha256='0' * 64),
                     lambda m: m['B']['sidecar']['require'].update(temperature=0.07),
                     lambda m: m['C2'].update(sha256='1' * 64)):
            bad = json.loads(json.dumps(self.models))
            edit(bad)
            with self.assertRaises(ValueError):
                self.part_e(models=bad)
        self.assertFalse((self.out / 'part-e').exists())

    def test_part_e_refuses_without_a_complete_part_t(self):
        with self.assertRaises(ValueError):
            self.part_e()
        self.part_t()
        (self.out / 'part-t/complete.txt').unlink()
        with self.assertRaises(ValueError):
            self.part_e()
        self.assertFalse((self.out / 'part-e').exists())

    def test_part_e_refuses_explicit_temperature_that_disagrees_with_part_t(self):
        t = self.part_t()
        wrong = next(x for x in fx.TEMPERATURES if x != t['analysis']['tstar'])
        with self.assertRaises(ValueError):
            self.part_e(temperature=wrong)
        self.part_e(temperature=t['analysis']['tstar'])

    def test_part_e_refuses_if_part_t_raw_records_were_altered(self):
        self.part_t()
        out = self.out / 'part-t'
        lines = self.lines(out)
        flipped = json.loads(lines[0])
        flipped['winners']['greedy'] = [1 - w for w in flipped['winners']['greedy']]
        (out / 'partial.jsonl').write_text('\n'.join([json.dumps(flipped, sort_keys=True)] + lines[1:]) + '\n')
        with self.assertRaises(ValueError):
            self.part_e()

    def test_part_e_refuses_if_sources_changed_since_part_t(self):
        self.part_t()
        (self.root / 'proyecto/helper.py').write_text('X = 9\n')
        git(self.root, 'commit', '-q', '-am', 'edit')
        with self.assertRaises(ValueError):
            self.part_e()
        self.assertFalse((self.out / 'part-e').exists())


# ---------------------------------------------------------------- exam configs (run 2 reproduction, run 3 support)

REPO = Path(__file__).resolve().parent.parent
RUN2_CONFIG = REPO / 'experiments/phase2-final-exam/config.json'
RUN3_CONFIG = REPO / 'experiments/phase2-run3-exam/config.json'


class Run2ConfigReproductionTests(unittest.TestCase):
    def setUp(self):
        self.exam = fx.load_exam_config(RUN2_CONFIG, root=REPO)

    def test_plans_equal_the_hardcoded_run_2_plans(self):
        self.assertEqual(self.exam['part_t'], fx.PART_T_PLAN)
        self.assertEqual(self.exam['part_e'], fx.PART_E_PLAN)

    def test_seed_ranges_equal_the_recorded_run_2_ranges(self):
        self.assertEqual(fx.seed_ranges(self.exam['part_t']),
                         ((180002000000000, 180004000000598), (2 ** 131, 2 ** 131 + 2399)))
        self.assertEqual(fx.seed_ranges(self.exam['part_e']),
                         ((182002000000000, 182010000001998), (2 ** 130, 2 ** 130 + 19999)))

    def test_identities_equal_the_hardcoded_run_2_identities(self):
        training = REPO / 'training_output/phase2-run2'
        self.assertEqual(self.exam['models'], fx.real_models(REPO, training))

    def test_evaluated_networks_follow_tstar_like_the_old_part_e(self):
        models = self.exam['models']
        for tstar in fx.TEMPERATURES:
            self.assertEqual(fx.resolve_evaluated(self.exam['evaluated'], tstar),
                             fx.part_e_evaluated(tstar, models))

    def test_run_2_audit_still_passes_against_the_run_2_namespaces(self):
        fx.audit_exam_seeds(self.exam)
