"""Synthetic tests for the all-vs-all round robin tool. Never plays with the real seeds or model files."""
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import phase2_final_exam as fx
import round_robin_all as rr
from test_phase2_final_exam import git, sha

try:
    import numpy as np
    from phase1_network import GENOME_SIZE
    from phase2_torch import Phase2Network
except ImportError:  # pragma: no cover
    Phase2Network = None

def plan_for(agent_count, deal_base, action_exp, count, targets=(15, 30)):
    return rr.make_plan(rr.pairings(agent_count), deal_base, action_exp, count, targets)


SYNTH = plan_for(4, 600001, 200, 2, (2, 3))     # 6 pairings x 2 pairs


class PairingTests(unittest.TestCase):
    def test_ten_agents_give_45_pairings_each_unordered_pair_once(self):
        table = rr.pairings(10)
        self.assertEqual(len(table), 45)
        self.assertEqual(len({frozenset(p) for p in table}), 45)
        self.assertTrue(all(a < b for a, b in table))
        self.assertEqual(table[0], (0, 1))
        self.assertEqual(table[-1], (8, 9))
        self.assertEqual(len(plan_for(10, 600001, 200, 1000)['opponents']), 45)


class SeedTests(unittest.TestCase):
    def test_formulas_and_uniqueness_over_all_pairings(self):
        plan = plan_for(10, 600001, 200, 1000)
        self.assertEqual(fx.deal_seed(plan, 3, 7), 2 * ((600001 + 3) * 10 ** 9 + 7))
        self.assertEqual(fx.action_seed(plan, 3, 7, 1, 1), 2 ** 200 + (((3 * 1000 + 7) * 2 + 1) * 2 + 1))
        deals, actions = fx.all_seeds(plan)
        self.assertEqual((len(deals), len(actions)), (45 * 1000, 45 * 1000 * 4))
        rr.audit_seeds(plan)

    def test_audit_refuses_overlap_with_the_run_3_exam_and_older_namespaces(self):
        for base, exp in ((93001, 200), (93004, 200), (600001, 133), (600001, 130), (70003, 200)):
            with self.assertRaises(ValueError, msg=(base, exp)):
                rr.audit_seeds(plan_for(4, base, exp, 2))

    def test_audit_accepts_disjoint_plan(self):
        rr.audit_seeds(SYNTH)


class PlayPairTests(unittest.TestCase):
    def play(self, winner_of=lambda matches: 0):
        made, matches = [], []

        def factory(spec, seed):
            policy = SimpleNamespace(spec=spec, seed=seed)
            made.append(policy)
            return policy

        def play_match(a, b, **kwargs):
            matches.append((a, b, kwargs))
            return SimpleNamespace(winner=winner_of(matches))
        with patch('phase2_final_exam.play_match', play_match):
            record = rr.play_pair(SYNTH, (1, 3), 4, 1, ['S0', 'S1', 'S2', 'S3'], factory)
        return record, made, matches

    def test_lower_index_agent_sits_first_then_second_with_fresh_policies(self):
        record, made, matches = self.play()
        self.assertEqual([m[0].spec for m in matches], ['S1', 'S3'])
        self.assertEqual([m[1].spec for m in matches], ['S3', 'S1'])
        self.assertEqual(len({id(p) for p in made}), 4)          # nothing carried across the pair's two games
        seeds = [int(s) for game in record['action_seeds'] for s in game]
        self.assertEqual(len(set(seeds)), 4)
        self.assertEqual([p.seed for p in made], seeds)
        for _, _, kwargs in matches:
            self.assertEqual(kwargs, {'seed': record['deal_seed'], 'target': 3, 'starting_score': (0, 0),
                                      'deal_partition': 0})
        self.assertEqual(record['deal_seed'], fx.deal_seed(SYNTH, 4, 1))

    def test_pair_score_of_the_row_agent(self):
        record, _, _ = self.play(lambda m: len(m) - 1)           # game 0 won by seat 0, game 1 by seat 1
        self.assertEqual(fx.pair_score(record['winners']['games']), 1.0)
        record, _, _ = self.play(lambda m: 1 - (len(m) - 1))
        self.assertEqual(fx.pair_score(record['winners']['games']), 0.0)
        record, _, _ = self.play(lambda m: 0)                    # seat 0 wins both: one win each
        self.assertEqual(fx.pair_score(record['winners']['games']), 0.5)


class AnalysisTests(unittest.TestCase):
    def test_interval_means_and_symmetry(self):
        a = [1.0, 0.5, 0.0, 0.5] * 25
        result = rr.analyze({'games': [a]}, ['x', 'y'])
        pairs = {(p['row'], p['col']): p for p in result['pairs']}
        sd = statistics.stdev(a)
        self.assertAlmostEqual(pairs['x', 'y']['mean'], 0.5)
        self.assertAlmostEqual(pairs['x', 'y']['lower'], 0.5 - 1.96 * sd / 10)
        self.assertAlmostEqual(pairs['x', 'y']['upper'], 0.5 + 1.96 * sd / 10)
        self.assertAlmostEqual(pairs['y', 'x']['mean'], 0.5)
        self.assertEqual(pairs['x', 'y']['n'], 100)

    def test_equal_weight_agent_mean_over_opponents(self):
        n = 100
        scores = {'games': [[0.75 if i % 2 else 0.25 for i in range(n)],     # pairing (a, b): a mean 0.5
                            [1.0 if i % 2 else 0.5 for i in range(n)],       # pairing (a, c): a mean 0.75
                            [0.5] * n]}                                      # pairing (b, c)
        result = rr.analyze(scores, ['a', 'b', 'c'])
        agent = {x['id']: x for x in result['agents']}
        self.assertAlmostEqual(agent['a']['equal_weight_mean'], (0.5 + 0.75) / 2)
        self.assertAlmostEqual(agent['b']['equal_weight_mean'], (0.5 + 0.5) / 2)
        self.assertAlmostEqual(agent['c']['equal_weight_mean'], (0.25 + 0.5) / 2)
        self.assertEqual(agent['a']['opponents'], 2)
        self.assertLess(agent['a']['lower'], agent['a']['equal_weight_mean'])

    def test_rating_order_center_and_two_agent_fit(self):
        n = 200
        scores = {'games': [[0.75] * (n - 2) + [0.5, 1.0], [0.9] * (n - 2) + [0.8, 1.0],
                            [0.7] * (n - 2) + [0.6, 0.8]]}
        result = rr.analyze(scores, ['a', 'b', 'c'])
        elo = {x['id']: x['elo'] for x in result['agents']}
        self.assertGreater(elo['a'], elo['b'])
        self.assertGreater(elo['b'], elo['c'])
        self.assertAlmostEqual(sum(math.log(x['bradley_terry_strength']) for x in result['agents']), 0.0, places=6)
        self.assertIn('Descriptive', result['rating_note'])
        s = rr.bradley_terry([[0, 75], [25, 0]], prior=0)         # P(0 beats 1) equals the observed share
        self.assertAlmostEqual(s[0] / (s[0] + s[1]), 0.75, places=6)

    def test_rating_stays_finite_with_a_shutout(self):
        s = rr.bradley_terry([[0, 100], [0, 0]])
        self.assertTrue(all(math.isfinite(x) and x > 0 for x in s))
        self.assertGreater(s[0], s[1])


class BuildPairingsTests(unittest.TestCase):
    IDS = ['r', 'u', 'a1', 'a2', 'a3', 'b1']

    def test_cross_lists_a_times_b_then_within_without_self_or_duplicate_pairs(self):
        spec = {'mode': 'cross', 'a': ['a1', 'a2', 'b1'], 'b': ['r', 'b1'],
                'within': [{'name': 'x', 'agents': ['a1', 'a2', 'b1']}, {'name': 'y', 'agents': ['a3', 'u']}]}
        table = rr.build_pairings(self.IDS, spec)
        names = [(self.IDS[a], self.IDS[b]) for a, b in table]
        self.assertEqual(names, [('a1', 'r'), ('a1', 'b1'), ('a2', 'r'), ('a2', 'b1'), ('b1', 'r'),
                                 ('a1', 'a2'), ('a3', 'u')])    # b1-b1 skipped; a1-b1, a2-b1 not repeated
        self.assertEqual(len({frozenset(p) for p in table}), len(table))

    def test_unknown_agent_and_mode_are_refused(self):
        with self.assertRaises(ValueError):
            rr.build_pairings(self.IDS, {'mode': 'cross', 'a': ['zz'], 'b': ['r']})
        with self.assertRaises(ValueError):
            rr.build_pairings(self.IDS, {'mode': 'what'})
        self.assertEqual(len(rr.build_pairings(self.IDS, {'mode': 'all'})), 15)


class CrossAnalysisTests(unittest.TestCase):
    def test_run_tables_and_within_matrix(self):
        ids = ['a1', 'a2', 'b']
        spec = {'mode': 'cross', 'a': ['a1', 'a2'], 'b': ['b'], 'within': [{'name': 'run1', 'agents': ['a1', 'a2']}]}
        table = rr.build_pairings(ids, spec)                      # (a1,b) (a2,b) (a1,a2)
        n = 100
        scores = {'games': [[1.0 if i % 2 else 0.5 for i in range(n)], [0.0 if i % 2 else 0.5 for i in range(n)],
                            [0.75] * n]}
        result = rr.analyze(scores, ids, table, spec)
        run = result['runs'][0]
        self.assertAlmostEqual(run['table']['a1']['b']['mean'], 0.75)
        self.assertAlmostEqual(run['table']['a2']['b']['mean'], 0.25)
        self.assertLess(run['table']['a1']['b']['lower'], 0.75)
        self.assertAlmostEqual(run['within']['a1']['a2']['mean'], 0.75)
        self.assertAlmostEqual(run['within']['a2']['a1']['mean'], 0.25)
        self.assertNotIn('agents', result)                        # no rating outside mode 'all'
        self.assertEqual(len(result['pairs']), 6)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, raw):
        path = Path(self.tmp.name) / 'c.json'
        path.write_text(json.dumps(raw), encoding='utf-8')
        return path

    def base(self):
        return {'name': 'x', 'rule': 'RULE.txt', 'output': 'out',
                'plan': {'deal_base': 600001, 'action_exp': 200, 'count': 2, 'targets': [2, 3]},
                'seed_audit': 'since_run3', 'pairing': {'mode': 'all'},
                'models': {'R': {'kind': 'reactive'}, 'AU': {'kind': 'au'}},
                'agents': [{'id': 'r', 'name': 'R', 'model': 'R'}, {'id': 'a', 'name': 'A', 'model': 'AU'}]}

    def real(self, name):
        return rr.load_config(Path(__file__).resolve().parent.parent / f'experiments/{name}/config.json')

    def test_all_vs_all_config_has_8_agents_28_pairings_and_passes_the_audit(self):
        config = self.real('all-vs-all')
        self.assertEqual([a['id'] for a in config['agents']][:3], ['reactive', 'adaptive_utility', 'phase1_g48'])
        self.assertEqual((len(config['agents']), len(config['pairings'])), (8, 28))
        self.assertTrue(all(agent['name'] for agent in config['agents']))
        rr.audit_seeds(config['plan'], rr.SEED_AUDITS[config['seed_audit']])

    def test_run_evolution_config_pairings_and_disjoint_seeds(self):
        config = self.real('run-evolution')
        ids = [a['id'] for a in config['agents']]
        self.assertEqual(len(ids), 18)
        keys = [frozenset((ids[a], ids[b])) for a, b in config['pairings']]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(keys), 15 * 4 - 1 + 3 * 10 - 4)       # cross minus self pair, within minus duplicates
        self.assertEqual(config['plan']['deal_base'], 95001)
        self.assertEqual(config['plan']['action_exp'], 135)
        rr.audit_seeds(config['plan'], rr.SEED_AUDITS[config['seed_audit']])
        with self.assertRaises(ValueError):     # and it would collide with the all-vs-all if both used its seeds
            rr.audit_seeds(plan_for(8, 94001, 134, 1000), rr.SEED_AUDITS['since_all_vs_all'])
        rr.audit_seeds(plan_for(8, 94001, 134, 1000), rr.SEED_AUDITS['since_run3'])

    def test_every_model_file_hash_in_both_configs_is_filled(self):
        for name in ('all-vs-all', 'run-evolution'):
            for key, entry in self.real(name)['models'].items():
                if 'path' in entry:
                    self.assertEqual(len(entry['sha256']), 64, key)
                    self.assertEqual(entry['profile_prior'], 5) if entry['kind'] == 'phase2' else None

    def test_refuses_placeholder_hash_unknown_model_and_duplicate_ids(self):
        rr.load_config(self.write(self.base()), self.tmp.name)
        bad = self.base()
        bad['models']['P'] = {'kind': 'phase1', 'path': 'p.npy', 'sha256': 'TODO'}
        with self.assertRaises(ValueError):
            rr.load_config(self.write(bad), self.tmp.name)
        bad = self.base()
        bad['agents'][1]['model'] = 'nope'
        with self.assertRaises(ValueError):
            rr.load_config(self.write(bad), self.tmp.name)
        bad = self.base()
        bad['agents'][1]['id'] = 'r'
        with self.assertRaises(ValueError):
            rr.load_config(self.write(bad), self.tmp.name)


@unittest.skipUnless(Phase2Network is not None, 'PyTorch is optional')
class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        self.root = tmp / 'repo'
        (self.root / 'proyecto').mkdir(parents=True)
        (self.root / 'proyecto/helper.py').write_text('X = 1\n')
        (self.root / 'RULE.txt').write_text('rule\n')
        git(self.root, 'init', '-q')
        git(self.root, 'add', '-A')
        git(self.root, 'commit', '-q', '-m', 'fixture')
        models = tmp / 'models'
        models.mkdir()
        Phase2Network(seed=1).save(models / 'n.pt')
        np.save(models / 'p.npy', np.zeros(GENOME_SIZE))
        self.models = {'R': {'kind': 'reactive'}, 'AU': {'kind': 'au'},
                       'P1': {'kind': 'phase1', 'path': str(models / 'p.npy'), 'sha256': sha(models / 'p.npy')},
                       'N': {'kind': 'phase2', 'path': str(models / 'n.pt'), 'sha256': sha(models / 'n.pt'),
                             'profile_prior': 5}}
        self.agents = [{'id': k.lower(), 'name': k, 'model': k} for k in ('R', 'AU', 'P1', 'N')]
        self.out = tmp / 'out'

    def config(self, **edit):
        config = {'agents': self.agents, 'plan': SYNTH, 'pairing': {'mode': 'all'}, 'pairings': rr.pairings(4),
                  'seed_audit': 'since_run3', 'models': json.loads(json.dumps(self.models)),
                  'output': str(self.out), 'rule_path': str(self.root / 'RULE.txt')}
        config.update(edit)
        return config

    def run_it(self, config=None, **kwargs):
        args = dict(root=self.root, workers=2)
        args.update(kwargs)
        return rr.run(config or self.config(), **args)

    def lines(self):
        return (self.out / 'partial.jsonl').read_text(encoding='utf-8').splitlines()

    def test_full_run_writes_everything_and_validates_all_pairs(self):
        result = self.run_it()
        for name in ('started.lock', 'config.json', 'partial.jsonl', 'results.json', 'complete.txt'):
            self.assertTrue((self.out / name).exists(), name)
        records = [json.loads(x) for x in self.lines()]
        self.assertEqual(len(records), 6 * 2)
        self.assertEqual(len(fx.validate_records(records, SYNTH, ['games'])), 12)
        config = json.loads((self.out / 'config.json').read_text())
        self.assertEqual(config['models']['P1']['sha256'], self.models['P1']['sha256'])
        self.assertEqual(config['sha256']['proyecto/helper.py'], sha(self.root / 'proyecto/helper.py'))
        self.assertEqual(len(config['pairings']), 6)
        self.assertEqual(len(result['analysis']['pairs']), 12)
        self.assertEqual(len(result['analysis']['agents']), 4)

    def test_second_run_without_resume_is_refused(self):
        self.run_it()
        with self.assertRaises(FileExistsError):
            self.run_it()

    def test_resume_plays_only_missing_pairs_and_reproduces_results(self):
        self.run_it()
        original = (self.out / 'results.json').read_bytes()
        lines = self.lines()
        (self.out / 'partial.jsonl').write_text('\n'.join(lines[:5]) + '\n', encoding='utf-8')
        (self.out / 'results.json').unlink()
        (self.out / 'complete.txt').unlink()
        played = []

        def spy(config, jobs, workers):
            played.extend(jobs)
            return rr._play_jobs(config, jobs, workers)
        self.run_it(resume=True, play_jobs=spy)
        kept = {(json.loads(x)['block'], json.loads(x)['i']) for x in lines[:5]}
        self.assertEqual(len(played), 7)
        self.assertFalse(kept & set(map(tuple, played)))
        self.assertEqual((self.out / 'results.json').read_bytes(), original)
        self.assertEqual(self.lines()[:5], lines[:5])

    def test_refuses_hash_mismatch_audit_overlap_and_dirty_tree_without_output(self):
        bad = self.config()
        bad['models']['P1']['sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            self.run_it(bad)
        with self.assertRaises(ValueError):
            self.run_it(self.config(plan=plan_for(4, 93001, 134, 2, (2, 3))))
        (self.root / 'proyecto/helper.py').write_text('X = 2\n')
        with self.assertRaises(ValueError):
            self.run_it()
        self.assertFalse(self.out.exists())

    def test_refuses_too_many_workers(self):
        with self.assertRaises(ValueError):
            self.run_it(workers=13)

    def test_specs_per_agent_and_greedy_phase2(self):
        specs = rr.build_specs(self.agents, self.config()['models'])
        self.assertEqual([s['kind'] for s in specs], ['reactive', 'au', 'phase1', 'phase2'])
        self.assertIsNone(specs[3]['temperature'])               # greedy


if __name__ == '__main__':
    unittest.main()
