"""Run 3 support of the final exam tool: no Part T, all greedy, encoder by input width, identity refusals.
Synthetic fixtures only; never the real exam seeds or model files."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import phase2_final_exam as fx
from test_phase2_final_exam import RUN2_CONFIG, RUN3_CONFIG, REPO, SYNTH_E, RunnerTestCase, sha, torch

if torch is not None:
    from phase2_network import INPUT_SIZES
    from phase2_torch import Phase2Network


@unittest.skipUnless(torch is not None, 'PyTorch is optional')
class NoPartTExamTests(RunnerTestCase):
    def greedy_evaluated(self):
        return [{'name': n, 'model': n, 'temperature': None} for n in ('C1', 'C2', 'B')]

    def part_e_no_t(self, **kwargs):
        args = dict(root=self.root, output=self.out / 'part-e', workers=2, models=self.models, plan=SYNTH_E,
                    rule_path=self.root / 'RULE.txt', evaluated=self.greedy_evaluated())
        args.update(kwargs)
        return fx.run_part_e(**args)

    def test_part_e_runs_without_part_t_and_every_network_plays_greedy(self):
        result = self.part_e_no_t()
        config = json.loads((self.out / 'part-e/config.json').read_text())
        self.assertNotIn('tstar', config)
        self.assertEqual([(e['name'], e['temperature']) for e in config['evaluated']],
                         [('C1', None), ('C2', None), ('B', None)])
        self.assertEqual(len(self.lines(self.out / 'part-e')), 5 * 2)
        self.assertIn(result['analysis']['decision'], ('C1', 'C2', 'B'))

    def test_refuses_without_explicit_evaluated_or_with_tstar_but_no_part_t(self):
        with self.assertRaises(ValueError):
            self.part_e_no_t(evaluated=None)
        with self.assertRaises(ValueError):
            self.part_e_no_t(evaluated=[{'name': 'C1', 'model': 'C1', 'temperature': 'tstar'}])
        self.assertFalse((self.out / 'part-e').exists())

    def test_each_network_is_loaded_with_the_encoder_of_its_input_width(self):
        v3 = Path(self.tmp.name) / 'models/c2.pt'
        Phase2Network(seed=2, input_size=INPUT_SIZES[3]).save(v3)
        self.models['C2'].update(sha256=sha(v3), input_size=INPUT_SIZES[3])
        self.models['B']['input_size'] = INPUT_SIZES[1]
        self.part_e_no_t()
        specs, _ = fx._build_specs({'models': self.models, 'plan': SYNTH_E, 'evaluated': self.greedy_evaluated()})
        self.assertEqual({n: s['model'].encoder_version for n, s in specs.items()}, {'C1': 1, 'C2': 3, 'B': 1})

    def test_refuses_a_network_whose_input_width_differs_from_the_config(self):
        self.models['C1']['input_size'] = INPUT_SIZES[3]
        with self.assertRaises(ValueError):
            self.part_e_no_t()
        self.assertFalse((self.out / 'part-e').exists())

    def log_with_checks(self, with_check, last):
        path = Path(self.tmp.name) / 'generations.jsonl'
        rows = [{'generation': g, 'check': {'gain': 0.0} if g in with_check else None} for g in range(1, last + 1)]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def test_c2_must_be_the_last_generation_with_a_check(self):
        folder = Path(self.tmp.name) / 'gens'
        folder.mkdir()
        for g in (4, 5):
            shutil.copy(Path(self.tmp.name) / 'models/c2.pt', folder / f'generation_{g:06d}.pt')
        path = self.log_with_checks({2, 4}, last=5)  # generation 5 was never checked
        self.models['C2'].update(path=str(folder / 'generation_000004.pt'), sha256=sha(folder / 'generation_000004.pt'),
                                 last_checked={'log': str(path), 'generation': 4})
        self.part_e_no_t()
        later = json.loads(json.dumps(self.models))
        later['C2'].update(path=str(folder / 'generation_000005.pt'), sha256=sha(folder / 'generation_000005.pt'),
                           last_checked={'log': str(path), 'generation': 5})
        with self.assertRaises(ValueError):
            self.part_e_no_t(models=later, output=self.out / 'other')
        wrong = json.loads(json.dumps(self.models))
        wrong['C2']['last_checked']['generation'] = 2
        with self.assertRaises(ValueError):
            self.part_e_no_t(models=wrong, output=self.out / 'other')
        self.assertFalse((self.out / 'other').exists())

    def test_last_checked_generation_reads_the_last_row_with_a_check(self):
        self.assertEqual(fx.last_checked_generation(self.log_with_checks({3, 6}, last=8)), 6)


class Run3ConfigTests(unittest.TestCase):
    def filled(self):
        raw = json.loads(RUN3_CONFIG.read_text(encoding='utf-8'))

        def fill(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, str) and value.startswith('TODO'):
                        node[key] = 'a' * 64 if key == 'sha256' else 1
                    elif isinstance(value, str):
                        node[key] = value.replace('NNNNNN', '000001')
                    fill(node[key])
        fill(raw)
        return raw

    def load(self, raw):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(raw))
            return fx.load_exam_config(path, root=REPO)

    def test_draft_config_is_refused_while_hashes_are_placeholders(self):
        with self.assertRaises(ValueError) as caught:
            fx.load_exam_config(RUN3_CONFIG, root=REPO)
        self.assertIn('sha256', str(caught.exception))

    def test_other_placeholders_are_refused_too(self):
        raw = self.filled()
        raw['models']['C2']['last_checked']['generation'] = 'TODO-last-generation-with-a-check'
        with self.assertRaises(ValueError):
            self.load(raw)
        raw = self.filled()
        raw['models']['C2']['path'] = 'training_output/phase2-run3/generation_NNNNNN.pt'
        with self.assertRaises(ValueError):
            self.load(raw)

    def test_run3_plan_seeds_and_evaluated_networks_follow_the_exam_design(self):
        exam = self.load(self.filled())
        self.assertIsNone(exam['part_t'])
        plan = exam['part_e']
        self.assertEqual((plan['deal_base'], plan['action_exp'], plan['count'], plan['opponents']),
                         (93001, 133, 1000, ['AU', 'R', 'P1', 'H1', 'H2']))
        self.assertEqual(fx.seed_ranges(plan), ((186002000000000, 186010000001998), (2 ** 133, 2 ** 133 + 19999)))
        self.assertEqual([(e['name'], e['model'], e['temperature']) for e in exam['evaluated']],
                         [('C1', 'C1', None), ('C2', 'C2', None), ('B', 'B', None)])
        models = exam['models']
        self.assertEqual((models['C1']['input_size'], models['C2']['input_size'], models['B']['input_size']),
                         (1635, 1635, 1628))
        self.assertEqual(models['C1']['sidecar']['require']['selection'], 'greedy')
        self.assertEqual(models['B']['sidecar']['require']['selection'], 'greedy')
        self.assertIn('last_checked', models['C2'])
        self.assertTrue(exam['output_e'].replace('\\', '/').endswith('experiments/phase2-run3-exam/part-e'))
        fx.audit_exam_seeds(exam)

    def test_evaluated_networks_without_an_explicit_temperature_are_refused(self):
        raw = self.filled()
        del raw['evaluated'][0]['temperature']
        with self.assertRaises(ValueError):
            self.load(raw)

    def test_since_run2_audit_covers_every_namespace_used_after_run_2(self):
        schedule, actions = fx.SEED_AUDITS['since_run2']
        for seed in (92001, 67001, 67002, 63001, 65001, 66000, 66999, 90001, 90002, *range(91001, 91006)):
            self.assertIn(seed, schedule)
        for exponent, size in ((130, 20000), (131, 2400), (132, 1200)):
            self.assertIn((2 ** exponent, 2 ** exponent + size - 1), actions)

    def test_audit_refuses_plans_that_reuse_a_namespace(self):
        exam = self.load(self.filled())
        for deal_base, exp in ((92001, 133), (93001, 132), (66003, 133)):
            bad = {**exam, 'part_e': fx.make_plan('E', deal_base, exp, 1000, exam['part_e']['opponents'])}
            with self.assertRaises(ValueError):
                fx.audit_exam_seeds(bad)
        old = fx.load_exam_config(RUN2_CONFIG, root=REPO)
        with self.assertRaises(ValueError):
            fx.audit_exam_seeds({**old, 'seed_audit': 'since_run2'})


if __name__ == '__main__':
    unittest.main()
