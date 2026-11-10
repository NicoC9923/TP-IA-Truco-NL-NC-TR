"""Run 3 pilot tool (ADR 0062): sizing, seed audit and a tiny end-to-end run on synthetic inputs only."""
import json
import tempfile
import unittest
from pathlib import Path
import phase2_final_exam as exam
import phase2_run3_pilot as pilot
from test_phase2_run3_trainer import noisy_network, small_table


class SizingTest(unittest.TestCase):
    def test_matches_per_generation_is_a_whole_number_of_rounds(self):
        self.assertEqual(pilot.matches_per_generation(96, seconds=500, per_round=240, check_share=0), 48000)
        self.assertEqual(pilot.matches_per_generation(100, seconds=450, per_round=240, check_share=.1), 40560)
        self.assertEqual(pilot.matches_per_generation(1, seconds=10, per_round=240, check_share=0), 240)  # at least one round

    def test_invalid_sizing_inputs(self):
        for bad in ({'rate': 0}, {'rate': 100, 'check_share': 1}, {'rate': 100, 'seconds': 0}):
            with self.assertRaises(ValueError):
                pilot.matches_per_generation(**{'seconds': 450, 'per_round': 240, 'check_share': 0, **bad})


class SeedAuditTest(unittest.TestCase):
    def test_proposed_seeds_are_new(self):
        audit = pilot.audit_seeds()
        self.assertEqual(audit['conflicts'], [])
        self.assertEqual(audit['pilot_plan']['count'], 300)

    def test_pilot_plan_is_disjoint_from_the_final_exam_and_earlier_namespaces(self):
        deals, actions = exam.all_seeds(pilot.PILOT_PLAN)
        for plan in (exam.PART_T_PLAN, exam.PART_E_PLAN):
            other_deals, other_actions = exam.all_seeds(plan)
            self.assertFalse(deals & other_deals)
            self.assertFalse(actions & other_actions)
        for low, high in exam.EARLIER_ACTION_RANGES:
            self.assertFalse(any(low <= a <= high for a in actions))
        schedule_seeds = {pilot.PILOT_PLAN['deal_base'] + b for b in range(len(pilot.PILOT_PLAN['opponents']))}
        self.assertFalse(schedule_seeds & exam.EARLIER_SCHEDULE_SEEDS)

    def test_a_reused_seed_is_reported(self):
        audit = pilot.audit_seeds(training_seed=57001)
        self.assertIn('pilot_training 57001', ' '.join(audit['conflicts']))


class PilotRunTest(unittest.TestCase):
    def test_tiny_pilot_trains_evaluates_and_reports(self):
        shares = {'self': 1.0, 'phase1': 0, 'snapshot': 0, 'bluffer': 0, 'cautious': 0, 'aggressive': 0, 'calculating': 0}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            warm = root / 'warm.pt'
            noisy_network(3).save(warm)
            table = root / 'equity.json'
            small_table().save(table)
            result = pilot.run_pilot(warm_start=warm, equity_table=table, output=root / 'pilot', updates=3,
                                     samples_per_match=35, matches_per_round=4, block_size=4, workers=2, pairs=3,
                                     minibatch=64, league_shares=shares, held_out_manifest=None)
            saved = json.loads((root / 'pilot/pilot.json').read_text())
            self.assertEqual(saved['evaluation']['pairs'], 3)
            for name in ('warm_start', 'pilot'):
                self.assertTrue(0 <= saved['evaluation'][name]['mean'] <= 1)
            self.assertIsInstance(saved['passes'], bool)
            self.assertEqual(saved['passes'], saved['evaluation']['pilot']['mean'] >= .68)
            self.assertGreater(saved['throughput']['matches_per_second'], 0)
            self.assertEqual(saved['generation_sizing']['matches_per_generation'] % 4, 0)
            self.assertGreaterEqual(saved['training']['updates'], 1)
            self.assertEqual(saved['seeds']['evaluation']['deal_base'], pilot.PILOT_PLAN['deal_base'])
            self.assertTrue((root / 'pilot/pilot_final.pt').exists())
            self.assertEqual(result['passes'], saved['passes'])

    def test_refuses_an_existing_output_folder_and_too_many_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileExistsError):
                pilot.run_pilot(warm_start='w', equity_table='t', output=root)
            with self.assertRaises(ValueError):
                pilot.run_pilot(warm_start='w', equity_table='t', output=root / 'new', workers=13)


if __name__ == '__main__':
    unittest.main()
