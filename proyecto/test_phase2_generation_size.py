"""One logged change of matches per generation on resume (ADR 0062 amendment)."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import phase2_trainer as trainer
from phase2_schedule import GenerationSchedule
from phase2_torch import Phase2Network


class ScheduleTest(unittest.TestCase):
    def test_fixed_size_matches_the_old_arithmetic(self):
        schedule = GenerationSchedule(5)
        for r in range(1, 40):
            self.assertEqual(schedule.is_boundary(r), r % 5 == 0)
            self.assertEqual(schedule.generation_at(r), r // 5)
            self.assertEqual(schedule.last_boundary(r), r // 5 * 5)
        self.assertEqual(schedule.total_rounds(7), 35)
        self.assertEqual(schedule.previous_boundary(15), 10)

    def test_one_change_at_a_boundary(self):
        schedule = GenerationSchedule(4).with_change(round=8, rounds_per_generation=3)
        self.assertEqual([r for r in range(1, 21) if schedule.is_boundary(r)], [4, 8, 11, 14, 17, 20])
        self.assertEqual([schedule.generation_at(r) for r in (4, 7, 8, 9, 11, 12, 14)], [1, 1, 2, 2, 3, 3, 4])
        self.assertEqual(schedule.last_boundary(10), 8)
        self.assertEqual(schedule.previous_boundary(11), 8)
        self.assertEqual(schedule.previous_boundary(8), 4)
        self.assertEqual(schedule.total_rounds(2), 8)
        self.assertEqual(schedule.total_rounds(5), 8 + 9)
        self.assertEqual(schedule.record()['change_generation'], 2)

    def test_change_must_be_at_a_boundary_and_only_once(self):
        with self.assertRaises(ValueError):
            GenerationSchedule(4).with_change(round=6, rounds_per_generation=3)
        changed = GenerationSchedule(4).with_change(round=8, rounds_per_generation=3)
        with self.assertRaises(ValueError):
            changed.with_change(round=11, rounds_per_generation=2)

    def test_record_round_trip(self):
        changed = GenerationSchedule(4).with_change(round=8, rounds_per_generation=3)
        again = GenerationSchedule.from_record(changed.record())
        self.assertEqual([again.generation_at(r) for r in range(1, 30)], [changed.generation_at(r) for r in range(1, 30)])


def standin_panel(root):
    opponent = root / 'opponent.pt'
    Phase2Network(seed=9).save(opponent)
    panel = root / 'panel.json'
    panel.write_text(json.dumps({'version': 1, 'members': [{
        'name': 'standin', 'path': 'opponent.pt', 'kind': 'phase2', 'encoder': 'v1',
        'sha256': hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
    return panel


class ResumeChangeTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.promotion = patch('phase2_trainer.promotion_decision', return_value={
            'gain': 0, 'lower_bound': 0, 'upper_bound': 0, 'head_to_head': .5, 'promoted': False})
        self.promotion.start()

    def tearDown(self):
        self.promotion.stop()
        self.folder.cleanup()

    def options(self, matches_per_generation, **extra):
        return dict(generations=5, matches_per_generation=matches_per_generation, matches_per_round=2, block_size=2,
                    target='mixed', panel_path=standin_panel(self.root), champions_dir=self.root / 'champions',
                    panel_pairs=2, head_to_head_pairs=2, clock=lambda: 0, progress=False, **extra)

    def rows(self, name):
        data = [json.loads(x) for x in (self.root / name / 'generations.jsonl').read_text().splitlines()]
        for row in data:
            row.pop('elapsed_seconds', None)
        return data

    def test_change_is_applied_logged_and_deterministic_afterwards(self):
        run = self.root / 'run'
        trainer.train(run, stop_after_round=4, **self.options(4))  # generations of 2 rounds, 2 done
        trainer.train(run, resume=run / 'checkpoint_000004.pt', change_matches_per_generation=True,
                      stop_after_round=8, **self.options(6))  # now 3 rounds each
        trainer.train(run, resume=run / 'checkpoint_000008.pt', **self.options(6))
        rows = self.rows('run')
        self.assertEqual([r['generation'] for r in rows], [1, 2, 3, 4, 5])
        self.assertEqual([r['matches'] for r in rows], [4, 4, 6, 6, 6])
        change = json.loads((run / 'generation_size_change.json').read_text())
        self.assertEqual((change['from_matches_per_generation'], change['to_matches_per_generation'], change['at_round']), (4, 6, 4))
        self.assertEqual(rows[2]['generation_size_change']['to_matches_per_generation'], 6)
        self.assertNotIn('generation_size_change', rows[3])
        state = torch.load(run / 'checkpoint_000013.pt', weights_only=False)
        self.assertEqual(state['generation_schedule']['change_round'], 4)
        # A straight resume of the same checkpoints gives the same log.
        again = self.root / 'again'
        trainer.train(again, stop_after_round=4, **self.options(4))
        trainer.train(again, resume=again / 'checkpoint_000004.pt', change_matches_per_generation=True, **self.options(6))
        self.assertEqual(self.rows('run'), self.rows('again'))

    def test_changing_the_size_needs_the_explicit_flag(self):
        run = self.root / 'run'
        trainer.train(run, stop_after_round=4, **self.options(4))
        with self.assertRaises(ValueError):
            trainer.train(run, resume=run / 'checkpoint_000004.pt', **self.options(6))

    def test_only_one_change_and_only_at_a_generation_boundary(self):
        run = self.root / 'run'
        trainer.train(run, stop_after_round=3, **self.options(4))
        with self.assertRaises(ValueError):
            trainer.train(run, resume=run / 'checkpoint_000003.pt', change_matches_per_generation=True, **self.options(6))
        trainer.train(run, resume=run / 'checkpoint_000003.pt', stop_after_round=4, **self.options(4))
        trainer.train(run, resume=run / 'checkpoint_000004.pt', change_matches_per_generation=True,
                      stop_after_round=7, **self.options(6))
        with self.assertRaises(ValueError):
            trainer.train(run, resume=run / 'checkpoint_000007.pt', change_matches_per_generation=True, **self.options(8))
        with self.assertRaises(ValueError):  # the old size is no longer accepted after the change
            trainer.train(run, resume=run / 'checkpoint_000007.pt', **self.options(4))

    def test_flag_without_a_new_size_is_refused(self):
        run = self.root / 'run'
        trainer.train(run, stop_after_round=4, **self.options(4))
        with self.assertRaises(ValueError):
            trainer.train(run, resume=run / 'checkpoint_000004.pt', change_matches_per_generation=True, **self.options(4))


if __name__ == '__main__':
    unittest.main()
