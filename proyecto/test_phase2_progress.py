import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from phase2_progress import ProgressReporter, generation_line, write_json_atomic
from phase2_torch import Phase2Network
from phase2_trainer import run_rounds, train


def make_config(root):
    opponent = root / 'opponent.pt'
    Phase2Network(seed=9).save(opponent)
    panel = root / 'panel.json'
    panel.write_text(json.dumps({'version': 0, 'measuring_only': True, 'members': [{'name': 'x', 'path': 'opponent.pt', 'kind': 'phase2', 'encoder': 'v1', 'sha256': hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
    return dict(generations=2, matches_per_generation=2, matches_per_round=1, block_size=1, target=1, panel_path=panel,
                champions_dir=root / 'champ', panel_pairs=2, head_to_head_pairs=2, clock=lambda: 0)


class ProgressFileTest(unittest.TestCase):
    def test_file_is_valid_after_every_round_with_expected_fields(self):
        snapshots = []
        original = ProgressReporter.round_done
        def spy(self, **kw):
            original(self, **kw)
            snapshots.append(json.loads(self.path.read_text(encoding='utf-8')))
        with tempfile.TemporaryDirectory() as directory, patch.object(ProgressReporter, 'round_done', spy):
            root = Path(directory)
            train(root / 'run', **make_config(root))
            final = json.loads((root / 'run/progress.json').read_text(encoding='utf-8'))
        self.assertEqual([s['round'] for s in snapshots], [1, 2, 3, 4])
        for s in snapshots:
            for key in ('elapsed', 'time_cap', 'generation', 'round', 'matches', 'samples', 'updates', 'matches_per_s',
                        'samples_per_s', 'recent_loss', 'league_mix', 'temperature', 'champion_generation',
                        'promotions', 'panel_version', 'status', 'stop_reason'):
                self.assertIn(key, s)
        self.assertEqual(snapshots[0]['status'], 'training')
        self.assertEqual(snapshots[1]['generation'], 1)
        self.assertEqual(final['matches'], 4)
        self.assertEqual(final['panel_version'], 0)
        self.assertEqual(final['status'], 'ended')
        self.assertEqual(final['stop_reason'], 'completed')
        self.assertAlmostEqual(sum(final['league_mix'].values()), 1.0)
        self.assertEqual(final['champion_generation'], 1)

    def test_no_progress_file_when_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(root / 'run', progress=False, **make_config(root))
            self.assertFalse((root / 'run/progress.json').exists())

    def test_progress_does_not_change_log_or_checkpoint_and_resume_is_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            train(root / 'off', progress=False, **config)
            train(root / 'on', console=io.StringIO(), **config)
            train(root / 'res', stop_after_round=3, **config)
            train(root / 'res', resume=root / 'res/checkpoint_000003.pt', **config)
            def read(name):
                return (root / name / 'generations.jsonl').read_text(encoding='utf-8')
            self.assertEqual(read('off'), read('on'))
            self.assertEqual(read('off'), read('res'))
            states = [torch.load(root / n / 'checkpoint_000004.pt', weights_only=False) for n in ('off', 'on', 'res')]
            for other in states[1:]:
                self.assertEqual(set(states[0]), set(other))
                for key in states[0]['model']:
                    self.assertTrue(torch.equal(states[0]['model'][key], other['model'][key]))
            self.assertNotIn('progress', json.dumps(states[0]['config']))
            self.assertNotIn('progress', read('on'))
            self.assertTrue((root / 'res/progress.json').exists())

    def test_interrupted_run_stays_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(root / 'run', stop_after_round=3, **make_config(root))
            progress = json.loads((root / 'run/progress.json').read_text(encoding='utf-8'))
        self.assertEqual((progress['round'], progress['status']), (3, 'training'))

    def test_resume_keeps_promotion_count(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'generations.jsonl').write_text('{"promoted": true}\n{"promoted": false}\n{"promoted": true}\n')
            seen = {}
            original = ProgressReporter.__init__
            def spy(self, *a, **k):
                original(self, *a, **k)
                seen['promotions'] = self.promotions
            with patch.object(ProgressReporter, '__init__', spy):
                run_rounds(rounds=1, matches_per_round=4, generation_rounds=1, log_path=run / 'generations.jsonl',
                           stop_after_round=0, progress_path=run / 'progress.json')
            self.assertEqual(seen['promotions'], 2)


class AtomicWriteTest(unittest.TestCase):
    def test_uses_temp_file_fsync_and_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'progress.json'
            calls = []
            real_replace, real_fsync = os.replace, os.fsync
            def replace(a, b):
                calls.append(('replace', Path(a).name, Path(b).name))
                real_replace(a, b)
            def fsync(fd):
                calls.append(('fsync',))
                real_fsync(fd)
            with patch('phase2_progress.os.replace', side_effect=replace), patch('phase2_progress.os.fsync', side_effect=fsync):
                write_json_atomic(target, {'a': 1})
            self.assertEqual(calls, [('fsync',), ('replace', 'progress.json.tmp', 'progress.json')])
            self.assertEqual(json.loads(target.read_text()), {'a': 1})
            self.assertFalse((Path(directory) / 'progress.json.tmp').exists())

    def test_old_file_survives_when_replace_keeps_failing(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'progress.json'
            target.write_text('{"old": true}')
            with patch('phase2_progress.os.replace', side_effect=PermissionError), patch('phase2_progress.time.sleep'):
                self.assertFalse(write_json_atomic(target, {'new': True}))
            self.assertEqual(json.loads(target.read_text()), {'old': True})


class TerminalTest(unittest.TestCase):
    def test_generation_line_formats(self):
        promoted = dict(generation=24, champion=24, check=dict(gain=.018, lower_bound=.007, upper_bound=.029, promoted=True))
        self.assertEqual(generation_line(promoted, 3 * 3600 + 12 * 60 + 5), 'gen 24 · mejora +1.8 pts (±1.1) · asciende · 3h12m')
        kept = dict(generation=25, champion=24, check=dict(gain=-.004, lower_bound=-.02, upper_bound=.012, promoted=False))
        self.assertIn('mejora -0.4 pts (±1.6) · se mantiene', generation_line(kept, 600))
        first = dict(generation=1, champion=1, check=None)
        self.assertEqual(generation_line(first, 60), 'gen 1 · primera campeona · 1m')

    def test_round_lines_overwrite_in_place_and_generation_line_is_permanent(self):
        out = io.StringIO()
        reporter = ProgressReporter(None, out, wall=iter([0, 1, 2]).__next__)
        for r in (1, 2):
            reporter.round_done(elapsed=r, generation=0, round=r, matches=r * 4, samples=r * 100, updates=r, recent_loss=.5,
                                league_mix={'self': 4}, champion_generation=None, panel_version=1)
        reporter.generation_done(dict(generation=1, champion=1, check=None), 2)
        text = out.getvalue()
        self.assertEqual(text.count('\n'), 1)
        self.assertTrue(text.startswith('\rgen 1 · ronda 1'))
        self.assertIn('\rgen 1 · ronda 2', text)
        self.assertTrue(text.endswith('\n'))
        self.assertTrue(text.rstrip().endswith('primera campeona · 0m'))


if __name__ == '__main__':
    unittest.main()
