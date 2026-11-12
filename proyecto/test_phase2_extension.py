import tempfile
import unittest
from pathlib import Path
from phase2_trainer import train

class ExtensionTest(unittest.TestCase):
    def test_requires_resume_and_finite_cap(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'requires.*resume'):
                train(root, extend_time_cap=10)

    def test_cap_validation_and_default_resume(self):
        import torch
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            ticks = iter(range(1000))
            cfg = dict(generations=3, matches_per_generation=4, matches_per_round=4,
                       target=1, panel_path=None, time_cap_seconds=1, clock=lambda: next(ticks))
            train(root, **cfg)
            ckpt = root / 'checkpoint_000001.pt'
            log = (root / 'generations.jsonl').read_bytes()
            train(root, resume=ckpt, **cfg)
            self.assertEqual(log, (root / 'generations.jsonl').read_bytes())
            for cap in (1, 0, float('nan')):
                with self.assertRaises(ValueError):
                    train(root, resume=ckpt, extend_time_cap=cap, **cfg)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train(root, resume=ckpt, extend_time_cap=100, **{**cfg, 'seed': 7})
            state = torch.load(ckpt, weights_only=False)
            state['stop_reason'] = 'patience'
            torch.save(state, ckpt)
            with self.assertRaisesRegex(ValueError, 'time.cap'):
                train(root, resume=ckpt, extend_time_cap=100, **cfg)

    def test_boundary_extension_matches_long_run_and_resume(self):
        import hashlib, json
        from unittest.mock import patch
        import torch
        from phase2_torch import Phase2Network
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opponent = root / 'opponent.pt'; Phase2Network(seed=9).save(opponent)
            panel = root / 'panel.json'
            panel.write_text(json.dumps({'version': 1, 'members': [{'name': 'x', 'path': opponent.name, 'kind': 'phase2', 'encoder': 'v1', 'sha256': hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
            def check(model, champion, panel, *args):
                return {'gain': 0, 'lower_bound': 0, 'upper_bound': 0, 'head_to_head': .5, 'promoted': False, 'members': {'x': {'challenger': [.5, .5], 'champion': [.5, .5]}}, 'direct': [.5, .5], 'panel_version': 1}
            cfg = dict(generations=4, matches_per_generation=4, matches_per_round=4, target=1, panel_path=panel, champions_dir=None, panel_pairs=2, head_to_head_pairs=2)
            with patch('phase2_trainer._selection_check', side_effect=check):
                train(root/'full', time_cap_seconds=100, clock=lambda: 0, **cfg)
                ticks = iter([0,0,0,0,0,10,10,10,10,10,10,10,10])
                train(root/'ext', time_cap_seconds=5, clock=lambda: next(ticks), **cfg)
                ckpt = root/'ext/checkpoint_000002.pt'
                train(root/'ext', time_cap_seconds=5, extend_time_cap=100, resume=ckpt, clock=lambda: 0, stop_after_round=3, **cfg)
                train(root/'ext', time_cap_seconds=5, extend_time_cap=100, resume=root/'ext/checkpoint_000003.pt', clock=lambda: 0, **cfg)
            a = torch.load(root/'full/checkpoint_000004.pt', weights_only=False)
            b = torch.load(root/'ext/checkpoint_000004.pt', weights_only=False)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]))
            for key, values in a['optimizer']['state'].items():
                for name, value in values.items():
                    self.assertTrue(torch.equal(value, b['optimizer']['state'][key][name]))
            self.assertTrue(torch.equal(a['torch_rng'], b['torch_rng']))
            def rows(path):
                result = [json.loads(line) for line in path.read_text().splitlines()]
                for row in result: row.pop('elapsed_seconds')
                return result
            self.assertEqual(rows(root/'full/generations.jsonl'), rows(root/'ext/generations.jsonl'))
            record = json.loads((root/'ext/time_cap_extension.json').read_text())
            self.assertEqual(json.loads((root/'ext/progress.json').read_text())['time_cap_extension'], record)
            self.assertEqual(record['old_cap'], 5)
            self.assertEqual(record['new_cap'], 100)
            self.assertEqual(len(record['checkpoint_sha256']), 64)
            self.assertTrue(record['timestamp'])
            self.assertTrue(record['code_revision'])
            self.assertEqual(b['extension'], record)
            record_bytes = (root/'ext/time_cap_extension.json').read_bytes()
            train(root/'ext', time_cap_seconds=5, extend_time_cap=100, resume=root/'ext/checkpoint_000004.pt', clock=lambda: 0, **cfg)
            self.assertEqual((root/'ext/time_cap_extension.json').read_bytes(), record_bytes)
            self.assertEqual(a['champion_generation'], b['champion_generation'])
            self.assertEqual(a['shuffle_rng'], b['shuffle_rng'])
            self.assertEqual(a['python_rng'], b['python_rng'])
            self.assertEqual(a['numpy_rng'][0], b['numpy_rng'][0])
            import numpy as np
            self.assertTrue(np.array_equal(a['numpy_rng'][1], b['numpy_rng'][1]))
            self.assertEqual(a['numpy_rng'][2:], b['numpy_rng'][2:])

    def test_partial_generation_extension_and_legacy_checkpoint(self):
        import json, torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = dict(generations=2, matches_per_generation=8, matches_per_round=4,
                       target=1, panel_path=None)
            train(root/'full', time_cap_seconds=100, clock=lambda: 0, **cfg)
            ticks = iter(range(1000))
            train(root/'ext', time_cap_seconds=1, clock=lambda: next(ticks), **cfg)
            ckpt = root/'ext/checkpoint_000001.pt'
            legacy = torch.load(ckpt, weights_only=False)
            for field in ('auxiliary', 'auxiliary_weight', 'card_prediction_situations'):
                legacy['config'].pop(field, None)
            torch.save(legacy, ckpt)
            train(root/'ext', time_cap_seconds=1, extend_time_cap=100, resume=ckpt, clock=lambda: 0, **cfg)
            a = torch.load(root/'full/checkpoint_000004.pt', weights_only=False)
            b = torch.load(root/'ext/checkpoint_000004.pt', weights_only=False)
            self.assertTrue(all(torch.equal(a['model'][key], b['model'][key]) for key in a['model']))
            def semantic(path):
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows: row.pop('elapsed_seconds')
                return rows
            self.assertEqual(semantic(root/'full/generations.jsonl'), semantic(root/'ext/generations.jsonl'))

    def test_pending_boundary_rotates_panel(self):
        import hashlib, json
        import numpy as np
        from unittest.mock import patch
        from phase1_network import GENOME_SIZE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); members = []
            for i in range(4):
                path = root / f'm{i}.npy'; np.save(path, np.zeros(GENOME_SIZE) + i)
                members.append({'name': f'm{i}', 'path': path.name, 'kind': 'phase1', 'encoder': 'v1', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            panel = root/'panel.json'; panel.write_text(json.dumps({'version': 1, 'members': members}))
            def check(model, champion, panel, *args):
                marker = float(next(iter(model.state_dict().values())).sum())
                return {'gain': marker, 'lower_bound': -1, 'upper_bound': 1, 'head_to_head': .5, 'promoted': False, 'members': {m['name']: {'challenger': [(.5 + (marker % .1)) if i % 2 else .5]*2, 'champion': [.8, .8]} for i, m in enumerate(panel['members'])}, 'direct': [.5, .5], 'panel_version': panel['version']}
            cfg = dict(generations=5, matches_per_generation=4, matches_per_round=4, target=1, panel_path=panel, champions_dir=None, panel_pairs=2, head_to_head_pairs=2, saturation_checks=1, saturation_pairs=2)
            with patch('phase2_trainer._selection_check', side_effect=check), patch('phase2_trainer._champion_overlap', return_value={}):
                train(root/'full', time_cap_seconds=100, clock=lambda: 0, **cfg)
                ticks = iter([0]*13 + [10]*50)
                train(root/'ext', time_cap_seconds=5, clock=lambda: next(ticks), **cfg)
                train(root/'ext', time_cap_seconds=5, extend_time_cap=100, resume=root/'ext/checkpoint_000004.pt', clock=lambda: 0, **cfg)
            def rows(folder):
                result = [json.loads(x) for x in (root/folder/'generations.jsonl').read_text().splitlines()]
                for row in result: row.pop('elapsed_seconds')
                return result
            self.assertEqual(rows('full'), rows('ext'))
            self.assertEqual(rows('ext')[3]['panel_boundary']['to'], 2)

    def test_pending_promotion_exports_champion(self):
        import hashlib, json, torch
        from unittest.mock import patch
        from phase2_torch import Phase2Network
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); opponent = root/'x.pt'; Phase2Network(seed=9).save(opponent)
            panel = root/'panel.json'
            panel.write_text(json.dumps({'version': 1, 'members': [{'name': 'x', 'path': 'x.pt', 'kind': 'phase2', 'encoder': 'v1', 'sha256': hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
            def check(model, champion, panel, *args):
                return {'gain': .1, 'lower_bound': .05, 'upper_bound': .15, 'head_to_head': .5, 'promoted': True, 'members': {'x': {'challenger': [.6, .6], 'champion': [.5, .5]}}, 'direct': [.5, .5], 'panel_version': 1}
            cfg = dict(generations=3, matches_per_generation=4, matches_per_round=4, target=1, panel_path=panel, panel_pairs=2, head_to_head_pairs=2)
            with patch('phase2_trainer._selection_check', side_effect=check):
                train(root/'full', champions_dir=root/'champ-full', time_cap_seconds=100, clock=lambda: 0, **cfg)
                ticks = iter([0]*5 + [10]*50)
                train(root/'ext', champions_dir=root/'champ-ext', time_cap_seconds=5, clock=lambda: next(ticks), **cfg)
                import os
                replace = os.replace
                def crash_checkpoint(source, destination):
                    if Path(destination) == root/'ext/checkpoint_000002.pt':
                        raise OSError('simulated crash before checkpoint commit')
                    return replace(source, destination)
                with patch('phase2_trainer.os.replace', side_effect=crash_checkpoint):
                    with self.assertRaisesRegex(OSError, 'simulated crash'):
                        train(root/'ext', champions_dir=root/'champ-ext', time_cap_seconds=5, extend_time_cap=100, resume=root/'ext/checkpoint_000002.pt', stop_after_round=2, clock=lambda: 0, **cfg)
                train(root/'ext', champions_dir=root/'champ-ext', time_cap_seconds=5, extend_time_cap=100, resume=root/'ext/checkpoint_000002.pt', stop_after_round=2, clock=lambda: 0, **cfg)
                boundary = torch.load(root/'ext/checkpoint_000002.pt', weights_only=False)
                self.assertFalse(boundary['phase_ended'])
                self.assertEqual(boundary['champion_generation'], 2)
                self.assertEqual(boundary['promotions'], 1)
                with patch('phase2_trainer._selection_check', side_effect=check) as resumed_check:
                    train(root/'ext', champions_dir=root/'champ-ext', time_cap_seconds=5, extend_time_cap=100, resume=root/'ext/checkpoint_000002.pt', clock=lambda: 0, **cfg)
                    self.assertEqual(resumed_check.call_count, 1)
            progress = json.loads((root/'ext/progress.json').read_text())
            self.assertEqual(progress['promotions'], 2)
            a = torch.load(root/'champ-full/reigning_champion.pt', weights_only=True)
            b = torch.load(root/'champ-ext/reigning_champion.pt', weights_only=True)
            self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))
            rows = [json.loads(x) for x in (root/'ext/generations.jsonl').read_text().splitlines()]
            self.assertTrue(rows[1]['promoted'])
            full_rows = [json.loads(x) for x in (root/'full/generations.jsonl').read_text().splitlines()]
            for row in rows + full_rows:
                row.pop('elapsed_seconds')
            self.assertEqual(rows, full_rows)
            full = torch.load(root/'full/checkpoint_000003.pt', weights_only=False)
            resumed = torch.load(root/'ext/checkpoint_000003.pt', weights_only=False)
            self.assertTrue(all(torch.equal(full['model'][key], resumed['model'][key]) for key in full['model']))
