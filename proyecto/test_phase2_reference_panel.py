import json
import tempfile
import unittest
from pathlib import Path

from phase2_promotion import load_panel


class ReferencePanelTest(unittest.TestCase):
    def test_reference_manifest_requires_known_agent_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'panel.json'
            members = [{'name': name, 'kind': 'reference', 'agent': name}
                       for name in ('reactive', 'adaptive_utility')]
            path.write_text(json.dumps({'version': 3, 'members': members}))
            self.assertEqual(load_panel(path)['members'], members)
            for bad in ({'name': 'x', 'kind': 'reference', 'agent': 'unknown'},
                        {'name': 'x', 'kind': 'reference', 'agent': 'reactive', 'path': 'x'},
                        {'name': 'x', 'kind': 'reference', 'agent': 'reactive', 'sha256': 'x'}):
                path.write_text(json.dumps({'version': 3, 'members': [bad]}))
                with self.assertRaises(ValueError):
                    load_panel(path)

    def test_six_member_check_is_deterministic_across_workers(self):
        import hashlib
        import multiprocessing as mp
        import numpy as np
        from phase1_network import GENOME_SIZE
        from phase2_torch import Phase2Network
        from phase2_trainer import _selection_check
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            members = []
            for i in range(4):
                phase2 = i >= 2
                file = root / (f'{i}.pt' if phase2 else f'{i}.npy')
                if phase2:
                    Phase2Network(seed=i).save(file)
                else:
                    np.save(file, np.zeros(GENOME_SIZE))
                member = {'name': str(i), 'kind': 'phase2' if phase2 else 'phase1',
                          'path': file.name, 'encoder': 'v1',
                          'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                          'resolved_path': str(file)}
                if phase2:
                    member.update(temperature=.05, profile_prior=5)
                members.append(member)
            members.extend({'name': name, 'kind': 'reference', 'agent': name}
                           for name in ('reactive', 'adaptive_utility'))
            panel = {'version': 3, 'members': members}
            challenger, champion = Phase2Network(seed=11), Phase2Network(seed=12)
            args = (challenger, champion.state_dict(), panel, 10001, 2, 2, 1, .1)
            single = _selection_check(*args, None, .01, .475)
            with mp.get_context('spawn').Pool(2) as pool:
                parallel = _selection_check(*args, pool, .01, .475)
            self.assertEqual(single, parallel)
            self.assertEqual(len(single['members']), 6)
            gain = sum(sum(row['challenger']) - sum(row['champion'])
                       for row in single['members'].values()) / 12
            self.assertAlmostEqual(single['gain'], gain)

    def test_rotation_off_persists_and_resume_validates_panel_and_switch(self):
        import torch
        from unittest.mock import patch
        from phase2_trainer import train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'panel.json'
            members = [{'name': name, 'kind': 'reference', 'agent': name}
                       for name in ('reactive', 'adaptive_utility')]
            manifest.write_text(json.dumps({'version': 3, 'members': members}))
            kwargs = dict(generations=3, matches_per_generation=1, matches_per_round=1,
                          block_size=1, target=1, panel_path=manifest, panel_pairs=2,
                          head_to_head_pairs=2, champions_dir=None, panel_rotation=False,
                          saturation_checks=1, saturation_pairs=2, saturation_threshold=0,
                          progress=False, clock=lambda: 0)
            with patch('phase2_trainer.saturated', side_effect=AssertionError('rotation checked')):
                train(root / 'run', stop_after_round=2, **kwargs)
                checkpoint = root / 'run/checkpoint_000002.pt'
                state = torch.load(checkpoint, weights_only=False)
                self.assertIs(state['config']['panel_rotation'], False)
                self.assertEqual(state['saturation_history'], [])
                self.assertEqual(state['panel']['members'][0]['identity'], 'reference:reactive:v1')
                train(root / 'run', resume=checkpoint, **kwargs)
            checkpoint = root / 'run/checkpoint_000003.pt'
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train(root / 'run', resume=checkpoint, **{**kwargs, 'panel_rotation': True})
            members[0]['agent'] = 'adaptive_utility'
            manifest.write_text(json.dumps({'version': 3, 'members': members}))
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train(root / 'run', resume=checkpoint, **kwargs)

    def test_phase2_member_uses_its_own_temperature_and_prior(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from phase2_torch import Phase2Network
        from phase2_trainer import _check_pair
        spec = {'kind': 'phase2', 'weights': Phase2Network(seed=4).state_dict(),
                'temperature': .05, 'profile_prior': 5}
        seen = []
        def play(a, b, **kwargs):
            seen.append((b.temperature, b.profile.prior_strength))
            return SimpleNamespace(winner=0), SimpleNamespace(winner=1)
        with patch('phase2_trainer.play_mirror_pair', side_effect=play):
            _check_pair((spec, spec, 3, 1, 7, .1))
        self.assertEqual(seen, [(.05, 5)])

    def test_no_rotation_also_guards_pending_extension_boundary(self):
        import torch
        from unittest.mock import patch
        from phase2_trainer import train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / 'panel.json'
            panel.write_text(json.dumps({'version': 3, 'members': [
                {'name': 'reactive', 'kind': 'reference', 'agent': 'reactive'}]}))
            cfg = dict(generations=4, matches_per_generation=4, matches_per_round=4,
                       target=1, panel_path=panel, champions_dir=None, panel_pairs=2,
                       head_to_head_pairs=2, saturation_checks=1, saturation_pairs=2,
                       panel_rotation=False, time_cap_seconds=5, progress=False)
            ticks = iter([0, 0, 0, 0, 0, 10] + [10] * 50)
            with patch('phase2_trainer.saturated', side_effect=AssertionError('rotation checked')):
                train(root / 'run', clock=lambda: next(ticks), **cfg)
                checkpoint = root / 'run/checkpoint_000002.pt'
                self.assertEqual(torch.load(checkpoint, weights_only=False)['stop_reason'], 'time_cap')
                self.assertIsNone(json.loads((root / 'run/generations.jsonl').read_text().splitlines()[-1])['check'])
                train(root / 'run', resume=checkpoint, extend_time_cap=100,
                      stop_after_round=3, clock=lambda: 0, **cfg)
            self.assertIsNotNone(json.loads((root / 'run/generations.jsonl').read_text().splitlines()[1])['check'])
            state = torch.load(root / 'run/checkpoint_000003.pt', weights_only=False)
            self.assertEqual(state['saturation_history'], [])
            self.assertEqual(state['panel']['version'], 3)

    def test_old_checkpoint_without_rotation_key_resumes_with_rotation_on(self):
        import torch
        from phase2_trainer import train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(generations=1, matches_per_generation=2, matches_per_round=1,
                       block_size=1, target=1, panel_path=None, progress=False, clock=lambda: 0)
            train(root, stop_after_round=1, **cfg)
            checkpoint = root / 'checkpoint_000001.pt'
            state = torch.load(checkpoint, weights_only=False)
            self.assertNotIn('panel_rotation', state['config'])
            train(root, resume=checkpoint, **cfg)

    def test_cli_persists_no_panel_rotation(self):
        import subprocess
        import sys
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('phase2_trainer.py')),
                                     '--output', tmp, '--generations', '1', '--matches-per-generation', '1',
                                     '--matches-per-round', '1', '--block-size', '1', '--target', '1',
                                     '--no-selection', '--no-progress', '--no-panel-rotation'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = torch.load(Path(tmp) / 'checkpoint_000001.pt', weights_only=False)
            self.assertIs(state['config']['panel_rotation'], False)

    def test_resume_rejects_panel_network_settings_and_reference_identity_changes(self):
        import hashlib
        import torch
        from phase2_torch import Phase2Network
        from phase2_trainer import train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / 'weights.pt'
            Phase2Network(seed=9).save(weights)
            panel = root / 'panel.json'
            members = [{'name': 'network', 'kind': 'phase2', 'path': weights.name, 'encoder': 'v1',
                        'sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
                        'temperature': .05, 'profile_prior': 5},
                       {'name': 'reactive', 'kind': 'reference', 'agent': 'reactive'}]
            panel.write_text(json.dumps({'version': 3, 'members': members}))
            cfg = dict(generations=1, matches_per_generation=2, matches_per_round=1,
                       block_size=1, target=1, panel_path=panel, champions_dir=None,
                       panel_pairs=2, head_to_head_pairs=2, progress=False, clock=lambda: 0)
            train(root / 'run', stop_after_round=1, **cfg)
            checkpoint = root / 'run/checkpoint_000001.pt'
            state = torch.load(checkpoint, weights_only=False)
            self.assertEqual(state['config']['panel_path']['members'][0]['temperature'], .05)
            self.assertEqual(state['config']['panel_path']['members'][0]['profile_prior'], 5)
            for field, value in [('temperature', .1), ('profile_prior', 10)]:
                old = members[0][field]
                members[0][field] = value
                panel.write_text(json.dumps({'version': 3, 'members': members}))
                with self.assertRaisesRegex(ValueError, 'configuration'):
                    train(root / 'run', resume=checkpoint, **cfg)
                members[0][field] = old
            panel.write_text(json.dumps({'version': 3, 'members': members}))
            state['config']['panel_path']['members'][1]['identity'] = 'reference:reactive:v0'
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train(root / 'run', resume=checkpoint, **cfg)

    def test_reference_is_fresh_and_seeded_independently_for_each_mirrored_seat(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from phase2_torch import Phase2Network
        from phase2_trainer import _check_pair
        network = {'kind': 'phase2', 'weights': Phase2Network(seed=4).state_dict()}
        reference = {'kind': 'reference', 'agent': 'adaptive_utility'}
        observations = []
        def inspect(a, b, **kwargs):
            ref = b if hasattr(b, 'agent') else a
            seat = 1 if ref is b else 0
            observations.append((ref, seat, ref.agent.rng.getstate()))
            for _ in range(20 if seat else 1):
                ref.agent.rng.random()
            return SimpleNamespace(winner=0)
        with patch('phase2_trainer.play_match', side_effect=inspect), patch('symmetric_engine.play_match', side_effect=inspect):
            _check_pair((network, reference, 10000, 15, 77, .1))
            _check_pair((network, reference, 10000, 15, 77, .1))
        self.assertEqual([x[1] for x in observations], [1, 0, 1, 0])
        self.assertIsNot(observations[0][0], observations[1][0])
        self.assertEqual(observations[0][2], observations[2][2])
        self.assertEqual(observations[1][2], observations[3][2])
        self.assertNotEqual(observations[0][2], observations[1][2])
