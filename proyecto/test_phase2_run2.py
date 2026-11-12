import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from phase1_network import GENOME_SIZE
from phase2_trainer import train


class HeldOutTest(unittest.TestCase):
    def test_held_out_phase1_or_panel_is_refused_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / 'weights.npy'
            np.save(weights, np.zeros(GENOME_SIZE))
            sha = hashlib.sha256(weights.read_bytes()).hexdigest()
            held = root / 'held.json'
            held.write_text(json.dumps({'sha256': [sha]}))
            member = {'name': 'x', 'kind': 'phase1', 'encoder': 'v1',
                      'path': weights.name, 'sha256': sha}
            manifest = root / 'phase1.json'
            manifest.write_text(json.dumps({'members': [member]}))
            kwargs = dict(generations=1, matches_per_generation=1, matches_per_round=1,
                          block_size=1, target=1, champions_dir=None, progress=False,
                          held_out_manifest=held)
            with self.assertRaisesRegex(ValueError, 'held.out'):
                train(root / 'phase1-run', phase1_manifest=manifest, panel_path=None, **kwargs)
            panel = root / 'panel.json'
            panel.write_text(json.dumps({'version': 3, 'members': [member]}))
            with self.assertRaisesRegex(ValueError, 'held.out'):
                train(root / 'panel-run', panel_path=panel, **kwargs)

    def test_held_out_snapshot_pool_and_spoofed_hash_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forbidden = root / 'forbidden.pt'
            forbidden.write_bytes(b'held-out weights')
            sha = hashlib.sha256(forbidden.read_bytes()).hexdigest()
            held = root / 'held.json'
            held.write_text(json.dumps({'sha256': [sha]}))
            cfg = dict(generations=1, matches_per_generation=2, matches_per_round=1,
                       block_size=1, target=1, panel_path=None, champions_dir=None,
                       held_out_manifest=held, progress=False)
            run = root / 'run'
            train(run, stop_after_round=1, **cfg)
            checkpoint = run / 'checkpoint_000001.pt'
            initial = torch.load(checkpoint, weights_only=False)
            for digest in (sha, '0' * 64):
                state = dict(initial)
                state['league_pool'] = [{'path': '../forbidden.pt', 'sha256': digest, 'generation': 1}]
                torch.save(state, checkpoint)
                with self.assertRaisesRegex(ValueError, 'held.out'):
                    train(run, resume=checkpoint, **cfg)

    def test_cli_files_timing_and_six_member_manifest(self):
        import subprocess
        import sys
        from phase2_promotion import load_panel
        panel_path = Path('experiments/phase2-run2/panel.json')
        panel = load_panel(panel_path)
        self.assertEqual([m['name'] for m in panel['members']],
                         ['v2s1-g46', 'v2s2-g25', 'run1-g12', 'run1-g28', 'reactive', 'adaptive_utility'])
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, 'proyecto/phase2_trainer.py', '--output', tmp,
                                     '--generations', '2', '--matches-per-generation', '1', '--matches-per-round', '1',
                                     '--block-size', '1', '--target', '1', '--panel-path', str(panel_path),
                                     '--champions-dir', str(Path(tmp) / 'champ'), '--panel-pairs', '2',
                                     '--head-to-head-pairs', '2', '--no-panel-rotation', '--no-progress',
                                     '--held-out-manifest', 'experiments/phase2-run2/held-out.json',
                                     '--league-shares', 'experiments/phase2-run2/league-shares.json', '--timing-log'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = torch.load(Path(tmp) / 'checkpoint_000002.pt', weights_only=False)
            self.assertEqual(len(state['config']['held_out_manifest']), 2)
            self.assertEqual(state['config']['league_shares']['reactive'], .1)
            self.assertFalse(state['config']['panel_rotation'])
            rows = [json.loads(row) for row in (Path(tmp) / 'generations.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['check_seconds'], 0)
            self.assertGreater(rows[1]['check_seconds'], 0)
            self.assertTrue(all(row['training_seconds'] > 0 for row in rows))

    def test_new_snapshot_matching_held_out_is_never_added(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forbidden_bytes = b'forbidden snapshot'
            held = root / 'held.json'
            held.write_text(json.dumps({'sha256': [hashlib.sha256(forbidden_bytes).hexdigest()]}))
            def fake_save(model, path):
                Path(path).write_bytes(forbidden_bytes)
            with patch('phase2_torch.Phase2Network.save', fake_save):
                with self.assertRaisesRegex(ValueError, 'held.out'):
                    train(root / 'run', generations=1, matches_per_generation=1,
                          matches_per_round=1, block_size=1, target=1, panel_path=None,
                          champions_dir=None, progress=False, held_out_manifest=held)

    def test_reference_kind_cannot_disguise_held_out_phase1_file(self):
        held = Path('experiments/phase2-run2/held-out.json')
        source = Path('experiments/phase1-seed1/reigning_champion.npy').resolve()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root / 'phase1.json'
            manifest.write_text(json.dumps({'members': [{'name': 'disguised', 'path': str(source),
                                'kind': 'reference', 'encoder': 'v1', 'sha256': sha}]}))
            with self.assertRaises(ValueError):
                train(root / 'run', generations=1, matches_per_generation=1, matches_per_round=1,
                      block_size=1, target=1, panel_path=None, champions_dir=None, progress=False,
                      phase1_manifest=manifest, held_out_manifest=held)
