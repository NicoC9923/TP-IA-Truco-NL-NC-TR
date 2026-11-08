import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from phase2_torch import Phase2Network
from phase2_trainer import train

class SelectionIntegrationTest(unittest.TestCase):
    def test_check_snapshot_export_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opponent = root / 'opponent.pt'
            Phase2Network(seed=9).save(opponent)
            panel = root / 'panel.json'
            panel.write_text(json.dumps({'version': 1, 'members': [{'name': 'standin', 'path': 'opponent.pt', 'kind': 'phase2', 'encoder': 'v1', 'sha256': hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
            with patch('phase2_trainer.promotion_decision', return_value={'gain': .1, 'lower_bound': .05, 'upper_bound': .15, 'head_to_head': .5, 'promoted': True}):
                train(root / 'run', generations=2, workers=2, matches_per_generation=1, matches_per_round=1, target=1, panel_path=panel, champions_dir=root/'champions', panel_pairs=2, head_to_head_pairs=2)
            rows = [json.loads(line) for line in (root/'run/generations.jsonl').read_text().splitlines()]
            self.assertTrue(rows[1]['promoted'])
            self.assertEqual(rows[1]['promotion_check']['panel_version'], 1)
            self.assertIn('challenger', rows[1]['promotion_check']['members']['standin'])
            self.assertTrue((root/'run/generation_000002.pt').exists())
            self.assertTrue((root/'champions/reigning_champion.pt').exists())
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in torch.load(root/'champions/reigning_champion.pt', weights_only=True).values()))

    def test_excluded_modules_not_imported(self):
        import sys
        self.assertFalse({'reactive', 'adaptive_utility'} & set(sys.modules))

