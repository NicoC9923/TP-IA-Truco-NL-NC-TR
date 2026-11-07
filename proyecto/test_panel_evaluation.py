import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import panel_evaluation as panel
from phase1_network import GENOME_SIZE
from validation_schedule import validation_schedule

class PanelTests(unittest.TestCase):
    def test_decision_requires_all_conditions(self):
        good = {'win_rate': .60, 'margin_of_error': .05}
        self.assertEqual(panel.decision(good, [.60, .50], [.55, .50]), 'promising')
        self.assertEqual(panel.decision({'win_rate': .55, 'margin_of_error': .05}, [.60], [.50]), 'no demonstrated benefit')
        self.assertEqual(panel.decision(good, [.50], [.50]), 'no demonstrated benefit')
        self.assertEqual(panel.decision(good, [.70, .47], [.60, .50]), 'no demonstrated benefit')
        self.assertEqual(panel.decision(good, [], []), 'no demonstrated benefit')

    def test_frozen_manifest_hashes(self):
        rows = json.loads((panel.PANEL / 'panel.json').read_text())
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(hashlib.sha256((panel.PANEL / row['file']).read_bytes()).hexdigest(), row['sha256'])

    def test_reserved_schedules(self):
        self.assertEqual(len({panel.BASELINE_SEED, panel.PANEL_SEED, 0, 1, 2, 777, 4242}), 7)
        for seed in (panel.BASELINE_SEED, panel.PANEL_SEED):
            self.assertTrue(all(s % 2 == 0 for s, _ in validation_schedule(seed, None, 4)))

    def test_report_and_self_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, value in [('candidate', 1), ('baseline', 2), ('other', 3)]:
                np.save(root / (name + '.npy'), np.full(GENOME_SIZE, value))
            rows = [{'name': name, 'file': name + '.npy', 'sha256': panel.digest(root / (name + '.npy'))} for name in ('candidate', 'other')]
            (root / 'panel.json').write_text(json.dumps(rows))
            def fake(job):
                a, b, seed, target, score, style = job
                winner = 0 if a[0] == 1 else 1
                return [(SimpleNamespace(winner=winner), 0), (SimpleNamespace(winner=1-winner), 1)]
            with patch.object(panel, '_evaluate', side_effect=fake):
                report = panel.evaluate(root / 'candidate.npy', root / 'baseline.npy', panel_dir=root, baseline_pairs=2, panel_pairs=2, workers=1)
            self.assertEqual(report['candidate_vs_baseline']['win_rate'], 1)
            self.assertEqual(report['panel'][0]['skipped'], 'candidate is panel member')
            self.assertEqual(report['candidate_panel_mean'], 1)
            self.assertEqual(report['baseline_panel_mean'], 0)
            self.assertEqual(report['decision'], 'promising')
            self.assertEqual(report['schedule_seeds'], {'baseline': 9001, 'panel': 9002})
            json.dumps(report)

    def test_baseline_panel_member_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, value in [('candidate', 1), ('baseline', 2), ('other', 3)]:
                np.save(root / (name + '.npy'), np.full(GENOME_SIZE, value))
            rows = [{'name': name, 'file': name + '.npy', 'sha256': panel.digest(root / (name + '.npy'))} for name in ('baseline', 'other')]
            (root / 'panel.json').write_text(json.dumps(rows))
            def fake(job):
                a, b, seed, target, score, style = job
                winner = 0 if a[0] == 1 else 1
                return [(SimpleNamespace(winner=winner), 0), (SimpleNamespace(winner=1-winner), 1)]
            with patch.object(panel, '_evaluate', side_effect=fake):
                report = panel.evaluate(root / 'candidate.npy', root / 'baseline.npy', panel_dir=root, baseline_pairs=2, panel_pairs=2, workers=1)
            self.assertEqual(report['panel'][0]['skipped'], 'baseline is panel member')
            self.assertNotIn('candidate', report['panel'][0])
            self.assertEqual([r['name'] for r in report['panel'] if 'candidate' in r], ['other'])

    def test_cli_rejects_invalid_counts(self):
        result = subprocess.run([sys.executable, str(Path(panel.__file__)), '--candidate', 'a', '--baseline', 'b', '--output', 'c', '--workers', '0'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'workers positive', result.stderr)

if __name__ == '__main__':
    unittest.main()
