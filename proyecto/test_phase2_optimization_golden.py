"""Permanent pre-optimization model/log/checkpoint guard, including restart."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class Phase2OptimizationGoldenTests(unittest.TestCase):
    def test_both_heads_match_original_after_fresh_generation_resume(self):
        root = Path(__file__).resolve().parents[1]
        scratch = root / 'training_output/perf35'
        scratch.mkdir(parents=True, exist_ok=True)
        # Two actors keep the suite short; the recorded benchmark matrix uses 12.
        for head in ('off', 'on'):
            with self.subTest(head=head), tempfile.TemporaryDirectory(dir=scratch) as directory:
                reference = root / f'docs/perf/phase2-baseline/golden-{head}.json'
                output = Path(directory) / 'result.json'
                command = [sys.executable, str(root / 'proyecto/bench_phase2.py'),
                           'golden', '--workers', '2', '--resume', '--resume-round', '2',
                           '--reference', str(reference), '--output', str(output)]
                if head == 'on':
                    command.append('--auxiliary')
                # The harness enforces its own 175-second process-tree deadline.
                completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=180)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertEqual(json.loads(output.read_text())['golden'],
                                 json.loads(reference.read_text())['golden'])
