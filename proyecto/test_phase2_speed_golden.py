"""Run 2 settings speed golden: bit-exact against the pre-optimization reference."""
from pathlib import Path
import subprocess
import sys
import unittest


class Phase2SpeedGoldenTests(unittest.TestCase):
    def test_run2_settings_match_pre_optimization_reference(self):
        root = Path(__file__).resolve().parents[1]
        reference = root / 'docs/perf/phase2-speed/golden.json'
        # Two actors keep the suite short; the recorded evidence also covers 1 and 12.
        completed = subprocess.run([sys.executable, str(root / 'proyecto/golden_phase2_speed.py'), 'compare',
                                    '--workers', '2', '--reference', str(reference)],
                                   cwd=root, capture_output=True, text=True, timeout=300)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('GOLDEN OK', completed.stdout)


if __name__ == '__main__':
    unittest.main()
