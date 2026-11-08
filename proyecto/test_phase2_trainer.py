"""Run trainer checks in a fresh process to isolate PyTorch global settings."""
import subprocess
import sys
import unittest
from pathlib import Path

class Phase2TrainerIsolationTest(unittest.TestCase):
    def test_trainer_in_fresh_process(self):
        script = Path(__file__).with_name('phase2_trainer_checks.py')
        completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

if __name__ == '__main__':
    unittest.main()
