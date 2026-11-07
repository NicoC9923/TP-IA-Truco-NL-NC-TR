import unittest
import numpy as np
from behavior_diversity import compare

class BehaviorTests(unittest.TestCase):
    def test_known_distributions(self):
        distance, agreement = compare(np.array([[.5, .5], [1., 0.]]), np.array([[1., 0.], [0., 1.]]))
        self.assertEqual(distance, .75)
        self.assertEqual(agreement, .5)

class CliTests(unittest.TestCase):
    def test_rejects_invalid_counts(self):
        import subprocess, sys
        from pathlib import Path
        script = Path(__file__).with_name('behavior_diversity.py')
        run = subprocess.run([sys.executable, str(script), 'missing', '--output', 'unused', '--observations', '0'], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('counts must be positive', run.stderr)
