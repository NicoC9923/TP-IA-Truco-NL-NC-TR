import unittest
import numpy as np
from offspring_diagnostic import variants

class OffspringTests(unittest.TestCase):
    def test_operators_match_trainer(self):
        a, b = np.zeros(1860), np.ones(1860)
        actual = variants(a, b, np.random.default_rng(42))
        rng = np.random.default_rng(42)
        cross = np.where(rng.random(1860) < .5, a, b)
        expected = cross + rng.normal(0, .05, 1860) * (rng.random(1860) < .1)
        np.testing.assert_array_equal(actual['crossover_mutation'], expected)
        np.testing.assert_array_equal(actual['crossover_only'], cross)

class SummaryTests(unittest.TestCase):
    def test_summary_and_namespace(self):
        from offspring_diagnostic import summary, SEED_BASE
        from behavior_diversity import BANK_SEED
        result = summary([-.2, .2])
        self.assertEqual(result['mean'], 0)
        self.assertEqual(result['fraction_beating_parent'], .5)
        self.assertLess(result['ci95'][0], 0)
        self.assertGreater(result['ci95'][1], 0)
        self.assertGreater(SEED_BASE, 10**17 + 10**6)
        self.assertGreater(BANK_SEED, SEED_BASE + 10**6)

class CliTests(unittest.TestCase):
    def test_rejects_invalid_workers(self):
        import subprocess, sys
        from pathlib import Path
        script = Path(__file__).with_name('offspring_diagnostic.py')
        run = subprocess.run([sys.executable, str(script), 'missing', '--output', 'unused', '--workers', '0'], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('workers positive', run.stderr)
