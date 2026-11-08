import unittest
import numpy as np
from parent_selection_diagnostic import summarize

class ParentSelectionTest(unittest.TestCase):
    def test_perfect_selection_and_random_baseline(self):
        report = summarize(list(range(10)), list(range(10)), tournaments=3000, bootstrap=200, seed=42)
        self.assertGreater(report['selection']['advantage'], 2.0)
        self.assertGreater(report['selection']['ci95'][0], 0)
        self.assertAlmostEqual(report['random']['advantage'], 0, delta=0.25)
        self.assertAlmostEqual(report['random']['best_fraction'], 1/3, delta=0.04)

if __name__ == '__main__': unittest.main()



class DiagnosticInterfaceTest(unittest.TestCase):
    def test_checkpoint_and_cli(self):
        import json, tempfile, subprocess, sys
        from pathlib import Path
        from phase1_network import GENOME_SIZE
        from parent_selection_diagnostic import diagnose
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.npz'
            output = Path(tmp) / 'report.json'
            config = [2, 1, 15, 1, 1, .55, 5, 1, 1, 1, .3, 'classic', .1, .5, True]
            def checkpoint(encoder):
                np.savez(path, population=np.zeros((2, GENOME_SIZE)), hall=np.zeros((0, GENOME_SIZE)),
                         reigning=np.zeros(GENOME_SIZE), state=json.dumps({'encoder': encoder, 'config': config, 'generation': 10}))
            checkpoint('v2')
            with self.assertRaisesRegex(ValueError, 'v1-only'):
                diagnose(path, 2, 1, 1, 2)
            checkpoint('v1')
            proc = subprocess.run([sys.executable, 'proyecto/parent_selection_diagnostic.py', str(path), '--output', str(output),
                                   '--workers', '1', '--measurement-pairs', '1', '--tournaments', '3', '--bootstrap', '4'], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['first_pass_pairs'], 1)
            self.assertEqual(report['evaluated_generation'], 11)
            self.assertNotEqual(report['schedules'][0][0]['seed'], report['schedules'][1][0]['seed'])
            self.assertEqual(report['tournaments'], 3)

    def test_invalid_cli(self):
        import subprocess, sys
        proc = subprocess.run([sys.executable, 'proyecto/parent_selection_diagnostic.py', 'missing', '--output', 'missing', '--workers', '0'], capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
