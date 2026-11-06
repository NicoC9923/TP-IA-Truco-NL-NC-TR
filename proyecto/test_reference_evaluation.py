import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from phase1_network import GENOME_SIZE, NetworkPolicy

class EvaluationTests(unittest.TestCase):
    def test_saved_champions_are_evaluated_reproducibly(self):
        from reference_evaluation import evaluate
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            NetworkPolicy(np.zeros(GENOME_SIZE)).save(root / 'champion_0001.npy')
            evaluate(root, pairs=1, target=2, seed=4)
            first = (root / 'reference_evaluation.jsonl').read_text()
            rows = [json.loads(line) for line in first.splitlines()]
            self.assertEqual({r['reference_opponent'] for r in rows}, {'reactive', 'adaptive_utility'})
            self.assertTrue(all(r['phase'] == 1 and r['generation'] == 1 and 0 <= r['win_rate'] <= 1 and 'margin_of_error' in r for r in rows))
            (root / 'reference_evaluation.jsonl').unlink()
            evaluate(root, pairs=1, target=2, seed=4)
            self.assertEqual(first, (root / 'reference_evaluation.jsonl').read_text())
