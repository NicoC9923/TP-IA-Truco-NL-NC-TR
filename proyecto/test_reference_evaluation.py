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

    def test_evaluation_preserves_human_history_and_training_files(self):
        from unittest.mock import patch
        from engine_policies import AdaptiveUtilityPolicy
        from reference_evaluation import evaluate
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            history = root / 'human_history.json'
            history.write_text('{"parametros_ia": {"prob_farol_envido": 0.99}, "partidas": []}')
            NetworkPolicy(np.zeros(GENOME_SIZE)).save(root / 'champion_0001.npy')
            (root / 'generations.jsonl').write_text('{"generation": 1}\n')
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with patch('reference_evaluation.AdaptiveUtilityPolicy', side_effect=lambda seed: AdaptiveUtilityPolicy(seed, history_path=history)):
                evaluate(root, pairs=1, target=2, seed=4)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir() if p.name != 'reference_evaluation.jsonl'})

    def test_training_neither_imports_evaluation_nor_reads_its_log(self):
        import subprocess
        import sys
        script = r'''
import sys, tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
    log = Path(directory) / 'reference_evaluation.jsonl'
    log.write_text('invalid log that training must never read')
    def audit(event, args):
        if event == 'open' and str(args[0]).endswith('reference_evaluation.jsonl'):
            raise AssertionError('training accessed reference evaluation log')
    sys.addaudithook(audit)
    import phase1_trainer
    assert 'reference_evaluation' not in sys.modules
    phase1_trainer.train(directory, population_size=2, generations=1, pairs=1, workers=1, target=1, validation_pairs=1)
    assert 'reference_evaluation' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).parent, check=True)

    def test_watch_retries_a_champion_still_being_written(self):
        from reference_evaluation import evaluate
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            champion = root / 'champion_0001.npy'
            champion.write_bytes(b'\x93NUMPY')
            evaluate(root, pairs=1, target=1, only_new=True)
            self.assertFalse((root / 'reference_evaluation.jsonl').exists())
            NetworkPolicy(np.zeros(GENOME_SIZE)).save(champion)
            evaluate(root, pairs=1, target=1, only_new=True)
            self.assertEqual(len((root / 'reference_evaluation.jsonl').read_text().splitlines()), 2)
            evaluate(root, pairs=1, target=1, only_new=True)
            self.assertEqual(len((root / 'reference_evaluation.jsonl').read_text().splitlines()), 2)
