import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from phase1_network import GENOME_SIZE

class ScheduleReviewTests(unittest.TestCase):
    def test_evaluation_uses_saved_training_schedule(self):
        from reference_evaluation import evaluate
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            np.save(root / 'champion_0001.npy', np.zeros(GENOME_SIZE))
            (root / 'run_config.json').write_text(json.dumps({'seed': 7, 'target': None}))
            calls = []
            def play(*args, **kw):
                calls.append((kw['seed'], kw['target']))
                return SimpleNamespace(winner=0)
            with patch('reference_evaluation.play_match', side_effect=play):
                evaluate(root, pairs=2)
            self.assertEqual(calls[:4], [(14000000000,15)]*2 + [(14000000002,30)]*2)
