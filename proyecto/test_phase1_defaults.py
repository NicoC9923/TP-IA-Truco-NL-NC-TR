import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from phase1_trainer import train


class Phase1DefaultsTests(unittest.TestCase):
    def test_default_patience_is_recorded_in_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            train(directory, population_size=2, generations=10, pairs=1, workers=1,
                  target=1, validation_pairs=1, promotion_threshold=0)
            with np.load(Path(directory) / 'checkpoint_0010.npz') as checkpoint:
                state = json.loads(str(checkpoint['state']))
            self.assertEqual(state['config'][6], 50)
