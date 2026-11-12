import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

import ranking_reliability
import offspring_diagnostic
import behavior_diversity


class DiagnosticEncoderTests(unittest.TestCase):
    def test_v2_checkpoint_is_rejected_by_v1_only_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.npz'
            np.savez(checkpoint, population=np.zeros((6, 1860)), hall=np.empty((0, 1860)),
                     reigning=np.zeros(1860), state=json.dumps({'encoder': 'v2'}))
            for module in (ranking_reliability, offspring_diagnostic, behavior_diversity):
                with self.subTest(tool=module.__name__):
                    with self.assertRaisesRegex(ValueError, 'v1-only'):
                        module.diagnose(checkpoint, workers=1)
