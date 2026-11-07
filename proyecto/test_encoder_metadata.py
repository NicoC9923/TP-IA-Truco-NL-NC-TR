import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from phase1_network import GENOME_SIZE, NetworkPolicy
from reference_evaluation import evaluate


class EncoderMetadataTests(unittest.TestCase):
    def test_explicit_encoder_must_match_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'champion.npy'
            NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v2').save(path)
            with self.assertRaisesRegex(ValueError, 'encoder'):
                NetworkPolicy.load(path, encoder='v1')
            self.assertEqual(NetworkPolicy.load(path, encoder='v2').encoder, 'v2')

    def test_reference_uses_config_only_without_sidecar_or_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'champion_0001.npy'
            np.save(path, np.zeros(GENOME_SIZE))
            (root / 'run_config.json').write_text(json.dumps({'encoder': 'v2'}))
            with patch('reference_evaluation._evaluate_pair', return_value=0.5) as match:
                evaluate(root, pairs=1, target=1)
                self.assertEqual({call.args[0][4] for call in match.call_args_list}, {'v2'})
