import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

import phase2_probe as p
from opponent_inference import _IDS


class ProbeTests(unittest.TestCase):
    def test_streams_split_and_balance(self):
        rows = [p.streams(i) for i in range(140)]
        self.assertEqual(len({x[1] for x in rows}), 140)
        self.assertEqual(len({seed for x in rows for seed in x[2]}), 280)
        for split in p.LIMITS:
            selected = [x for x in rows if x[0] == split]
            self.assertEqual({x[3] for x in selected}, {0, 1})
            self.assertEqual({x[4] for x in selected}, {15, 30})
            self.assertEqual({x[5] for x in selected}, set(p.MIXTURE))
        self.assertGreater(min(x[1] for x in rows), 140012000001998)
        self.assertGreater(min(s for x in rows for s in x[2]), 2**128+23999)

    def test_compatible_support_loss_and_gradients(self):
        logits = torch.zeros((1, 40), requires_grad=True)
        mask = torch.zeros((1, len(_IDS)), dtype=torch.bool); mask[0, :2] = True
        probability = torch.softmax(p.masked_scores(logits, mask), dim=1)
        self.assertEqual(probability[0, 2:].count_nonzero(), 0)
        self.assertAlmostEqual(probability.sum().item(), 1)
        self.assertAlmostEqual(p.combination_loss(logits, mask, torch.tensor([0])).item(), np.log(2))
        with self.assertRaises(ValueError):
            p.combination_loss(logits, mask, torch.tensor([2]))
        p.combination_loss(logits, mask, torch.tensor([0])).backward()
        self.assertIsNotNone(logits.grad)

    def test_control_same_architecture_and_frozen_trunks(self):
        from phase2_torch import Phase2Network
        a, b = Phase2Network(seed=1), Phase2Network(seed=39171)
        for model in (a,b):
            for param in model.parameters(): param.requires_grad_(False)
        head = p.make_probe(); result = head(a.layers[:4](torch.zeros((1, 1628))))
        result.sum().backward()
        self.assertEqual(result.shape, (1,40))
        self.assertTrue(all(x.grad is None for model in (a,b) for x in model.parameters()))
        self.assertEqual([x.shape for x in a.parameters()], [x.shape for x in b.parameters()])
        self.assertFalse(torch.equal(a.layers[0].weight,b.layers[0].weight))

    def test_loader_and_champion_unchanged(self):
        from phase2_network import ENCODER_VERSION
        with tempfile.TemporaryDirectory() as directory:
            champion = Path(directory) / 'champion.pt'; champion.write_bytes(b'original')
            before = hashlib.sha256(champion.read_bytes()).hexdigest()
            probe = p.make_probe()
            payload = {'schema': 1, 'champion_sha256': before, 'encoder_version': ENCODER_VERSION,
                       'architecture': 'linear128x40', 'weights': probe.state_dict()}
            path = Path(directory) / 'probe.pt'; torch.save(payload, path)
            self.assertTrue(torch.equal(p.load_probe(path, champion).weight, probe.weight))
            self.assertEqual(hashlib.sha256(champion.read_bytes()).hexdigest(), before)
            champion.write_bytes(b'changed')
            with self.assertRaises(ValueError): p.load_probe(path, champion)
            payload['champion_sha256'] = p.digest(champion); payload['architecture'] = 'wrong'; torch.save(payload, path)
            with self.assertRaises(ValueError): p.load_probe(path, champion)

    def test_private_labels_do_not_change_public_encoding_or_mask(self):
        from opponent_profile import OpponentProfile
        from opponent_inference import compact_evidence, compatible_mask
        from phase2_network import encode_observation
        obs = {'player': 0, 'hand': ((1, 'Espada'), (2, 'Oro'), (3, 'Copa')), 'available': (0, 1, 2), 'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1, 'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}
        other = dict(obs, opponent_hand=((4, 'Oro'), (5, 'Oro'), (6, 'Oro')))
        np.testing.assert_array_equal(encode_observation(obs, OpponentProfile(), ('card_0',)), encode_observation(other, OpponentProfile(), ('card_0',)))
        np.testing.assert_array_equal(compatible_mask([compact_evidence(obs)]), compatible_mask([compact_evidence(other)]))

    def test_cluster_interval(self):
        value = p.grouped_summary([(1,0), (1,0), (2,1), (2,1)])
        self.assertEqual(value['mean'], .5)
        self.assertEqual(value['matches'], 2)
        self.assertAlmostEqual(value['upper'], 1.48)
        self.assertIsNone(p.grouped_summary([(1,1)])['upper'])

    def test_timeout_produces_explicit_partial_report(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(p, 'OUTPUT', Path(directory)), patch.object(p, 'verify_models'), patch.object(subprocess, 'check_output', return_value=''), patch.object(sys, 'argv', ['phase2_probe.py']), patch.object(subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 595)):
                with self.assertRaises(SystemExit): p.main()
                report = json.loads((Path(directory) / 'card-probe-report.json').read_text())
                self.assertEqual(report['status'], 'hard_timeout')
                self.assertEqual(report['predictors'], {})
                self.assertTrue((Path(directory) / 'card-probe-started.lock').exists())
                with self.assertRaises(FileExistsError): p.main()


if __name__ == '__main__': unittest.main()
