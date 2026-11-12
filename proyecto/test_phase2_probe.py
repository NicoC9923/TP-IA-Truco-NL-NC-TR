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

    def test_production_initialization_freezes_and_matches_heads(self):
        from phase2_torch import Phase2Network
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'champion.pt'
            Phase2Network(seed=1).save(path)
            (a,b), (first, second) = p.initialize_models(path)
            self.assertTrue(all(not x.requires_grad for model in (a,b) for x in model.parameters()))
            self.assertEqual([x.shape for x in a.parameters()], [x.shape for x in b.parameters()])
            self.assertFalse(torch.equal(a.layers[0].weight,b.layers[0].weight))
            for x, y in zip(first.parameters(), second.parameters()):
                self.assertTrue(torch.equal(x, y))
            first(a.layers[:4](torch.zeros((1, 1628)))).sum().backward()
            self.assertTrue(all(x.grad is None for model in (a,b) for x in model.parameters()))

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

    def test_public_capture_and_label_attachment_are_separate(self):
        from phase2_torch import Phase2Network
        obs = self.observation()
        features = np.zeros(1628, dtype=np.float32)
        captured = p.capture_public(obs, features, .4)
        original_evidence = captured['evidence']
        obs['table'] = ((1, (7, 'Oro')),)
        features[0] = 99
        a, b = Phase2Network(seed=1), Phase2Network(seed=2)
        one = p.attach_label(captured, ((4,'Oro'), (5,'Oro'), (6,'Oro')), 1, 'test', 0, a, b)
        two = p.attach_label(captured, ((4,'Copa'), (5,'Copa'), (6,'Copa')), 1, 'test', 0, a, b)
        self.assertEqual(one[1]['table'], ())
        self.assertEqual(one[5], original_evidence)
        self.assertEqual(one[5], two[5])
        np.testing.assert_array_equal(one[7], two[7])
        np.testing.assert_array_equal(one[8], two[8])
        self.assertNotEqual(one[6], two[6])
        self.assertEqual(one[9:], ('test', 0))

    @staticmethod
    def observation():
        return {'player': 0, 'hand': ((1, 'Espada'), (2, 'Oro'), (3, 'Copa')), 'available': (0, 1, 2), 'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1, 'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}

    def test_common_evidence_recomputes_metrics_and_stable_log_loss(self):
        from phase2_torch import Phase2Network
        from opponent_inference import _CARD_IDS
        capture = p.capture_public(self.observation(), np.zeros(1628, dtype=np.float32), .4)
        row = p.attach_label(capture, ((4,'Oro'), (5,'Oro'), (6,'Oro')), 6, 'test', 3, Phase2Network(), Phase2Network(seed=2))
        first, second = p.make_probe(), p.make_probe()
        with torch.no_grad():
            first.weight.zero_(); first.bias.zero_()
            first.bias[_CARD_IDS[(4,'Oro')]] = -1000
            second.load_state_dict(first.state_dict())
        record, marginals = p.score_situation(row, (first, second))
        self.assertGreater(record['metrics']['champion']['log_loss'], 900)
        self.assertEqual(record['metrics']['champion'], record['metrics']['control'])
        self.assertEqual(record['split'], 'test')
        self.assertEqual(record['label']['hand_index'], row[6])
        self.assertAlmostEqual(sum(marginals['exact']), 3)
        rebuilt = p.summarize_evidence(json.loads(json.dumps([record])))
        self.assertEqual(rebuilt['champion']['log_loss']['mean'], record['metrics']['champion']['log_loss'])
        self.assertEqual(rebuilt['exact']['hidden_card_accuracy']['situations'], 1)
        json.dumps(record, allow_nan=False)

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

    def test_child_failure_preserves_champion_and_explicit_report(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(p, 'OUTPUT', Path(directory)), patch.object(p, 'verify_models'), patch.object(subprocess, 'check_output', return_value=''), patch.object(sys, 'argv', ['phase2_probe.py']), patch.object(subprocess, 'run', return_value=SimpleNamespace(returncode=9)):
                with self.assertRaises(SystemExit): p.main()
                report = json.loads((Path(directory) / 'card-probe-report.json').read_text())
                self.assertIn('failed', report['status'])
                self.assertTrue(report['champion_unchanged'])
                self.assertEqual(report['champion_sha256_before'], report['champion_sha256_after'])


if __name__ == '__main__': unittest.main()
