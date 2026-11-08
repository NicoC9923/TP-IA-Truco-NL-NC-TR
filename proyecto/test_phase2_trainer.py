import unittest
import torch
from phase2_trainer import run_rounds

class Phase2TrainerTest(unittest.TestCase):
    def test_rounds_are_deterministic_across_workers(self):
        a = run_rounds(seed=7, rounds=2, matches_per_round=2, workers=1, target=1)
        b = run_rounds(seed=7, rounds=2, matches_per_round=2, workers=2, target=1)
        self.assertTrue(all(torch.equal(a.model.state_dict()[k], v) for k, v in b.model.state_dict().items()))
        self.assertGreater(a.samples, 0)
    def test_updates_and_seed(self):
        from phase2_torch import Phase2Network
        a = run_rounds(seed=11, rounds=1, matches_per_round=1, target=1)
        b = run_rounds(seed=11, rounds=1, matches_per_round=1, target=1)
        c = run_rounds(seed=12, rounds=1, matches_per_round=1, target=1)
        initial = Phase2Network(seed=11)
        self.assertTrue(any(not torch.equal(v, initial.state_dict()[k]) for k, v in a.model.state_dict().items()))
        self.assertTrue(all(torch.equal(v, b.model.state_dict()[k]) for k, v in a.model.state_dict().items()))
        self.assertTrue(any(not torch.equal(v, c.model.state_dict()[k]) for k, v in a.model.state_dict().items()))
        signs = {seat: result for _, _, result, seat in a.last_samples}
        self.assertEqual(set(signs.values()), {-1.0, 1.0})
