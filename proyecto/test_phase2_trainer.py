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
    def test_samples_match_winner_and_learner_threads_are_fixed(self):
        from unittest.mock import patch
        import phase2_trainer
        real_play = phase2_trainer.play_match
        winners = []
        def capture(*args, **kwargs):
            result = real_play(*args, **kwargs)
            winners.append(result.winner)
            return result
        with patch.object(phase2_trainer, 'play_match', side_effect=capture):
            trained = run_rounds(seed=17, rounds=1, matches_per_round=1, target=1)
        self.assertEqual(len(winners), 1)
        self.assertEqual({sample[3] for sample in trained.last_samples}, {0, 1})
        for _, _, value, seat in trained.last_samples:
            self.assertEqual(value, 1.0 if seat == winners[0] else -1.0)
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_interop_threads(), 1)
