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
    def test_recording_policy_matches_plain_policy_with_style(self):
        from unittest.mock import patch
        import numpy as np
        from phase2_network import INPUT_SIZE
        from phase2_torch import Phase2Network, Phase2Policy
        from phase2_trainer import RecordingPolicy
        model = Phase2Network(seed=3)
        plain = Phase2Policy(model, seed=5, style='cautious')
        recording = RecordingPolicy(model, seed=5, style='cautious')
        inputs = np.zeros(INPUT_SIZE, dtype=np.float32)
        with patch.object(plain, 'encode', return_value=inputs), patch.object(recording, 'encode', return_value=inputs):
            self.assertEqual([plain({}, ['quiero', 'no_quiero']) for _ in range(10)],
                             [recording({}, ['quiero', 'no_quiero']) for _ in range(10)])

    def test_cli_logs_generations_and_training_seeds_are_disjoint(self):
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from phase2_trainer import training_seed
        from validation_schedule import validation_schedule
        self.assertFalse({training_seed(7, i) for i in range(100)} &
                         {seed for seed, _ in validation_schedule(7, 15, 100)})
        with tempfile.TemporaryDirectory() as directory:
            script = Path(__file__).with_name('phase2_trainer.py')
            subprocess.run([sys.executable, str(script), '--output', directory, '--generations', '2',
                            '--workers', '1', '--seed', '7', '--target', '1',
                            '--matches-per-generation', '3', '--matches-per-round', '2'], check=True)
            entries = [json.loads(line) for line in (Path(directory) / 'generations.jsonl').read_text().splitlines()]
        self.assertEqual(len(entries), 2)
        for generation, entry in enumerate(entries, 1):
            self.assertEqual((entry['phase'], entry['generation'], entry['matches'], entry['check']), (2, generation, 4, None))
            self.assertGreater(entry['learner_samples'], 0)
            self.assertGreater(entry['updates'], 0)
            self.assertGreaterEqual(entry['mean_loss'], 0)
            self.assertGreaterEqual(entry['elapsed_seconds'], 0)
