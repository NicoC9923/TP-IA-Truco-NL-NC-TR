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
                            '--no-selection', '--workers', '1', '--seed', '7', '--target', '1',
                            '--matches-per-generation', '3', '--matches-per-round', '2'], check=True)
            entries = [json.loads(line) for line in (Path(directory) / 'generations.jsonl').read_text().splitlines()]
        self.assertEqual(len(entries), 2)
        for generation, entry in enumerate(entries, 1):
            self.assertEqual((entry['phase'], entry['generation'], entry['matches'], entry['check']), (2, generation, 4, None))
            self.assertGreater(entry['learner_samples'], 0)
            self.assertGreater(entry['updates'], 0)
            self.assertGreaterEqual(entry['mean_loss'], 0)
            self.assertGreaterEqual(entry['elapsed_seconds'], 0)
    def test_resume_round_boundary_and_config_mismatch(self):
        import json
        import tempfile
        from pathlib import Path
        from phase2_trainer import train as _train
        from functools import partial
        train = partial(_train, panel_path=None)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            config = dict(generations=2, matches_per_generation=2, matches_per_round=1,
                          workers=2, seed=9, target=1, minibatch=8)
            full = train(root / 'full', **config)
            train(root / 'resumed', stop_after_round=1, **config)
            checkpoint = root / 'resumed' / 'checkpoint_000001.pt'
            (root / 'resumed' / 'generations.jsonl').write_text('{"generation": 99}\n')
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train(root / 'resumed', resume=checkpoint, **{**config, 'seed': 10})
            resumed = train(root / 'resumed', resume=checkpoint, **config)
            for key, tensor in full.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.model.state_dict()[key]))
            for key, state in full.optimizer.state_dict()['state'].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, resumed.optimizer.state_dict()['state'][key][name]))
            logs = []
            for directory in ('full', 'resumed'):
                lines = [json.loads(line) for line in (root / directory / 'generations.jsonl').read_text().splitlines()]
                for line in lines:
                    line.pop('elapsed_seconds')
                logs.append(lines)
            self.assertEqual(logs[0], logs[1])

    def test_match_settings_are_deterministic_and_mixed(self):
        from phase2_trainer import match_settings
        settings = [match_settings(7, i, 'mixed', .3) for i in range(100)]
        self.assertEqual(settings, [match_settings(7, i, 'mixed', .3) for i in range(100)])
        self.assertEqual({target for target, score in settings}, {15, 30})
        self.assertTrue(any(score != (0, 0) for target, score in settings))
        self.assertTrue(all(0 <= x < target for target, score in settings for x in score))
    def test_training_deal_seeds_are_unique_for_logical_matches(self):
        from phase2_trainer import training_seed
        ids = list(range(1000)) + [299999, 300000, 2**32, 2**48]
        seeds = {training_seed(seed, i) for seed in range(4) for i in ids}
        self.assertEqual(len(seeds), 4 * len(ids))
    def test_resume_generation_boundary_with_new_workers_and_extended_run(self):
        import json
        import random
        import shutil
        import tempfile
        import numpy as np
        from pathlib import Path
        from phase2_trainer import train as _train
        from functools import partial
        train = partial(_train, panel_path=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(matches_per_generation=2, matches_per_round=1, workers=2, seed=29, target=1)
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            full = train(root / 'full', generations=2, **config)
            full_checkpoint = torch.load(root / 'full' / 'checkpoint_000002.pt', weights_only=False)
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            train(root / 'resume', generations=1, **config)
            checkpoint = root / 'resume' / 'checkpoint_000002.pt'
            stopped = torch.load(checkpoint, weights_only=False)
            self.assertEqual(stopped['python_rng'], full_checkpoint['python_rng'])
            self.assertEqual(stopped['numpy_rng'][0], full_checkpoint['numpy_rng'][0])
            np.testing.assert_array_equal(stopped['numpy_rng'][1], full_checkpoint['numpy_rng'][1])
            self.assertEqual(stopped['numpy_rng'][2:], full_checkpoint['numpy_rng'][2:])
            self.assertTrue(torch.equal(stopped['torch_rng'], full_checkpoint['torch_rng']))
            shutil.copy(root / 'full' / 'generations.jsonl', root / 'resume' / 'generations.jsonl')
            resumed = train(root / 'resume', generations=2, resume=checkpoint, **{**config, 'workers': 1})
            for key, value in full.model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]))
            for key, state in full.optimizer.state_dict()['state'].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, resumed.optimizer.state_dict()['state'][key][name]))
            def semantic(path):
                records = [json.loads(line) for line in path.read_text().splitlines()]
                for record in records: record.pop('elapsed_seconds')
                return records
            self.assertEqual(semantic(root / 'full' / 'generations.jsonl'), semantic(root / 'resume' / 'generations.jsonl'))
            with self.assertRaisesRegex(ValueError, 'output directory'):
                train(root / 'other', generations=2, resume=checkpoint, **config)

    def test_checkpoint_contract_and_retention(self):
        import tempfile
        from pathlib import Path
        from phase2_trainer import train as _train
        from functools import partial
        train = partial(_train, panel_path=None)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'generation_rounds.*log_path'):
                run_rounds(rounds=1, target=1, checkpoint_dir=directory)
            with self.assertRaisesRegex(ValueError, 'generation_rounds.*log_path'):
                run_rounds(rounds=1, target=1, generation_rounds=1)
            train(directory, generations=2, matches_per_generation=2, matches_per_round=1,
                  workers=1, target=1, stop_after_round=3)
            self.assertEqual({p.name for p in Path(directory).glob('checkpoint_*.pt')},
                             {'checkpoint_000002.pt', 'checkpoint_000003.pt'})



class LeaguePoolCheckpointTest(unittest.TestCase):
    def test_pool_survives_resume(self):
        import tempfile
        from pathlib import Path
        from phase2_trainer import train
        with tempfile.TemporaryDirectory() as directory:
            options = dict(generations=3, matches_per_generation=1, matches_per_round=1,
                           workers=1, target=1, panel_path=None, seed=8)
            train(directory, stop_after_round=2, **options)
            checkpoint = Path(directory) / 'checkpoint_000002.pt'
            before = torch.load(checkpoint, weights_only=False)['league_pool']
            self.assertEqual([s['generation'] for s in before], [1, 2])
            train(directory, resume=checkpoint, **options)
            after = torch.load(Path(directory) / 'checkpoint_000003.pt', weights_only=False)['league_pool']
            self.assertEqual(before, after[:2])
            self.assertEqual([s['generation'] for s in after], [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
