import tempfile
import unittest
from pathlib import Path
import torch
from phase2_trainer import train


class TrainerPriorTests(unittest.TestCase):
    def test_checkpoint_prior_and_legacy_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(generations=1, matches_per_generation=2, matches_per_round=1,
                          block_size=1, target=1, panel_path=None, seed=9, clock=lambda: 0)
            train(root, stop_after_round=1, **config)
            checkpoint = root / 'checkpoint_000001.pt'
            state = torch.load(checkpoint, weights_only=False)
            self.assertNotIn('profile_prior', state['config'])
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train(root, resume=checkpoint, profile_prior=5, **config)
            import subprocess
            import sys
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('phase2_trainer.py')),
                                     '--output', str(root), '--generations', '1', '--matches-per-generation', '2',
                                     '--matches-per-round', '1', '--block-size', '1', '--target', '1', '--seed', '9',
                                     '--no-selection', '--resume', str(checkpoint), '--profile-prior', '5'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Checkpoint configuration does not match', result.stderr)
            state['config']['profile_prior'] = 20
            torch.save(state, checkpoint)
            train(root, resume=checkpoint, stop_after_round=1, **config)
            del state['config']['profile_prior']
            torch.save(state, checkpoint)
            train(root, resume=checkpoint, **config)

    def test_nondefault_prior_is_stored_and_rejects_default_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(generations=1, matches_per_generation=2, matches_per_round=1,
                          block_size=1, target=1, panel_path=None, seed=9, clock=lambda: 0)
            train(root, profile_prior=5, stop_after_round=1, **config)
            checkpoint = root / 'checkpoint_000001.pt'
            state = torch.load(checkpoint, weights_only=False)
            self.assertEqual(state['config']['profile_prior'], 5)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train(root, resume=checkpoint, **config)

    def test_run_uses_prior_for_both_self_play_policies(self):
        from unittest.mock import patch
        import phase2_trainer
        seen = []
        play = phase2_trainer.play_match
        def inspect(a, b, **kwargs):
            seen.append((a.profile.prior_strength, b.profile.prior_strength))
            return play(a, b, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(phase2_trainer, 'play_match', inspect):
            train(directory, generations=1, matches_per_generation=1, matches_per_round=1,
                  block_size=1, target=1, panel_path=None, seed=9, profile_prior=5,
                  league_shares={'self': 1, 'snapshot': 0, 'phase1': 0, 'bluffer': 0, 'cautious': 0, 'aggressive': 0, 'calculating': 0})
        self.assertEqual(seen, [(5, 5)])


    def test_check_uses_frozen_panel_prior_and_run_prior(self):
        from unittest.mock import patch
        import phase2_trainer
        from test_phase2_progress import make_config
        seen = []
        play = phase2_trainer.play_mirror_pair
        def inspect(a, b, **kwargs):
            seen.append((a.profile.prior_strength, b.profile.prior_strength))
            return play(a, b, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(phase2_trainer, 'play_mirror_pair', inspect):
            root = Path(directory)
            train(root / 'run', profile_prior=5, **make_config(root))
        self.assertIn((5, 20), seen)
        self.assertIn((5, 5), seen)

    def test_export_records_prior(self):
        import json
        from test_phase2_progress import make_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(root / 'run', profile_prior=5, **make_config(root))
            metadata = json.loads((root / 'champ/reigning_champion.json').read_text())
            self.assertEqual(metadata['profile_prior'], 5)
            weights = torch.load(root / 'champ/reigning_champion.pt', weights_only=True)
            self.assertTrue(all(isinstance(value, torch.Tensor) for value in weights.values()))

    def test_example_hand_uses_sidecar_prior_and_legacy_default(self):
        import json
        import shutil
        from unittest.mock import patch
        from phase2_example_hand import example_hand_payload
        from phase2_torch import Phase2Network, Phase2Policy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            champ = root / 'champions/phase2'
            champ.mkdir(parents=True)
            rival = root / 'champions/phase1'
            rival.mkdir(parents=True)
            Phase2Network(seed=3).save(champ / 'reigning_champion.pt')
            shutil.copy(Path(__file__).resolve().parent.parent / 'champions/phase1/reigning_champion.npy', rival)
            sidecar = champ / 'reigning_champion.json'
            with patch('phase2_torch.Phase2Policy', wraps=Phase2Policy) as policy:
                sidecar.write_text(json.dumps({'profile_prior': 5}))
                self.assertTrue(example_hand_payload(root)['available'])
                self.assertEqual(policy.call_args.kwargs['profile_prior'], 5)
                sidecar.write_text('{}')
                self.assertTrue(example_hand_payload(root)['available'])
                self.assertEqual(policy.call_args.kwargs['profile_prior'], 20)
                self.assertEqual(policy.call_count, 2)

    def test_validation_predictions_use_run_prior(self):
        from unittest.mock import patch
        import phase2_card_metrics as metrics
        from phase2_torch import Phase2Network
        seen = []
        play = metrics.play_match
        def inspect(a, b, **kwargs):
            seen.append((a.profile.prior_strength, b.profile.prior_strength))
            return play(a, b, **kwargs)
        with patch.object(metrics, 'play_match', inspect):
            metrics.validation_predictions(Phase2Network(auxiliary=True), schedule_seed=10000,
                                           target=1, temperature=.1, situation_count=1, profile_prior=5)
        self.assertEqual(seen, [(5, 5)])

    def test_prior_five_resume_matches_uninterrupted_with_two_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(generations=2, matches_per_generation=2, matches_per_round=2,
                          workers=2, block_size=1, target=1, panel_path=None,
                          seed=17, profile_prior=5, clock=lambda: 0)
            full = train(root / 'full', **config)
            train(root / 'resume', stop_after_round=1, **config)
            resumed = train(root / 'resume', resume=root / 'resume/checkpoint_000001.pt', **config)
            for key, value in full.model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]))
            for key, state in full.optimizer.state_dict()['state'].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, resumed.optimizer.state_dict()['state'][key][name]))
            self.assertEqual((root / 'full/generations.jsonl').read_text(),
                             (root / 'resume/generations.jsonl').read_text())

    def test_panel_explicit_prior_is_used_and_configured_for_resume(self):
        import json
        from unittest.mock import patch
        import phase2_trainer
        from test_phase2_progress import make_config
        seen = []
        play = phase2_trainer.play_mirror_pair
        def inspect(a, b, **kwargs):
            seen.append((a.profile.prior_strength, b.profile.prior_strength))
            return play(a, b, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(phase2_trainer, 'play_mirror_pair', inspect):
            root = Path(directory)
            config = make_config(root)
            panel_path = config['panel_path']
            panel = json.loads(panel_path.read_text())
            panel['members'][0]['profile_prior'] = 7
            panel_path.write_text(json.dumps(panel))
            train(root / 'run', profile_prior=5, stop_after_round=4, **config)
            self.assertIn((5, 7), seen)
            state = torch.load(root / 'run/checkpoint_000004.pt', weights_only=False)
            self.assertEqual(state['panel_snapshots'][0]['profile_prior'], 5)
            panel['members'][0]['profile_prior'] = 8
            panel_path.write_text(json.dumps(panel))
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train(root / 'run', profile_prior=5, resume=root / 'run/checkpoint_000004.pt', **config)


    def test_trainer_passes_prior_to_diagnostics(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory, patch('phase2_card_metrics.validation_predictions', return_value={}) as predict:
            train(directory, generations=1, matches_per_generation=1, matches_per_round=1,
                  block_size=1, target=1, panel_path=None, auxiliary=True, profile_prior=5,
                  clock=lambda: 0)
            self.assertEqual(predict.call_args.kwargs['profile_prior'], 5)

    def test_snapshot_and_styled_opponents_use_run_prior(self):
        from unittest.mock import patch
        import phase2_trainer
        play = phase2_trainer.play_match
        for category in ('snapshot', 'bluffer'):
            seen = []
            def inspect(a, b, **kwargs):
                seen.append((a.profile.prior_strength, b.profile.prior_strength))
                return play(a, b, **kwargs)
            shares = dict.fromkeys(('self', 'snapshot', 'phase1', 'bluffer', 'cautious', 'aggressive', 'calculating'), 0)
            shares[category] = 1
            with self.subTest(category=category), tempfile.TemporaryDirectory() as directory, patch.object(phase2_trainer, 'play_match', inspect):
                train(directory, generations=2, matches_per_generation=1, matches_per_round=1,
                      block_size=1, target=1, panel_path=None, seed=9, profile_prior=5,
                      league_shares=shares, clock=lambda: 0)
                self.assertEqual(seen, [(5, 5), (5, 5)])

