import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from phase1_network import GENOME_SIZE
from phase2_trainer import train


class TrainerRobustnessTests(unittest.TestCase):
    def test_empty_learner_round_is_checkpointed_without_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            genome = np.zeros(GENOME_SIZE)
            genome[-1] = 10000
            weights = root / 'fold.npy'
            np.save(weights, genome)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'members': [{'name': 'fold', 'path': weights.name,
                'encoder': 'v1', 'sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}]}))
            shares = dict.fromkeys(('self', 'phase1', 'snapshot', 'bluffer', 'cautious', 'aggressive', 'calculating'), 0)
            shares['phase1'] = 1
            config = dict(generations=2, matches_per_generation=1, matches_per_round=1,
                          block_size=1, target=1, random_start_share=0, panel_path=None,
                          phase1_manifest=manifest, league_shares=shares, progress=False, seed=9)
            output = root / 'run'
            train(output, stop_after_round=1, **config)
            before = torch.load(output / 'checkpoint_000001.pt', weights_only=False)
            train(output, resume=output / 'checkpoint_000001.pt', **config)
            after = torch.load(output / 'checkpoint_000002.pt', weights_only=False)
            self.assertEqual(after['matches'], 2)
            self.assertEqual(after['samples'], before['samples'])
            self.assertEqual(after['updates'], before['updates'])
            self.assertTrue(all(torch.equal(value, after['model'][name]) for name, value in before['model'].items()))
            entry = json.loads((output / 'generations.jsonl').read_text().splitlines()[-1])
            self.assertEqual((entry['learner_samples'], entry['updates'], entry['mean_loss']), (0, 0, None))

    @patch("phase2_trainer.run_rounds")
    def test_invalid_float_settings_fail_before_output_creation(self, mocked_run):
        names = ('time_cap_seconds', 'min_gain', 'min_direct', 'saturation_threshold',
                 'style_target', 'neutral_rate', 'lr', 'random_start_share',
                 'temperature', 'profile_prior', 'auxiliary_weight')
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                for value in (float('nan'), float('inf'), float('-inf')):
                    with self.subTest(name=name, value=value):
                        output = Path(directory) / (name + str(value))
                        with self.assertRaises(ValueError):
                            train(output, panel_path=None, **{name: value})
                        self.assertFalse(output.exists())
            for name, value in (('min_gain', -0.1), ('min_gain', 1.1), ('min_direct', -0.1),
                                ('min_direct', 1.1), ('saturation_threshold', -0.1),
                                ('saturation_threshold', 1.1), ('style_target', -0.1),
                                ('neutral_rate', 1.1), ('random_start_share', -0.1),
                                ('lr', 0), ('auxiliary_weight', -0.1)):
                with self.subTest(name=name, value=value):
                    output = Path(directory) / (name + str(value))
                    with self.assertRaises(ValueError):
                        train(output, panel_path=None, **{name: value})
                    self.assertFalse(output.exists())

    def test_checkpoint_rejects_missing_or_changed_contract_versions(self):
        from phase2_network import ENCODER_VERSION
        from symmetric_engine import ENGINE_VERSION
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(generations=2, matches_per_generation=1, matches_per_round=1,
                          block_size=1, target=1, panel_path=None, progress=False, seed=9)
            train(root, stop_after_round=1, **config)
            checkpoint = root / 'checkpoint_000001.pt'
            state = torch.load(checkpoint, weights_only=False)
            self.assertEqual(state['config']['engine_version'], ENGINE_VERSION)
            self.assertEqual(state['config']['encoder_version'], ENCODER_VERSION)
            for key in ('engine_version', 'encoder_version'):
                for missing in (True, False):
                    with self.subTest(key=key, missing=missing):
                        config_state = state['config'].copy()
                        if missing:
                            del config_state[key]
                        else:
                            config_state[key] = -1
                        altered = dict(state, config=config_state)
                        torch.save(altered, checkpoint)
                        with self.assertRaisesRegex(ValueError, 'configuration'):
                            train(root, resume=checkpoint, **config)
            torch.save(state, checkpoint)
            train(root, resume=checkpoint, **config)

    def test_resume_repairs_stale_or_missing_champion_export(self):
        from test_phase2_progress import make_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            output = root / 'run'
            train(output, stop_after_round=2, **config)
            checkpoint = output / 'checkpoint_000002.pt'
            state = torch.load(checkpoint, weights_only=False)
            champion = root / 'champ/reigning_champion.pt'
            metadata = root / 'champ/reigning_champion.json'
            for missing in (False, True):
                with self.subTest(missing=missing):
                    if missing:
                        champion.unlink()
                        metadata.unlink()
                    else:
                        changed = {k: v + 1 for k, v in state['champion_weights'].items()}
                        torch.save(changed, champion)
                        metadata.write_text(json.dumps({'generation': 2, 'temperature': .1, 'profile_prior': 20}))
                    train(output, resume=checkpoint, stop_after_round=2, **config)
                    exported = torch.load(champion, weights_only=True)
                    self.assertTrue(all(torch.equal(value, exported[name]) for name, value in state['champion_weights'].items()))
                    self.assertEqual(json.loads(metadata.read_text())['generation'], state['champion_generation'])

    def test_champion_export_replaces_files_atomically(self):
        import os
        from test_phase2_progress import make_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            destinations = []
            real_replace = os.replace
            def observe(source, destination):
                if Path(destination).name in ('reigning_champion.pt', 'reigning_champion.json'):
                    self.assertNotEqual(Path(source), Path(destination))
                    self.assertEqual(Path(source).parent, Path(destination).parent)
                    self.assertTrue(Path(source).exists())
                    destinations.append(Path(destination).suffix)
                return real_replace(source, destination)
            with patch('phase2_trainer.os.replace', side_effect=observe):
                train(root / 'run', stop_after_round=2, **config)
            self.assertCountEqual(destinations, ['.pt', '.json'])

    def test_reconciliation_survives_uncommitted_export_and_rejected_promotion(self):
        from test_phase2_progress import make_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            output = root / 'run'
            train(output, stop_after_round=2, **config)
            checkpoint = output / 'checkpoint_000002.pt'
            state = torch.load(checkpoint, weights_only=False)
            champion = root / 'champ/reigning_champion.pt'
            metadata = root / 'champ/reigning_champion.json'
            torch.save({name: value + 1 for name, value in state['champion_weights'].items()}, champion)
            metadata.write_text(json.dumps({'generation': 2, 'temperature': .1, 'profile_prior': 20}))
            decision = {'gain': 0, 'lower_bound': -1, 'upper_bound': 1, 'head_to_head': .5, 'promoted': False}
            with patch('phase2_trainer.promotion_decision', return_value=decision):
                train(output, resume=checkpoint, **config)
            final = torch.load(output / 'checkpoint_000004.pt', weights_only=False)
            exported = torch.load(champion, weights_only=True)
            self.assertEqual(final['champion_generation'], 1)
            self.assertEqual(json.loads(metadata.read_text())['generation'], 1)
            self.assertTrue(all(torch.equal(value, exported[name]) for name, value in final['champion_weights'].items()))

    def test_failed_champion_replace_preserves_existing_file(self):
        import os
        from test_phase2_progress import make_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            champion = root / 'champ/reigning_champion.pt'
            champion.parent.mkdir()
            champion.write_bytes(b'previous')
            real_replace = os.replace
            def fail_weights(source, destination):
                if Path(destination) == champion:
                    raise OSError('injected replace failure')
                return real_replace(source, destination)
            with patch('phase2_trainer.os.replace', side_effect=fail_weights):
                with self.assertRaisesRegex(OSError, 'injected'):
                    train(root / 'run', stop_after_round=2, **config)
            self.assertEqual(champion.read_bytes(), b'previous')
            self.assertFalse(list(champion.parent.glob('.champion-*')))

    @patch('phase2_trainer.run_rounds')
    def test_invalid_counts_are_rejected_before_creating_output(self, mocked_run):
        names = ('generations', 'matches_per_generation', 'matches_per_round', 'workers',
                 'seed', 'minibatch', 'panel_pairs', 'head_to_head_pairs', 'check_seed_base',
                 'patience', 'saturation_checks', 'saturation_pairs', 'pool_recent',
                 'pool_diverse', 'block_size', 'style_ramp_matches', 'card_prediction_situations')
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                with self.subTest(name=name):
                    output = Path(directory) / name
                    with self.assertRaises(ValueError):
                        train(output, panel_path=None, **{name: float('nan')})
                    self.assertFalse(output.exists())
