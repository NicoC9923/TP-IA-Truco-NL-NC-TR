import unittest
import torch
from phase2_trainer import run_rounds as _run_rounds
from functools import partial
run_rounds = partial(_run_rounds, block_size=1)

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
        self.assertTrue(all(sample[2] in (-1.0, 1.0) for sample in a.last_samples))
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
            trained = run_rounds(seed=17, rounds=1, matches_per_round=1, target=1, league_shares={'self': 1, 'phase1': 0, 'snapshot': 0, 'bluffer': 0, 'cautious': 0, 'aggressive': 0, 'calculating': 0})
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
                            '--matches-per-generation', '3', '--matches-per-round', '2', '--block-size', '1'], check=True)
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
        train = partial(_train, panel_path=None, block_size=1)
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
        train = partial(_train, panel_path=None, block_size=1)
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
        train = partial(_train, panel_path=None, block_size=1)
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
                           workers=1, target=1, panel_path=None, seed=8, block_size=1)
            train(directory, stop_after_round=2, **options)
            checkpoint = Path(directory) / 'checkpoint_000002.pt'
            before = torch.load(checkpoint, weights_only=False)['league_pool']
            self.assertEqual([s['generation'] for s in before], [1, 2])
            train(directory, resume=checkpoint, **options)
            after = torch.load(Path(directory) / 'checkpoint_000003.pt', weights_only=False)['league_pool']
            self.assertEqual(before, after[:2])
            self.assertEqual([s['generation'] for s in after], [1, 2, 3])



class LeagueActorTest(unittest.TestCase):
    def test_nonself_records_only_learner_seat(self):
        from phase2_trainer import _league_block
        from phase2_torch import Phase2Network
        spec = {'category': 'styled', 'identity': 'current', 'style': 'bluffer', 'offset': .1, 'neutral_profile': False}
        from unittest.mock import patch
        import phase2_trainer
        real_play = phase2_trainer.play_match
        winners = []
        def capture(*args, **kwargs):
            result = real_play(*args, **kwargs)
            winners.append(result.winner)
            return result
        with patch.object(phase2_trainer, 'play_match', side_effect=capture):
            results = _league_block((Phase2Network(seed=5).state_dict(), 5, list(range(10)), 1, 0, spec, {}, .1))
        self.assertEqual([{s[3] for s in samples} for _, samples in results], [{i % 2} for i in range(10)])
        for (match_id, samples), winner in zip(results, winners):
            self.assertTrue(all(s[2] == (1.0 if match_id % 2 == winner else -1.0) for s in samples))

class LeagueRunTest(unittest.TestCase):
    def test_league_mix_and_worker_determinism(self):
        import json
        import tempfile
        from pathlib import Path
        from phase2_trainer import train
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            opts = dict(generations=1, matches_per_generation=8, matches_per_round=8,
                        seed=22, target=1, panel_path=None, champions_dir=None)
            a = train(root / 'a', workers=1, **opts)
            b = train(root / 'b', workers=2, **opts)
            self.assertTrue(all(torch.equal(v, b.model.state_dict()[k]) for k, v in a.model.state_dict().items()))
            log = json.loads((root / 'a' / 'generations.jsonl').read_text())
            self.assertEqual(sum(log['league_mix'].values()), 8)
            self.assertAlmostEqual(log['style_offset_round_end'], .1 * 8 / 20000)



class LeagueStyleTest(unittest.TestCase):
    def test_styles_change_observed_behavior(self):
        import tempfile
        from pathlib import Path
        from phase2_torch import Phase2Network, Phase2Policy
        from symmetric_engine import play_match, bluff
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / 'generation_000001.pt'
            Phase2Network(seed=4).save(snapshot)
            model = Phase2Network.load(snapshot)
        def evaluate(style):
            counts = [0, 0, 0, 0]
            for seed in range(300):
                result = play_match(Phase2Policy(model, seed=seed, style=style, style_strength=.1),
                                    Phase2Policy(model, seed=seed + 1000), seed=seed, target=1)
                for d in result.decisions:
                    if d['player'] != 0:
                        continue
                    if any(bluff(a, d['observation']['hand']) for a in d['legal_actions']):
                        counts[1] += 1
                        counts[0] += d['bluff'] is True
                    if 'quiero' in d['legal_actions']:
                        counts[3] += 1
                        counts[2] += d['action'] == 'quiero'
            return counts
        bluffer, calculating = evaluate('bluffer'), evaluate('calculating')
        cautious, aggressive = evaluate('cautious'), evaluate('aggressive')
        self.assertGreater(bluffer[0] / bluffer[1], calculating[0] / calculating[1])
        self.assertLess(cautious[2] / cautious[3], aggressive[2] / aggressive[3])



class LeaguePanelTest(unittest.TestCase):
    def test_real_run_excludes_panel_members(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import phase2_trainer as trainer
        from phase2_league import panel_exclusions
        real_draw = trainer.draw_opponent
        seen = []
        def capture(*args, **kwargs):
            spec = real_draw(*args, **kwargs)
            self.assertNotIn(spec['identity'], kwargs['exclusions'])
            seen.append(spec['identity'])
            return spec
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory, patch.object(trainer, 'draw_opponent', side_effect=capture):
            trainer.train(directory, generations=1, matches_per_generation=8, matches_per_round=4,
                          target=1, workers=1, champions_dir=None,
                          panel_path=Path(__file__).resolve().parent.parent / 'experiments/phase2-selection-panel/panel-v0-measuring.json',
                          league_shares={'self': 0, 'phase1': 1, 'snapshot': 0,
                                         'bluffer': 0, 'cautious': 0, 'aggressive': 0, 'calculating': 0})
        self.assertEqual(len(seen), 2)
        self.assertNotIn('current', seen)




class BlockProfileTest(unittest.TestCase):
    def test_profiles_are_retained_within_block(self):
        from unittest.mock import patch
        import phase2_trainer as trainer
        from phase2_torch import Phase2Network
        real_play = trainer.play_match
        profiles = []
        def capture(a, b, **kwargs):
            learner, opponent = (a, b) if len(profiles) % 2 == 0 else (b, a)
            profiles.append((learner.profile, opponent.profile, sum(learner.profile.confidence().values())))
            return real_play(a, b, **kwargs)
        spec = {'category': 'styled', 'identity': 'current', 'style': 'bluffer', 'offset': .1, 'neutral_profile': True}
        with patch.object(trainer, 'play_match', side_effect=capture):
            trainer._league_block((Phase2Network(seed=9).state_dict(), 9, [0, 1], 5, 0, spec, {}, .1))
        self.assertIs(profiles[0][0], profiles[1][0])
        self.assertIs(profiles[0][1], profiles[1][1])
        self.assertGreater(profiles[1][2], profiles[0][2])




class LeagueIntegrityTest(unittest.TestCase):
    def test_frozen_opponent_hash_is_verified(self):
        import tempfile
        from pathlib import Path
        from phase2_trainer import _league_block
        from phase2_torch import Phase2Network
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.pt'
            path.write_bytes(b'changed')
            spec = {'category': 'snapshot', 'identity': 'expected', 'style': None, 'offset': 0, 'neutral_profile': False}
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                _league_block((Phase2Network().state_dict(), 1, [0], 1, 0, spec,
                               {'expected': {'resolved_path': str(path)}}, .1))



class OpponentCacheTest(unittest.TestCase):
    def test_snapshot_loaded_once_for_repeated_blocks(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import phase2_trainer as trainer
        from phase2_torch import Phase2Network
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.pt'
            Phase2Network(seed=71).save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            spec = {'category': 'snapshot', 'identity': digest, 'style': None, 'offset': 0, 'neutral_profile': False}
            args = (Phase2Network(seed=2).state_dict(), 2, [0], 1, 0, spec, {digest: {'resolved_path': str(path)}}, .1)
            with patch.object(Phase2Network, 'load', wraps=Phase2Network.load) as load:
                trainer._league_block(args)
                trainer._league_block(args)
                self.assertEqual(load.call_count, 1)



class LeagueConfigurationTest(unittest.TestCase):
    def test_invalid_shares_rejected_before_workers_start(self):
        from unittest.mock import patch
        import phase2_trainer as trainer
        with patch.object(trainer.mp, 'get_context', side_effect=AssertionError('workers started')):
            with self.assertRaisesRegex(ValueError, 'League shares'):
                trainer.run_rounds(workers=2, league_shares={'self': 1})



class ManifestPortabilityTest(unittest.TestCase):
    def test_resume_accepts_manifest_at_new_path(self):
        import json
        import tempfile
        from pathlib import Path
        from phase2_trainer import train
        original = Path(__file__).resolve().parent.parent / 'experiments/phase2-selection-panel/league-phase1.json'
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            data = json.loads(original.read_text())
            for member in data['members']:
                member['path'] = str((original.parent / member['path']).resolve())
            relocated = root / 'manifest.json'
            relocated.write_text(json.dumps(data))
            panel1 = root / 'panel1.json'
            panel2 = root / 'panel2.json'
            panel_text = json.dumps({'version': 0, 'members': [data['members'][0]], 'measuring_only': True})
            panel1.write_text(panel_text)
            panel2.write_text(panel_text)
            opts = dict(generations=2, matches_per_generation=4, matches_per_round=4,
                        target=1, workers=1, champions_dir=None, panel_pairs=2, head_to_head_pairs=2)
            train(root / 'run', panel_path=panel1, phase1_manifest=original, stop_after_round=1, **opts)
            train(root / 'run', panel_path=panel2, phase1_manifest=relocated, resume=root / 'run/checkpoint_000001.pt', **opts)



class LeagueLoggingTest(unittest.TestCase):
    def test_phase1_and_neutral_counts(self):
        import json
        import tempfile
        from pathlib import Path
        from phase2_trainer import train
        shares = {'self': 0, 'phase1': 1, 'snapshot': 0, 'bluffer': 0, 'cautious': 0, 'aggressive': 0, 'calculating': 0}
        with tempfile.TemporaryDirectory() as directory:
            train(directory, generations=1, matches_per_generation=8, matches_per_round=4,
                  target=1, panel_path=None, champions_dir=None, league_shares=shares, neutral_rate=1)
            log = json.loads((Path(directory) / 'generations.jsonl').read_text())
        self.assertEqual(sum(log['phase1_opponents'].values()), 8)
        self.assertEqual(log['neutral_blocks'], 2)
        self.assertIn('style_offset_round_end', log)



class LeagueBlockResumeTest(unittest.TestCase):
    def test_resume_with_pool_and_two_workers(self):
        import json
        import tempfile
        from pathlib import Path
        from phase2_trainer import train
        shares = {'self': 0, 'phase1': 0, 'snapshot': 1, 'bluffer': 0, 'cautious': 0, 'aggressive': 0, 'calculating': 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opts = dict(generations=3, matches_per_generation=8, matches_per_round=8, block_size=4,
                        workers=2, seed=31, target=1, panel_path=None, champions_dir=None, league_shares=shares)
            full = train(root / 'full', **opts)
            train(root / 'resume', stop_after_round=2, **opts)
            checkpoint = root / 'resume/checkpoint_000002.pt'
            state = torch.load(checkpoint, weights_only=False)
            self.assertTrue(state['league_pool'])
            resumed = train(root / 'resume', resume=checkpoint, **opts)
            self.assertTrue(all(torch.equal(v, resumed.model.state_dict()[k]) for k, v in full.model.state_dict().items()))
            for key, values in full.optimizer.state_dict()['state'].items():
                for name, value in values.items():
                    self.assertTrue(torch.equal(value, resumed.optimizer.state_dict()['state'][key][name]))
            def logs(folder):
                result = [json.loads(line) for line in (root / folder / 'generations.jsonl').read_text().splitlines()]
                for entry in result:
                    entry.pop('elapsed_seconds')
                return result
            self.assertEqual(logs('full'), logs('resume'))
            self.assertEqual(logs('full')[1]['league_mix'], {'snapshot': 8})



class OverlappingPanelTest(unittest.TestCase):
    def test_panel_member_in_manifest_is_never_drawn(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import phase2_trainer as trainer
        source = Path(__file__).resolve().parent.parent / 'experiments/phase2-selection-panel/league-phase1.json'
        member = json.loads(source.read_text())['members'][0]
        member['path'] = str((source.parent / member['path']).resolve())
        seen = []
        real_draw = trainer.draw_opponent
        def capture(*args, **kwargs):
            spec = real_draw(*args, **kwargs)
            seen.append(spec['identity'])
            return spec
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            manifest = root / 'league.json'
            panel = root / 'panel.json'
            manifest.write_text(json.dumps({'members': [member]}))
            panel.write_text(json.dumps({'version': 0, 'members': [member], 'measuring_only': True}))
            with patch.object(trainer, 'draw_opponent', side_effect=capture):
                trainer.train(root / 'run', generations=1, matches_per_generation=8, matches_per_round=4,
                              target=1, workers=1, champions_dir=None, panel_path=panel, phase1_manifest=manifest,
                              league_shares={'self': 0, 'phase1': 1, 'snapshot': 0,
                                             'bluffer': 0, 'cautious': 0, 'aggressive': 0, 'calculating': 0})
        self.assertEqual(seen, ['current', 'current'])

if __name__ == '__main__':
    unittest.main()
