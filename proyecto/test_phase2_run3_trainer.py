"""Run 3 trainer options (ADR 0062): encoder version, warm start, blended target, greedy checks."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import phase2_equity as equity
import phase2_network as net
import phase2_trainer as trainer
from phase2_torch import Phase2Network, Phase2Policy, widen_state_dict

SELF = {'category': 'self', 'identity': 'current', 'style': None, 'offset': 0, 'neutral_profile': False}


def noisy_network(seed=0, input_size=net.INPUT_SIZE):
    model = Phase2Network(seed=seed, input_size=input_size)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model


def small_table():
    rows = [equity.Observation(15, a, b, (a + b) % 2, 0 if a >= b else 1) for a in range(15) for b in range(15)]
    rows += [equity.Observation(30, a, b, (a + b) % 2, 0 if a >= b else 1) for a in range(30) for b in range(30)]
    return equity.fit_equity_table(rows)


def block(weights, rollout=None, temperature=.05, ids=range(8)):
    job = (weights, 3, list(ids), 'mixed', .3, SELF, {}, temperature, 5)
    return trainer._league_block(job if rollout is None else (*job, rollout))


class RolloutTest(unittest.TestCase):
    weights = noisy_network().state_dict()

    def test_blend_changes_only_the_targets(self):
        old = block(self.weights)
        new = block(self.weights, {'blend': .5, 'table': small_table()})
        self.assertEqual([m for m, _ in old], [m for m, _ in new])
        changed = 0
        for (_, a), (_, b) in zip(old, new):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assertEqual(x[0].tobytes(), y[0].tobytes())
                self.assertEqual((x[1], x[3]), (y[1], y[3]))
                self.assertLessEqual(abs(y[2] - .5 * x[2]), .5 + 1e-12)
                changed += y[2] != x[2]
        self.assertGreater(changed, 0)

    def test_without_a_rollout_the_results_keep_their_old_shape(self):
        for row in block(self.weights):
            self.assertEqual(len(row), 2)

    def test_blend_zero_reproduces_the_match_result_target(self):
        old = block(self.weights)
        zero = block(self.weights, {'blend': 0.0, 'table': small_table()})
        self.assertEqual([[s[2] for s in samples] for _, samples in old], [[s[2] for s in samples] for _, samples in zero])

    def test_greedy_disagreement_is_counted_per_match(self):
        rollout = {'blend': 0.0, 'table': small_table(), 'track_greedy': True}
        rows = block(self.weights, rollout, temperature=100.0)
        differs = sum(r[2][0] for r in rows)
        total = sum(r[2][1] for r in rows)
        self.assertEqual(total, sum(len(r[1]) for r in rows))
        self.assertGreater(differs / total, .2)
        cold = block(self.weights, rollout, temperature=1e-4)
        self.assertLess(sum(r[2][0] for r in cold) / sum(r[2][1] for r in cold), .02)

    def test_chunks_and_rounds_pass_the_rollout_through(self):
        rollout = {'blend': .5, 'table': small_table(), 'track_greedy': True}
        blocks = [([0, 1, 2, 3], SELF)]
        results = trainer._play_round(None, weights=self.weights, seed=3, blocks=blocks, target='mixed',
                                      random_start_share=.3, opponents={}, temperature=.05, profile_prior=5,
                                      rollout=rollout)
        self.assertEqual([r[0] for r in results], [0, 1, 2, 3])
        self.assertTrue(all(len(r) == 3 for r in results))


class CheckSelectionTest(unittest.TestCase):
    def test_greedy_flag_reaches_the_policies(self):
        weights = noisy_network().state_dict()
        spec = {'kind': 'phase2', 'weights': weights, 'greedy': True}
        seen = []

        def play(a, b, **kwargs):
            seen.append((a, b))
            from types import SimpleNamespace
            return SimpleNamespace(winner=0), SimpleNamespace(winner=1)
        with patch('phase2_trainer.play_mirror_pair', side_effect=play):
            trainer._check_pair((spec, dict(spec), 3, 15, 8, .05))
            trainer._check_pair(({'kind': 'phase2', 'weights': weights}, {'kind': 'phase2', 'weights': weights}, 3, 15, 8, .05))
        self.assertTrue(all(p.greedy for p in seen[0]))
        self.assertFalse(any(p.greedy for p in seen[1]))

    def test_selection_check_marks_every_network_greedy_and_nothing_else_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            opponent = Path(folder) / 'o.pt'
            noisy_network(2).save(opponent)
            panel = {'version': 1, 'members': [
                {'name': 'net', 'kind': 'phase2', 'encoder': 'v1', 'resolved_path': str(opponent)},
                {'name': 'au', 'kind': 'reference', 'agent': 'adaptive_utility'}]}
            model = noisy_network(1)
            champion = noisy_network(3).state_dict()
            captured = {}

            def run(jobs, keys, pool):
                captured['jobs'] = jobs
                return {key: [.5, .5] for key in keys}
            for greedy in (False, True):
                with patch('phase2_trainer._run_check_jobs', side_effect=run):
                    trainer._selection_check(model, champion, panel, 10, 2, 2, 15, .05, None, .01, .475, 5, greedy=greedy)
                specs = [spec for job in captured['jobs'] for spec in job[:2] if spec['kind'] == 'phase2']
                self.assertTrue(specs)
                self.assertEqual([spec.get('greedy', False) for spec in specs], [greedy] * len(specs))


class InitialModelTest(unittest.TestCase):
    def test_warm_start_widens_with_zero_columns(self):
        old = noisy_network(4).state_dict()
        model = trainer._initial_model(0, False, 3, old)
        self.assertTrue(torch.equal(model.layers[0].weight[:, :net.INPUT_SIZE], old['layers.0.weight']))
        self.assertEqual(float(model.layers[0].weight[:, net.INPUT_SIZE:].abs().max()), 0.0)
        self.assertEqual(model.encoder_version, 3)

    def test_without_warm_start_the_model_is_the_seeded_fresh_network(self):
        model = trainer._initial_model(5, False, 2, None)
        fresh = Phase2Network(seed=5, input_size=net.INPUT_SIZES[2])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), fresh.state_dict().values())))

    def test_warm_start_wider_than_the_encoder_is_refused(self):
        wide = widen_state_dict(noisy_network(4).state_dict(), 3)
        with self.assertRaises(ValueError):
            trainer._initial_model(0, False, 2, wide)


def standin_panel(root):
    opponent = root / 'opponent.pt'
    Phase2Network(seed=9).save(opponent)
    panel = root / 'panel.json'
    panel.write_text(json.dumps({'version': 1, 'members': [{
        'name': 'standin', 'path': 'opponent.pt', 'kind': 'phase2', 'encoder': 'v1',
        'sha256': hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
    return panel


class RunThreeTrainingTest(unittest.TestCase):
    def options(self, root, warm, table):
        return dict(generations=2, matches_per_generation=4, matches_per_round=2, block_size=2, target='mixed',
                    panel_path=standin_panel(root), champions_dir=root / 'champions', panel_pairs=2,
                    head_to_head_pairs=2, clock=lambda: 0, encoder_version=3, warm_start=warm,
                    target_blend=.5, equity_table=table, check_selection='greedy', progress=False)

    def test_end_to_end_records_the_options_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            warm = root / 'warm.pt'
            noisy_network(6).save(warm)
            table = root / 'equity.json'
            small_table().save(table)
            options = self.options(root, warm, table)
            with patch('phase2_trainer.promotion_decision', return_value={'gain': .1, 'lower_bound': .05, 'upper_bound': .15, 'head_to_head': .5, 'promoted': True}):
                trainer.train(root / 'full', **options)
                trainer.train(root / 'part', stop_after_round=3, **options)
                trainer.train(root / 'part', resume=root / 'part/checkpoint_000003.pt', **options)

            def rows(name):
                data = [json.loads(x) for x in (root / name / 'generations.jsonl').read_text().splitlines()]
                for row in data:
                    row.pop('elapsed_seconds', None)
                return data
            self.assertEqual(rows('full'), rows('part'))
            for row in rows('full'):
                self.assertIn('greedy_disagreement', row)
                self.assertGreater(row['decisions'], 0)
                self.assertTrue(0 <= row['greedy_disagreement'] <= 1)
            snapshot = torch.load(root / 'full/generation_000002.pt', weights_only=True)
            self.assertEqual(snapshot['layers.0.weight'].shape[1], net.INPUT_SIZES[3])
            state = torch.load(root / 'full/checkpoint_000004.pt', weights_only=False)
            config = state['config']
            self.assertEqual((config['encoder_version'], config['target_blend'], config['check_selection']), (3, .5, 'greedy'))
            self.assertEqual(config['equity_table_sha256'], hashlib.sha256(table.read_bytes()).hexdigest())
            self.assertEqual(config['warm_start_sha256'], hashlib.sha256(warm.read_bytes()).hexdigest())
            metadata = json.loads((root / 'champions/reigning_champion.json').read_text())
            self.assertEqual(metadata['selection'], 'greedy')
            from phase2_loader import load_phase2_policy
            self.assertTrue(load_phase2_policy(root / 'champions/reigning_champion.pt', seed=1).greedy)

    def test_resume_with_a_different_target_blend_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            warm = root / 'warm.pt'
            noisy_network(6).save(warm)
            table = root / 'equity.json'
            small_table().save(table)
            options = self.options(root, warm, table)
            with patch('phase2_trainer.promotion_decision', return_value={'gain': 0, 'lower_bound': 0, 'upper_bound': 0, 'head_to_head': .5, 'promoted': False}):
                trainer.train(root / 'run', stop_after_round=2, **options)
                options['target_blend'] = .25
                with self.assertRaises(ValueError):
                    trainer.train(root / 'run', resume=root / 'run/checkpoint_000002.pt', **options)

    def test_invalid_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for bad in ({'target_blend': .5, 'equity_table': None}, {'target_blend': 1.5},
                        {'encoder_version': 7}, {'check_selection': 'argmax'}):
                options = self.options(root, None, None)
                options.update(bad)
                with self.assertRaises(ValueError):
                    trainer.train(root / 'bad', **options)


if __name__ == '__main__':
    unittest.main()
