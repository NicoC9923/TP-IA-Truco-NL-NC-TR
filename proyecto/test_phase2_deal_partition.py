import tempfile
import unittest
from unittest.mock import patch

import phase2_trainer
from symmetric_engine import play_match


def canonical(deal):
    return frozenset(frozenset(hand) for hand in deal)


class Phase2DealPartitionTests(unittest.TestCase):
    def test_training_and_reproduced_validation_deals_are_disjoint(self):
        training_deals = []
        partitions = []
        actual = phase2_trainer.play_match
        def capture(*policies, **kwargs):
            partitions.append(kwargs.get('deal_partition'))
            result = actual(*policies, **kwargs)
            training_deals.extend(map(canonical, result.deals))
            return result
        shares = dict.fromkeys(('self', 'snapshot', 'phase1', 'bluffer', 'cautious', 'aggressive', 'calculating'), 0)
        shares['self'] = 1
        with tempfile.TemporaryDirectory() as directory, patch.object(phase2_trainer, 'play_match', side_effect=capture), patch.object(phase2_trainer, 'training_seed', return_value=2022795541):
            phase2_trainer.train(directory, generations=1, matches_per_generation=1, matches_per_round=1,
                                 block_size=1, target=1, panel_path=None, seed=29, league_shares=shares)
        fold = lambda obs, actions: 'fold'
        validation = play_match(fold, fold, seed=20000000000048, target=1, deal_partition=0)
        self.assertEqual(partitions, [1])
        self.assertTrue(set(training_deals).isdisjoint(map(canonical, validation.deals)))

    def test_checks_use_validation_partition_for_panel_and_direct_matches(self):
        from pathlib import Path
        from test_phase2_progress import make_config
        seen = []
        actual = phase2_trainer.play_mirror_pair
        def capture(*policies, **kwargs):
            seen.append(kwargs.get('deal_partition'))
            return actual(*policies, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(phase2_trainer, 'play_mirror_pair', side_effect=capture):
            root = Path(directory)
            phase2_trainer.train(root / 'run', **make_config(root))
        self.assertGreaterEqual(len(seen), 6)
        self.assertEqual(set(seen), {0})

    def test_card_validation_replay_uses_validation_partition(self):
        import phase2_card_metrics as metrics
        from phase2_torch import Phase2Network
        seen = []
        actual = metrics.play_match
        def capture(*policies, **kwargs):
            seen.append(kwargs.get('deal_partition'))
            return actual(*policies, **kwargs)
        with patch.object(metrics, 'play_match', side_effect=capture):
            metrics.validation_predictions(Phase2Network(auxiliary=True), schedule_seed=12345,
                                           target=1, temperature=.1, situation_count=1)
        self.assertEqual(seen, [0])

    def test_many_training_and_validation_seeds_have_no_deal_collisions(self):
        from symmetric_engine import play_mirror_pair
        from validation_schedule import validation_schedule
        fold = lambda obs, actions: 'fold'
        training, validation = set(), set()
        for match_id, (seed, _) in enumerate(validation_schedule(10000, 1, 1000)):
            match = play_match(fold, fold, seed=phase2_trainer.training_seed(29, match_id), target=1, deal_partition=1)
            training.update(map(canonical, match.deals))
            pair = play_mirror_pair(fold, fold, seed=seed, target=1, deal_partition=0)
            self.assertEqual(pair[0].deals, pair[1].deals)
            validation.update(map(canonical, pair[0].deals))
        self.assertTrue(training.isdisjoint(validation))

    def test_phase1_panel_opponent_uses_validation_partition(self):
        import hashlib
        import json
        from pathlib import Path
        import numpy as np
        from phase1_network import GENOME_SIZE, NetworkPolicy
        from test_phase2_progress import make_config
        seen = []
        actual = phase2_trainer.play_mirror_pair
        def capture(a, b, **kwargs):
            if isinstance(b, NetworkPolicy):
                seen.append(kwargs.get('deal_partition'))
            return actual(a, b, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(phase2_trainer, 'play_mirror_pair', side_effect=capture):
            root = Path(directory)
            config = make_config(root)
            weights = root / 'phase1.npy'
            np.save(weights, np.zeros(GENOME_SIZE))
            config['panel_path'].write_text(json.dumps({'version': 0, 'measuring_only': True, 'members': [
                {'name': 'phase1', 'path': weights.name, 'kind': 'phase1', 'encoder': 'v1',
                 'sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}]}))
            phase2_trainer.train(root / 'run', **config)
        self.assertGreaterEqual(len(seen), 4)
        self.assertEqual(set(seen), {0})
