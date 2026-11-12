import unittest
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from phase2_league import REFERENCE_IDENTITIES
from phase2_torch import Phase2Network
from phase2_trainer import _league_block, _league_chunk


class ReferenceBlockTest(unittest.TestCase):
    def test_blocks_repeat_across_workers_and_record_only_learner(self):
        torch.manual_seed(7)
        weights = Phase2Network().state_dict()
        jobs = []
        for i, category in enumerate(('reactive', 'adaptive_utility')):
            spec = {'category': category, 'identity': REFERENCE_IDENTITIES[category],
                    'style': None, 'offset': 0, 'neutral_profile': False}
            jobs.append((weights, 9, [([2*i, 2*i+1], spec)], 15, 0, {}, .05, 5))
        sequential = [_league_chunk(job) for job in jobs]
        with ProcessPoolExecutor(max_workers=2) as pool:
            parallel = list(pool.map(_league_chunk, jobs))
        for first, second in zip(sequential, parallel):
            for (match, samples), (other_match, other_samples) in zip(first, second):
                self.assertEqual(match, other_match)
                self.assertTrue(samples)
                self.assertEqual(len(samples), len(other_samples))
                for sample, other in zip(samples, other_samples):
                    np.testing.assert_array_equal(sample[0], other[0])
                    self.assertEqual(sample[1:], other[1:])
                    self.assertEqual(sample[3], match % 2)

    def test_generation_log_counts_reference_draws(self):
        import json
        import tempfile
        from pathlib import Path
        from phase2_league import DEFAULT_SHARES
        from phase2_trainer import run_rounds
        for category in ('reactive', 'adaptive_utility'):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / 'generations.jsonl'
                shares = dict.fromkeys(DEFAULT_SHARES, 0)
                shares[category] = 1
                run_rounds(seed=9, rounds=1, matches_per_round=2, block_size=1,
                           target=1, generation_rounds=1, log_path=log, league_shares=shares)
                row = json.loads(log.read_text().splitlines()[0])
                self.assertEqual(row['league_mix'], {category: 2})
