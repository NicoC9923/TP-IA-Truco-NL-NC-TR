"""Synthetic protocol checks: no game engine matches are played."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('round_robin', Path(__file__).with_name('run_rr.py'))
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

class ProtocolTests(unittest.TestCase):
    def test_schedules_and_action_seeds_are_unique_and_balanced(self):
        deals = [rr.deal_seed(p, i) for p in range(6) for i in range(1000)]
        actions = [rr.action_seed(p, i, m, s) for p in range(6) for i in range(1000) for m in range(2) for s in range(2)]
        self.assertEqual(len(set(deals)), 6000)
        self.assertEqual(len(set(actions)), 24000)
        self.assertTrue(all(x % 2 == 0 for x in deals))
        self.assertEqual(min(actions), 2**128)
        self.assertEqual(max(actions), 2**128 + 23999)

    def test_mirror_fresh_instances_and_correct_scores(self):
        created, calls = [], []
        def factory(name, seed):
            item = (name, seed, object())
            created.append(item)
            return item
        def play(a, b, **kwargs):
            calls.append((a, b, kwargs))
            return SimpleNamespace(winner=len(calls) - 1)
        with patch.object(rr, 'make_policy', factory), patch.object(rr, 'play_match', play):
            winners = rr.play_pair((0, 3))
        self.assertEqual(winners, [0, 1])
        self.assertEqual([x[0] for x in created], ['reactive', 'adaptive_utility', 'adaptive_utility', 'reactive'])
        self.assertEqual(len({id(x[2]) for x in created}), 4)
        self.assertEqual(calls[0][2], calls[1][2])
        self.assertEqual(calls[0][2]['target'], 30)
        self.assertEqual(calls[0][2]['deal_partition'], 0)
        self.assertEqual(rr.summarize([winners, [1, 0]])['mean'], .5)

    def test_uncertainty_clusters_by_pair(self):
        result = rr.summarize([[0, 1], [1, 0], [0, 0], [1, 1]])
        self.assertEqual(result['mean'], .5)
        self.assertAlmostEqual(result['margin'], 1.96 * math.sqrt(1 / 6) / 2)

    def test_frozen_files(self):
        import hashlib
        for path, digest in rr.HASHES.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

if __name__ == '__main__':
    unittest.main()
