"""Synthetic protocol checks: no game engine matches are played."""
import importlib.util
import math
import json
import tempfile
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('round_robin', Path(__file__).with_name('run_rr.py'))
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

original_validate = rr.validate_records

def validate_small(rows, **kwargs):
    return original_validate(rows, pair_count=2, **kwargs)

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

    def test_constructor_seed_boundaries(self):
        rr.initialize()
        for seed in (2**128, 2**128 + 23999):
            for name in rr.NAMES:
                with self.subTest(seed=seed, name=name):
                    self.assertIsNotNone(rr.make_policy(name, seed))

    def test_validation_rejects_duplicates_incomplete_and_corruption(self):
        rows = [rr.record(0, i, [0, 1]) for i in range(2)]
        self.assertEqual(len(rr.validate_records(rows, matchup_count=1, pair_count=2)), 2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            rr.validate_records(rows + rows[:1], matchup_count=1, pair_count=2)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            rr.validate_records(rows[:1], matchup_count=1, pair_count=2)
        rows[0]['target'] = 30
        with self.assertRaisesRegex(ValueError, 'invalid'):
            rr.validate_records(rows, matchup_count=1, pair_count=2)

    def test_persistence_and_exclusive_run(self):
        class Pool:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def submit(self, function, job):
                f = Future()
                f.set_result([0, 1])
                return f
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'experiments/four-mode-round-robin'
            out.mkdir(parents=True)
            with patch.object(rr, 'ROOT', root), patch.object(rr, 'HASHES', {}), \
                 patch.object(rr, 'COUNT', 2), patch.object(rr, 'ProcessPoolExecutor', Pool), \
                 patch.object(rr, 'provenance', return_value={'synthetic': True}), \
                 patch.object(rr, 'validate_records', wraps=lambda rows, **kw: validate_small(rows, **kw)):
                rr.main()
                raw = [json.loads(line) for line in (out / 'partial.jsonl').read_text().splitlines()]
                self.assertEqual(len(raw), 12)
                self.assertEqual(len(json.loads((out / 'progress.json').read_text())['completed_matchups']), 6)
                self.assertEqual(json.loads((out / 'results.json').read_text())['pair_count'], 12)
                self.assertTrue((out / 'complete.txt').exists())
                with self.assertRaises(FileExistsError):
                    rr.main()

    def test_interruption_preserves_partial_and_refuses_retry(self):
        class Pool:
            calls = 0
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def submit(self, function, job):
                f = Future()
                Pool.calls += 1
                if Pool.calls == 2:
                    f.set_exception(RuntimeError('synthetic failure'))
                else:
                    f.set_result([0, 1])
                return f
        # Force deterministic completion order so one successful record precedes failure.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'experiments/four-mode-round-robin'
            out.mkdir(parents=True)
            with patch.object(rr, 'ROOT', root), patch.object(rr, 'HASHES', {}), \
                 patch.object(rr, 'COUNT', 2), patch.object(rr, 'ProcessPoolExecutor', Pool), \
                 patch.object(rr, 'as_completed', side_effect=lambda futures: iter(futures)), \
                 patch.object(rr, 'provenance', return_value={'synthetic': True}):
                with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                    rr.main()
                self.assertEqual(len((out / 'partial.jsonl').read_text().splitlines()), 1)
                self.assertFalse((out / 'results.json').exists())
                self.assertFalse((out / 'complete.txt').exists())
                with self.assertRaises(FileExistsError):
                    rr.main()

if __name__ == '__main__':
    unittest.main()
