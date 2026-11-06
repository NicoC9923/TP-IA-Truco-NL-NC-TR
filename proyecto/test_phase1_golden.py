"""Golden phase-one behavior captured before inference optimization."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from phase1_network import GENOME_SIZE, NetworkPolicy
from phase1_trainer import train
from symmetric_engine import play_match

class GoldenPhase1Tests(unittest.TestCase):
    def test_matches_and_training_log(self):
        golden = json.loads(Path(__file__).with_name('golden_phase1.json').read_text(encoding='utf-8'))
        matches = []
        for seed in (11, 22, 33):
            rng = np.random.default_rng(seed)
            a = NetworkPolicy(rng.normal(0, .1, GENOME_SIZE), seed=seed * 2)
            b = NetworkPolicy(rng.normal(0, .1, GENOME_SIZE), seed=seed * 2 + 1)
            result = play_match(a, b, seed=seed, target=3)
            matches.append(dict(winner=result.winner, points=list(result.points), actions=[d['action'] for d in result.decisions]))
        self.assertEqual(matches, golden['matches'])
        with tempfile.TemporaryDirectory() as directory:
            train(directory, population_size=2, generations=2, pairs=1, workers=1, target=1, seed=13, validation_pairs=2)
            self.assertEqual(Path(directory, 'generations.jsonl').read_text(encoding='utf-8'), golden['generations'])
