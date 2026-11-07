import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from phase1_trainer import train


class Phase1DefaultsTests(unittest.TestCase):
    def test_default_patience_is_recorded_in_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            train(directory, population_size=2, generations=10, pairs=1, workers=1,
                  target=1, validation_pairs=1, promotion_threshold=0)
            with np.load(Path(directory) / 'checkpoint_0010.npz') as checkpoint:
                state = json.loads(str(checkpoint['state']))
            self.assertEqual(state['config'][6], 50)
    def test_shared_league_schedule_is_recorded(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            train(directory, population_size=4, generations=2, pairs=8, workers=1,
                  target=1, validation_pairs=1)
            entries = [json.loads(line) for line in (Path(directory) / 'generations.jsonl').read_text().splitlines()]
            for entry in entries:
                self.assertEqual(len(entry['schedule']), 8)
                self.assertEqual(sum(item['style'] is None for item in entry['schedule']), 4)
            self.assertEqual(entries[1]['schedule'][0]['opponent'], 'reigning')
    def test_finalist_retest_scores_are_logged(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            train(directory, population_size=3, generations=1, pairs=1, workers=1,
                  target=1, validation_pairs=1, finalists=2, finalist_pairs_multiplier=2)
            entry = json.loads((Path(directory) / 'generations.jsonl').read_text())
            self.assertEqual(len(entry['finalists']), 2)
            self.assertEqual(entry['finalist_pairs'], 2)
            self.assertEqual(entry['selected_individual'], max(entry['finalists'], key=lambda row: row['score'])['individual'])
    def test_output_initialization_sigma_is_configurable(self):
        from phase1_network import INPUTS, HIDDEN
        with tempfile.TemporaryDirectory() as directory:
            train(directory, population_size=2, generations=1, pairs=1, workers=1,
                  target=1, finalists=0, output_init_sigma=0)
            champion = np.load(Path(directory) / 'champion_0001.npy')
            self.assertTrue(np.all(champion[INPUTS * HIDDEN + HIDDEN:] == 0))
            self.assertTrue(np.any(champion[:INPUTS * HIDDEN + HIDDEN] != 0))
    def test_finalist_retest_can_be_disabled(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            train(directory, population_size=2, generations=1, pairs=1, workers=1,
                  target=1, finalists=0)
            entry = json.loads((Path(directory) / 'generations.jsonl').read_text())
            self.assertEqual(entry['finalists'], [])
            self.assertEqual(entry['finalist_pairs'], 0)
    def test_fitness_modes_change_logged_scores_as_specified(self):
        scores = []
        for mode, weight in [('classic', 0), ('classic', 1), ('mixed', .5), ('mixed', 1)]:
            with tempfile.TemporaryDirectory() as directory:
                train(directory, population_size=2, generations=1, pairs=1, workers=1,
                      target=1, seed=12, finalists=0, fitness_mode=mode, points_weight=weight)
                entry = json.loads((Path(directory) / 'generations.jsonl').read_text())
                scores.append(entry['fitness']['average'])
        wins, classic, mixed, points = scores
        self.assertAlmostEqual(mixed, classic / 2)
        self.assertAlmostEqual(points, classic - wins)
