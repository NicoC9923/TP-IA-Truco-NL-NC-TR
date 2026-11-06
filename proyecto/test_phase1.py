import unittest
import numpy as np
from symmetric_engine import play_match
from phase1_network import NetworkPolicy, GENOME_SIZE

class NetworkEngineTests(unittest.TestCase):
    def test_network_plays_a_complete_match_with_legal_actions(self):
        policy = NetworkPolicy(np.zeros(GENOME_SIZE), seed=1)
        result = play_match(policy, NetworkPolicy(np.zeros(GENOME_SIZE), seed=2), seed=3, target=2)
        self.assertTrue(result.decisions)
        self.assertTrue(all(d['action'] in d['legal_actions'] for d in result.decisions))

class TrainerTests(unittest.TestCase):
    def test_training_reports_league_and_check_times(self):
        import contextlib
        import io
        import tempfile
        from phase1_trainer import train
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output):
            train(directory, population_size=2, generations=2, pairs=1, workers=1, target=1, seed=13, validation_pairs=2)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertRegex(line, r"\| league \d+\.\d{2} s \| check \d+\.\d{2} s$")

    def test_generation_logs_and_saves_a_loadable_champion(self):
        import json
        import tempfile
        from pathlib import Path
        from phase1_trainer import train
        with tempfile.TemporaryDirectory() as directory:
            train(directory, population_size=2, generations=1, pairs=1, workers=1, target=1, seed=4)
            lines = Path(directory, 'generations.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry['generation'], 1)
            self.assertEqual(entry['phase'], 1)
            self.assertEqual(set(entry['fitness']), {'max', 'average', 'min'})
            self.assertEqual(set(entry['behavior']), {'envido', 'bluff', 'truco_acceptance'})
            self.assertIn('diversity', entry)
            self.assertTrue(entry['example_hand'])
            NetworkPolicy.load(Path(directory, entry['champion']))

class PolicyBehaviorTests(unittest.TestCase):
    def test_bias_increases_envido_frequency_in_real_matches(self):
        def count(bias):
            total = 0
            for seed in range(8):
                policy = NetworkPolicy(np.zeros(GENOME_SIZE), seed=seed, bias=bias)
                opponent = NetworkPolicy(np.zeros(GENOME_SIZE), seed=seed + 100)
                result = play_match(policy, opponent, seed=seed, target=2)
                total += sum(d['player'] == 0 and d['action'] == 'envido' for d in result.decisions)
            return total
        self.assertGreater(count({'envido': 8}), count({'envido': -8}))

    def test_folding_scores_below_playing(self):
        from phase1_trainer import fitness
        class Fold:
            def __call__(self, obs, actions):
                return 'fold'
        class Play:
            def __call__(self, obs, actions):
                return 'quiero' if 'quiero' in actions else next(a for a in actions if a.startswith('card_'))
        results = [(play_match(Fold(), Play(), seed=i, target=5), 0) for i in range(4)]
        self.assertLess(fitness(results), fitness([(r, 1) for r, _ in results]))

class IndependenceTests(unittest.TestCase):
    def test_trainer_does_not_import_existing_agents(self):
        import subprocess
        import sys
        code = "import sys; import phase1_trainer; assert 'truco_ia' not in sys.modules; assert 'engine_policies' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], check=True, cwd=__import__('pathlib').Path(__file__).parent)


class ProfileMatchTests(unittest.TestCase):
    def test_network_learns_opponent_profile_during_match(self):
        class Fold:
            def __call__(self, obs, actions):
                return 'fold'
        policy = NetworkPolicy(np.zeros(GENOME_SIZE), seed=3)
        play_match(policy, Fold(), seed=7, target=5)
        self.assertGreater(policy.profile.rates()['fold'], .5)

    def test_seed_reproduces_generation(self):
        import tempfile
        from pathlib import Path
        from phase1_trainer import train
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for directory in (a, b):
                train(directory, population_size=2, generations=1, pairs=1, workers=1, target=1, seed=9)
            self.assertEqual(Path(a, 'generations.jsonl').read_text(), Path(b, 'generations.jsonl').read_text())
            np.testing.assert_array_equal(np.load(Path(a, 'champion_0001.npy')), np.load(Path(b, 'champion_0001.npy')))

class LeagueAcceptanceTests(unittest.TestCase):
    def test_training_records_validation_and_promotion(self):
        import tempfile, json
        from pathlib import Path
        from phase1_trainer import train
        with tempfile.TemporaryDirectory() as directory:
            train(directory, population_size=2, generations=2, pairs=1, workers=1, target=1, seed=5, validation_pairs=2)
            entries = [json.loads(line) for line in Path(directory, 'generations.jsonl').read_text().splitlines()]
            self.assertEqual(len(entries), 2)
            self.assertIn('promotion_check', entries[1])
            self.assertIn('margin_of_error', entries[1]['promotion_check'])
            self.assertIn('reigning_champion', entries[1])
            self.assertEqual(entries[1]['promoted'], entries[1]['promotion_check']['win_rate'] >= .55)

    def test_resume_and_promotions(self):
        import tempfile, json
        from pathlib import Path
        from phase1_trainer import train
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            options = dict(population_size=2, pairs=1, workers=1, target=1, seed=7, validation_pairs=1, patience=30, promotion_threshold=0)
            train(a, generations=11, **options)
            train(b, generations=10, **options)
            train(b, generations=11, resume=Path(b, 'checkpoint_0010.npz'), **options)
            self.assertEqual(Path(a, 'generations.jsonl').read_text(), Path(b, 'generations.jsonl').read_text())
            lines = [json.loads(s) for s in Path(a, 'generations.jsonl').read_text().splitlines()]
            self.assertTrue(lines[1]['promoted'])
            self.assertEqual(lines[1]['reigning_champion'], 2)
            with np.load(Path(b, 'checkpoint_0010.npz')) as checkpoint:
                self.assertEqual(len(checkpoint['hall']), 9)
                self.assertEqual(len(checkpoint['population']), 2)
        with tempfile.TemporaryDirectory() as d:
            options.update(promotion_threshold=1.1, patience=2)
            train(d, generations=4, **options)
            lines = [json.loads(s) for s in Path(d, 'generations.jsonl').read_text().splitlines()]
            self.assertEqual(len(lines), 3)
            self.assertFalse(lines[1]['promoted'])
            self.assertTrue(lines[-1]['phase_ended'])

    def test_league_mix_is_balanced(self):
        import tempfile, json
        from pathlib import Path
        from collections import Counter
        from phase1_trainer import train
        with tempfile.TemporaryDirectory() as d:
            train(d, population_size=8, generations=1, pairs=1, workers=1, target=1, seed=6, validation_pairs=1)
            entry = json.loads(Path(d, 'generations.jsonl').read_text())
            self.assertEqual(Counter(entry['league']), {'unbiased': 4, 'bluffer': 1, 'cautious': 1, 'aggressive': 1, 'calculating': 1})

    def test_styled_opponents_change_observed_behavior(self):
        def counts(style):
            calls = bluffs = accepts = chances = 0
            for seed in range(30):
                a = NetworkPolicy(np.zeros(GENOME_SIZE), seed=seed, style=style)
                b = NetworkPolicy(np.zeros(GENOME_SIZE), seed=seed + 100)
                result = play_match(a, b, seed=seed, target=3)
                for d in result.decisions:
                    if d['player'] != 0:
                        continue
                    if d['bluff'] is not None:
                        calls += 1
                        bluffs += bool(d['bluff'])
                    if 'quiero' in d['legal_actions']:
                        chances += 1
                        accepts += d['action'] != 'no_quiero'
            return bluffs, accepts / max(chances, 1)
        self.assertGreater(counts('bluffer')[0], counts('calculating')[0])
        self.assertLess(counts('cautious')[1], counts('aggressive')[1])

    def test_training_and_validation_deals_do_not_overlap(self):
        class Fold:
            def __call__(self, obs, actions):
                return 'fold'
        def canonical(deal):
            return tuple(sorted(tuple(sorted(hand)) for hand in deal))
        training, validation = set(), set()
        for seed in range(40):
            training.update(map(canonical, play_match(Fold(), Fold(), seed=seed, target=15, deal_partition=1).deals))
            validation.update(map(canonical, play_match(Fold(), Fold(), seed=seed, target=15, deal_partition=0).deals))
        self.assertFalse(training & validation)
