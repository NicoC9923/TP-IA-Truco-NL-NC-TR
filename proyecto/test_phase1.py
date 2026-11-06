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

