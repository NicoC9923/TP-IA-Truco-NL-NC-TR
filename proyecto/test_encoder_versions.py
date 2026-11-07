import hashlib
import unittest
import numpy as np
import truco_reglas as rules
from opponent_profile import OpponentProfile
from phase1_network import encode_observation, NetworkPolicy, GENOME_SIZE


def observation():
    return {'player': 1, 'hand': ((1, 'Espada'), (2, 'Oro'), (3, 'Copa')), 'available': (0, 1, 2), 'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1, 'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}


class EncoderTests(unittest.TestCase):
    def test_v1_golden(self):
        obs = observation()
        for available, expected in zip(((0, 1, 2), (1, 2), ()), ('c14a4fedf50845f6c8403d5e1199477217ce2d30ec158caeddd0cd5aa005901f', '75d9e506d8cf38ac32ca0838e7b08b0d45592e8183ef2e3189074986e58460d2', 'a3e93030bf647f72bf58405d7c6e6c79e862438c083990a7bac089350be5a911')):
            obs['available'] = available
            self.assertEqual(hashlib.sha256(encode_observation(obs, OpponentProfile(), encoder='v1').tobytes()).hexdigest(), expected)

    def test_all_cards_use_rule_strength_and_only_held_features_change(self):
        obs = observation()
        for value in rules.VALORES:
            for suit in rules.PALOS:
                obs['hand'] = ((value, suit), (2, 'Oro'), (3, 'Copa'))
                a = encode_observation(obs, OpponentProfile(), encoder='v1')
                b = encode_observation(obs, OpponentProfile(), encoder='v2')
                self.assertEqual(b[0], rules.valor_carta_truco(value, suit) / 13)
                np.testing.assert_array_equal(np.delete(a, [0, 4, 8]), np.delete(b, [0, 4, 8]))

    def test_unknown_encoder_rejected(self):
        with self.assertRaises(ValueError):
            encode_observation(observation(), OpponentProfile(), encoder='v3')
        with self.assertRaises(ValueError):
            NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v3')

class ArtifactTests(unittest.TestCase):
    def test_policy_sidecar_and_bare_genome(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'champion.npy'
            NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v2').save(path)
            self.assertEqual(NetworkPolicy.load(path).encoder, 'v2')
            path.with_suffix('.encoder.json').unlink()
            self.assertEqual(NetworkPolicy.load(path).encoder, 'v1')

    def test_resume_rejects_encoder_mismatch(self):
        import tempfile
        import json
        from pathlib import Path
        from phase1_trainer import train
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'old.npz'
            state = {'config': [2, 1, 1, 0, 1, .55, 50, 200, 0, 4, .3, 'classic', .1, .5, True, 'reject-only'], 'encoder': 'v2'}
            np.savez(checkpoint, population=np.zeros((2, GENOME_SIZE)), reigning=np.zeros(GENOME_SIZE), hall=np.empty((0, GENOME_SIZE)), state=json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'encoder'):
                train(directory, population_size=2, target=1, workers=1, validation_pairs=1, finalists=0, resume=checkpoint)

class TrainerEncoderTests(unittest.TestCase):
    def test_tiny_v2_run_uses_v2_for_all_matches(self):
        import tempfile
        from unittest.mock import patch
        from phase1_trainer import train
        import phase1_network
        seen = []
        original = phase1_network.encode_observation
        def record(*args, **kwargs):
            seen.append(kwargs.get('encoder'))
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch('phase1_network.encode_observation', side_effect=record):
            train(directory, population_size=2, generations=2, pairs=1, workers=1, target=1, validation_pairs=1, finalists=1, finalist_pairs_multiplier=1, encoder='v2')
        self.assertTrue(seen)
        self.assertEqual(set(seen), {'v2'})

class EvaluationEncoderTests(unittest.TestCase):
    def test_mixed_encoder_mirror_uses_both_encoders(self):
        from unittest.mock import patch
        from panel_evaluation import _measure
        original = NetworkPolicy.__call__
        seen = []
        def record(policy, observation, legal_actions):
            seen.append((policy.encoder, observation['player']))
            return original(policy, observation, legal_actions)
        with patch.object(NetworkPolicy, '__call__', record):
            _measure(np.zeros(GENOME_SIZE), np.zeros(GENOME_SIZE), 2, 123, 1, 'v2', 'v1')
        self.assertEqual(set(seen), {('v1', 0), ('v1', 1), ('v2', 0), ('v2', 1)})

    def test_reference_uses_champion_sidecar(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from reference_evaluation import evaluate
        original = NetworkPolicy.__call__
        seen = []
        def record(policy, observation, legal_actions):
            seen.append(policy.encoder)
            return original(policy, observation, legal_actions)
        with tempfile.TemporaryDirectory() as directory:
            NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v2').save(Path(directory) / 'champion_0001.npy')
            with patch.object(NetworkPolicy, '__call__', record):
                evaluate(directory, pairs=1, target=1)
        self.assertTrue(seen)
        self.assertEqual(set(seen), {'v2'})
