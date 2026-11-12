"""The shipped phase 2 champion is run 3 generation 73 (ADR 0063): encoder v3, 1,635 inputs."""
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / 'champions/phase2'
RETIRED_RUN2 = SHIPPED / 'retired/run2-g38'
G73_SHA = '9866d6e406ef23502976d201cf185702aae9aec46388c37ba3a32ce050c20a47'
G38_SHA = '95eee708e8839233c27e4b3d710b6b5da74f3a149f7fe5ce8deef4cb720b60bf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ShippedFilesTests(unittest.TestCase):
    def test_shipped_champion_is_run3_g73(self):
        self.assertEqual(sha(SHIPPED / 'reigning_champion.pt'), G73_SHA)
        self.assertEqual(sha(REPO / 'champions/phase2-run3/reigning_champion.pt'), G73_SHA)
        sidecar = json.loads((SHIPPED / 'reigning_champion.json').read_text())
        self.assertEqual((sidecar['generation'], sidecar['run'], sidecar['selection'], sidecar['profile_prior']),
                         (73, 'run3', 'greedy', 5.0))

    def test_run2_g38_is_retired_with_its_sidecar(self):
        self.assertEqual(sha(RETIRED_RUN2 / 'reigning_champion.pt'), G38_SHA)
        sidecar = json.loads((RETIRED_RUN2 / 'reigning_champion.json').read_text())
        self.assertEqual((sidecar['generation'], sidecar['run'], sidecar['selection']), (38, 'run2', 'greedy'))

    def test_new_and_old_champions_load_with_their_own_encoder(self):
        from phase2_loader import load_phase2_policy
        new = load_phase2_policy(SHIPPED / 'reigning_champion.pt', seed=1)
        self.assertTrue(new.greedy)
        self.assertEqual((new.model.layers[0].in_features, new.model.encoder_version), (1635, 3))
        old = load_phase2_policy(RETIRED_RUN2 / 'reigning_champion.pt', seed=1)
        self.assertEqual((old.model.layers[0].in_features, old.model.encoder_version), (1628, 1))


class GameMatchesTrainerPathTests(unittest.TestCase):
    def play(self, match, decisions):
        for _ in range(600):
            if len(decisions) >= 12:
                return
            if match.mano_terminada:
                match.nueva_mano()
            elif match.canto_pendiente is not None:
                options = match.canto_pendiente['opciones']
                match.responder_canto_humano('QUIERO' if 'QUIERO' in options else options[0])
            elif match.cartas_humano_disponibles:
                match.jugar_carta_humano(match.cartas_humano_disponibles[0])

    def test_game_decision_equals_trainer_path_decision_for_real_observations(self):
        from partida_truco import PartidaTruco
        from phase2_game_policy import GamePhase2Policy
        from phase2_loader import load_phase2_policy
        from phase2_torch import Phase2Policy, _q_values
        import torch
        recorded = []
        original = GamePhase2Policy.__call__

        def recording(policy, observation, legal_actions):
            captured = {}
            policy.record_decision = lambda inputs, action: captured.update(inputs=inputs.copy())
            action = original(policy, observation, legal_actions)
            recorded.append((observation, tuple(legal_actions), action, captured['inputs']))
            return action

        with tempfile.TemporaryDirectory() as tmp, patch.object(GamePhase2Policy, '__call__', recording):
            match = PartidaTruco(historial_path=str(Path(tmp) / 'history.json'), rng=random.Random(11))
            match.cambiar_modo_ia('refuerzo')
            match.adapt_to_opponent = False
            self.assertEqual(match.network.model.layers[0].in_features, 1635)
            self.play(match, recorded)
        self.assertGreaterEqual(len(recorded), 4)
        trainer = load_phase2_policy(SHIPPED / 'reigning_champion.pt', seed=3, adapt=False)
        trainer = Phase2Policy(trainer.model, seed=3, temperature=trainer.temperature,
                               profile_prior=5.0, greedy=True, adapt=False)
        previous = -1
        for observation, legal, action, game_inputs in recorded:
            if len(observation['history']) <= previous:
                trainer.exact_inference.reset(); trainer.weighted_inference.reset()
            previous = len(observation['history'])
            inputs = trainer.encode(observation, legal)
            self.assertEqual(inputs.shape, (1635,))
            np.testing.assert_array_equal(inputs, game_inputs)
            self.assertEqual(trainer(observation, legal), action)
            # The hand-fact columns are really filled in, not left at zero.
        self.assertTrue(any(np.any(inputs[1628:]) for *_, inputs in recorded))


if __name__ == '__main__':
    unittest.main()
