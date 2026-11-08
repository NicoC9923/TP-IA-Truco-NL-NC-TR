import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from phase1_network import encode_observation, NetworkPolicy, GENOME_SIZE
from opponent_profile import OpponentProfile
from phase1_trainer import _behavior
from partida_truco import PartidaTruco
from test_encoder_versions import observation


def event(action, pending):
    return dict(player=0, action=action, hand_index=0, bluff=None, revealed=False,
                observation={'pending': None if pending is None else {'kind': pending}})

class Chunk2Tests(unittest.TestCase):
    def test_stakes_distinguish_aliases_without_changing_old_encoders(self):
        a = observation(); b = observation()
        a['pending'] = dict(kind='envido', caller=0, call='real_envido', accepted=3, declined=1)
        b['pending'] = dict(kind='envido', caller=0, call='real_envido', accepted=7, declined=4)
        for version in ('v1', 'v2'):
            np.testing.assert_array_equal(encode_observation(a, OpponentProfile(), encoder=version), encode_observation(b, OpponentProfile(), encoder=version))
        x = encode_observation(a, OpponentProfile(), encoder='v3')
        y = encode_observation(b, OpponentProfile(), encoder='v3')
        np.testing.assert_allclose(x[60:63], [3/30, -3/30, -1/30])
        np.testing.assert_allclose(y[60:63], [7/30, -7/30, -4/30])

    def test_falta_depends_on_winner_and_target(self):
        obs = observation(); obs['score'] = (8, 3)
        obs['pending'] = dict(kind='envido', caller=0, call='falta_envido', accepted=2, declined=2)
        for target, expected in ((15, (7, -7, -2)), (30, (27, -22, -2))):
            obs['target'] = target
            np.testing.assert_allclose(encode_observation(obs, OpponentProfile(), encoder='v3')[60:63], np.array(expected)/30)
        obs['pending'] = None
        np.testing.assert_array_equal(encode_observation(obs, OpponentProfile(), encoder='v3')[60:63], [0, 0, 0])

    def test_per_policy_semantics_and_behavior(self):
        decisions = [event('envido', 'truco'), event('quiero', 'truco')]
        legacy = NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v2')
        modern = NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v3')
        legacy.observe_hand(decisions, 0); modern.observe_hand(decisions, 0)
        self.assertEqual(legacy.profile.counts['truco_accepted'], [1, 2])
        self.assertEqual(modern.profile.counts['truco_accepted'], [1, 1])
        self.assertEqual(_behavior([(SimpleNamespace(decisions=decisions), 0)])['truco_acceptance'], 1)
        for action, accepted in [('no_quiero', 0), ('fold', 0), ('retruco', 1), ('vale_cuatro', 1)]:
            policy = NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v3')
            policy.observe_hand([event(action, 'truco')], 0)
            self.assertEqual(policy.profile.counts['truco_accepted'], [accepted, 1])

    def test_interruption_in_mixed_matches(self):
        from symmetric_engine import play_match

        class Scripted(NetworkPolicy):
            def __init__(self, encoder, script):
                super().__init__(np.zeros(GENOME_SIZE), encoder=encoder)
                self.script = iter(script)
                self.snapshots = []

            def __call__(self, obs, legal):
                return next(self.script, 'fold')

            def observe_hand(self, decisions, opponent):
                super().observe_hand(decisions, opponent)
                self.snapshots.append(self.profile.counts['truco_accepted'][:])

        for observer_version, expected in (('v2', [1, 2]), ('v3', [1, 1])):
            caller = Scripted(observer_version, ['truco', 'quiero', 'fold'])
            interrupter = Scripted('v3' if observer_version == 'v2' else 'v2', ['envido', 'quiero'])
            result = play_match(caller, interrupter, seed=3, target=15)
            self.assertEqual([d['action'] for d in result.decisions[:4]],
                             ['truco', 'envido', 'quiero', 'quiero'])
            self.assertEqual(caller.snapshots[0], expected)

    def test_human_adapter_keeps_separate_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            match = PartidaTruco(historial_path=str(Path(tmp) / 'history.json'))
            match.network = NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v3')
            match._human_decisions = [event('envido', 'truco'), event('quiero', 'truco')]
            match._mano_analizada = False
            match._cerrar_aprendizaje_mano()
            self.assertEqual(match.opponent_profile.counts['truco_accepted'], [1, 2])
            self.assertEqual(match.opponent_profile_v3.counts['truco_accepted'], [1, 1])
