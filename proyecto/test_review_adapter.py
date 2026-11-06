import unittest
from unittest.mock import Mock, patch
from engine_policies import ReactivePolicy

class AdapterReviewTests(unittest.TestCase):
    def test_pending_responses_receive_score_and_raise(self):
        policy = ReactivePolicy(1)
        policy.agent.responder_envido = Mock(return_value=('ENVIDO_ENVIDO', None))
        obs = dict(hand=((1, 'oro'), (2, 'oro'), (3, 'oro')), available=(0,1,2), pending=dict(kind='envido', call='envido', declined=2), table=(), tricks=(), player=0, mano=0, score=(12, 25), target=30)
        self.assertEqual(policy(obs, ('envido','no_quiero')), 'envido')
        policy.agent.responder_envido.assert_called_once_with(unittest.mock.ANY, 'ENVIDO', True, puntos_acumulados_no_quiero=2, puntos_humano=25, puntos_robot=12, meta_puntos=30)
        obs['pending'] = dict(kind='truco', level=2)
        policy.agent.responder_canto_truco = Mock(return_value=('QUIERO', None))
        self.assertEqual(policy(obs, ('quiero','no_quiero')), 'quiero')
        self.assertEqual(policy.agent.responder_canto_truco.call_args.kwargs, dict(puntos_humano=25, meta_puntos=30))

    def test_reactive_does_not_read_history(self):
        with patch('historial_truco.HistorialTruco.cargar', side_effect=AssertionError('history read')):
            ReactivePolicy(1)
    def test_adapter_matches_direct_agent_for_endgame_envido(self):
        hand = ((7, 'Oro'), (6, 'Oro'), (1, 'Basto'))
        obs = dict(hand=hand, available=(0,1,2), pending=dict(kind='envido', call='falta_envido', declined=3), table=(), tricks=(), player=1, mano=0, score=(23, 11), target=30)
        adapter, direct = ReactivePolicy(7), ReactivePolicy(7)
        cards = [{'valor': v, 'palo': p} for v,p in hand]
        expected = direct.agent.responder_envido(cards, 'FALTA_ENVIDO', False, puntos_acumulados_no_quiero=3, puntos_humano=23, puntos_robot=11, meta_puntos=30)[0].lower()
        self.assertEqual(adapter(obs, ('quiero', 'no_quiero')), expected)
