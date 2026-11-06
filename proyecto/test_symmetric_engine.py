import unittest

from symmetric_engine import play_match


class EngineTests(unittest.TestCase):
    def test_match_returns_winner_score_and_decisions(self):
        def fold(observation, actions):
            return 'fold'
        result = play_match(fold, fold, seed=7, target=2)
        self.assertEqual(result.winner, 1)
        self.assertEqual(result.points, (0, 2))
        self.assertEqual(result.decisions[0]['action'], 'fold')

    def test_card_play_is_seeded_and_hides_opponent_cards(self):
        def first_card(observation, actions):
            self.assertNotIn('opponent_hand', observation)
            return next(a for a in actions if a.startswith('card_'))
        one = play_match(first_card, first_card, seed=4, target=3)
        two = play_match(first_card, first_card, seed=4, target=3)
        self.assertEqual(one, two)
        self.assertTrue(any(d['action'].startswith('card_') for d in one.decisions))

    def test_rejects_illegal_action(self):
        with self.assertRaises(ValueError):
            play_match(lambda o, a: 'cheat', lambda o, a: 'fold')

    def test_starting_score_and_mirror_deals(self):
        from symmetric_engine import play_mirror_pair
        fold = lambda o, a: 'fold'
        first, second = play_mirror_pair(fold, fold, seed=8, target=15, starting_score=(13, 14))
        self.assertEqual(first.points, (13, 16))
        self.assertEqual(first.deals[0], second.deals[0])

    def test_truco_raise_and_rejection_scores_prior_level(self):
        def scripted(observation, actions):
            pending = observation['pending']
            if pending is None:
                return 'truco' if 'truco' in actions else 'card_0'
            return 'retruco' if 'retruco' in actions else 'no_quiero'
        result = play_match(scripted, scripted, seed=1, target=2)
        self.assertEqual(result.points, (0, 2))

    def test_runner_reports_throughput_and_mirrors(self):
        import subprocess
        import sys
        from pathlib import Path
        output = subprocess.check_output([sys.executable, str(Path(__file__).with_name('match_runner.py')), '--matches', '2', '--mirror'], text=True)
        self.assertIn('matches/second', output)
        self.assertIn('16 threads', output)
        self.assertIn('mirror', output)

    def test_mirror_keeps_mano_and_swaps_policies(self):
        from symmetric_engine import play_mirror_pair
        fold = lambda o, a: 'fold'
        first, second = play_mirror_pair(fold, fold, seed=9, target=2)
        self.assertEqual(first.decisions[0]['player'], 0)
        self.assertEqual(second.decisions[0]['player'], 0)

    def test_human_match_equivalence_for_accepted_envido_calls(self):
        import contextlib
        import io
        import random
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from partida_truco import PartidaTruco
        for call, target in [('envido', 2), ('real_envido', 3), ('falta_envido', 15)]:
            with self.subTest(call=call), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                human = PartidaTruco(meta_puntos=target, historial_path=str(Path(tmp) / 'history.json'), rng=random.Random(3))
                with patch.object(human.ia, 'responder_envido', return_value=('QUIERO', '')):
                    human.cantar_envido_humano(call.upper())
                def caller(obs, actions):
                    return call if call in actions else 'quiero'
                result = play_match(caller, lambda o, a: 'quiero', seed=3, target=target)
                score = human.estado_publico()['tanteador']
                self.assertEqual(result.points, (score['humano'], score['robot']))

    def test_human_match_equivalence_for_truco_levels(self):
        import contextlib
        import io
        import random
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from partida_truco import PartidaTruco
        for level in (2, 3, 4):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                human = PartidaTruco(historial_path=str(Path(tmp) / 'history.json'), rng=random.Random(3))
                with patch.object(human.ia, 'decidir_iniciar_envido', return_value=(None, '')), patch.object(human.ia, 'responder_canto_truco', side_effect=[('QUIERO' if level == 2 else 'RETRUCO', ''), ('QUIERO', '')]):
                    human.cantar_truco_humano()
                    if level == 3:
                        human.responder_canto_humano('QUIERO')
                    elif level == 4:
                        human.responder_canto_humano('VALE_CUATRO')
                    human.irse_al_mazo_humano()
                def a(obs, actions):
                    if 'truco' in actions: return 'truco'
                    if level == 4 and 'vale_cuatro' in actions: return 'vale_cuatro'
                    if 'quiero' in actions: return 'quiero'
                    return 'fold'
                def b(obs, actions):
                    if level > 2 and 'retruco' in actions: return 'retruco'
                    return 'quiero'
                result = play_match(a, b, seed=3, target=level)
                score = human.estado_publico()['tanteador']
                self.assertEqual(result.points, (score['humano'], score['robot']))

    def test_calls_record_objective_bluff_status(self):
        def caller(obs, actions):
            return 'envido' if 'envido' in actions else 'fold'
        result = play_match(caller, lambda o, a: 'no_quiero', seed=3, target=1)
        self.assertIsInstance(result.decisions[0]['bluff'], bool)

    def test_human_match_equivalence_for_parda_and_folding(self):
        import contextlib
        import io
        import random
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from partida_truco import PartidaTruco
        for seed, fold in ((16, False), (3, True)):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                human = PartidaTruco(meta_puntos=1, historial_path=str(Path(tmp) / 'history.json'), rng=random.Random(seed))
                with patch.object(human.ia, 'decidir_iniciar_envido', return_value=(None, '')), patch.object(human.ia, 'decidir_iniciar_truco', return_value=(None, '')), patch.object(human.ia, 'elegir_carta_a_jugar', side_effect=lambda cards, *args, **kwargs: cards[0]):
                    if fold:
                        human.irse_al_mazo_humano()
                    else:
                        while not human.estado_publico()['mano_terminada']:
                            human.jugar_carta_humano(human.estado_publico()['cartas_humano_disponibles'][0])
                policy = (lambda o, a: 'fold') if fold else (lambda o, a: next(x for x in a if x.startswith('card_')))
                result = play_match(policy, policy, seed=seed, target=1)
                score = human.estado_publico()['tanteador']
                self.assertEqual(result.points, (score['humano'], score['robot']))

    def test_human_match_equivalence_when_envido_interrupts_truco(self):
        import contextlib
        import io
        import random
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from partida_truco import PartidaTruco
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            human = PartidaTruco(meta_puntos=3, historial_path=str(Path(tmp) / 'history.json'), rng=random.Random(3))
            with patch.object(human.ia, 'decidir_iniciar_envido', return_value=('ENVIDO', '')), patch.object(human.ia, 'responder_canto_truco', return_value=('NO_QUIERO', '')):
                human.cantar_truco_humano()
                human.responder_canto_humano('QUIERO')
            def a(obs, actions):
                if 'truco' in actions: return 'truco'
                if 'quiero' in actions: return 'quiero'
                return 'fold'
            def b(obs, actions):
                if 'envido' in actions: return 'envido'
                return 'no_quiero'
            result = play_match(a, b, seed=3, target=3)
            score = human.estado_publico()['tanteador']
            self.assertEqual(result.points, (score['humano'], score['robot']))

    def test_fold_during_envido_awards_declined_envido_and_hand(self):
        def caller(obs, actions):
            return 'envido' if 'envido' in actions else 'fold'
        result = play_match(caller, lambda o, a: 'fold', seed=1, target=2)
        self.assertEqual(result.points, (2, 0))


    def test_observation_includes_prior_calls_without_hidden_cards(self):
        seen = []
        def caller(obs, actions):
            seen.append(obs)
            return 'envido' if 'envido' in actions else 'fold'
        play_match(caller, lambda o, a: 'no_quiero', seed=1, target=2, starting_score=(0, 1))
        self.assertEqual(seen[-1]['history'][:2], ((0, 'envido'), (1, 'no_quiero')))

