import http.client
import json
import random
import tempfile
import threading
import unittest
from pathlib import Path
from partida_truco import PartidaTruco
from servidor_truco import crear_servidor

class NetworkModesServerTests(unittest.TestCase):
    def test_modes_and_unavailable_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            match = PartidaTruco(historial_path=str(Path(tmp) / 'history.json'), rng=random.Random(0))
            server = crear_servidor(match, puerto=0, sin_robot=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
                connection.request('GET', '/truco/estado')
                state = json.loads(connection.getresponse().read())
                modes = {m['codigo']: m for m in state['aprendizaje_ia']['modos_disponibles']}
                self.assertEqual(set(modes), {'reactivo', 'utilidad_adaptativa', 'neuroevolucion', 'refuerzo'})
                self.assertTrue(modes['neuroevolucion']['disponible'])
                self.assertTrue(modes['refuerzo']['disponible'])
                connection.request('POST', '/truco/modo_ia', json.dumps({'modo': 'foo'}))
                response = connection.getresponse(); self.assertEqual(response.status, 400); response.read()
                connection.request('GET', '/truco/estado')
                self.assertEqual(json.loads(connection.getresponse().read())['aprendizaje_ia']['modo_agente'], 'utilidad_adaptativa')
                connection.request('POST', '/truco/modo_ia', json.dumps({'modo': 'refuerzo'}))
                response = connection.getresponse(); self.assertEqual(response.status, 200); response.read()
                self.assertEqual(match.ia.modo_agente, 'refuerzo')
                self.assertEqual(match.network.encoder, 'phase2')
                connection.request('POST', '/truco/modo_ia', json.dumps({'modo': ' NEUROEVOLUCION '}))
                response = connection.getresponse(); self.assertEqual(response.status, 200); response.read()
                self.assertEqual(match.ia.modo_agente, 'neuroevolucion')
                connection.close()
            finally:
                server.shutdown(); thread.join(); server.server_close()
    def test_phase2_game_uses_published_model_and_resets_adaptation(self):
        from phase2_torch import Phase2Network
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            match = PartidaTruco(historial_path=str(Path(tmp) / 'history.json'), rng=random.Random(7))
            with patch.object(Phase2Network, 'load', wraps=Phase2Network.load) as load:
                match.cambiar_modo_ia('refuerzo')
                self.assertEqual(load.call_args.args[0], match.ia.champions_dir / 'phase2/reigning_champion.pt')
            self.assertTrue(match.adapt_to_opponent)
            match.adapt_to_opponent = False
            match.jugar_carta_humano(match.cartas_humano_disponibles[0])
            self.assertTrue(match.ia.traza_razonamiento)
            self.assertIsNone(match.ia.traza_razonamiento[-1]['perfil_rival'])
            match.reiniciar_partido()
            self.assertTrue(match.adapt_to_opponent)

    def test_phase2_mode_unavailable_without_torch_or_file(self):
        from unittest.mock import patch
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            match = PartidaTruco(historial_path=str(Path(tmp) / 'history.json'), champions_dir=tmp)
            self.assertFalse(match.ia.mode_available('refuerzo'))
            phase2 = Path(tmp) / 'phase2'; phase2.mkdir()
            (phase2 / 'reigning_champion.pt').write_bytes(b'placeholder')
            with patch.dict(sys.modules, {'torch': None}):
                self.assertFalse(match.ia.mode_available('refuerzo'))
                match.cambiar_modo_ia('refuerzo')
                self.assertEqual(match.ia.modo_agente, 'utilidad_adaptativa')

    def test_phase2_sidecar_sets_sampling_and_profile_prior(self):
        from phase2_torch import Phase2Network
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); phase2 = root / 'phase2'; phase2.mkdir()
            Phase2Network().save(phase2 / 'reigning_champion.pt')
            match = PartidaTruco(historial_path=str(root / 'history.json'), champions_dir=root)
            self.assertFalse(match.ia.mode_available('refuerzo'))
            sidecar = phase2 / 'reigning_champion.json'
            for invalid in ('{}', '{"temperature": 0, "profile_prior": 9}', '{"temperature": "bad", "profile_prior": 9}'):
                sidecar.write_text(invalid)
                self.assertFalse(match.ia.mode_available('refuerzo'))
                with self.assertRaises(ValueError):
                    from phase2_game_policy import GamePhase2Policy
                    GamePhase2Policy.load(phase2 / 'reigning_champion.pt', seed=1)
            sidecar.write_text('{"temperature": 0.3, "profile_prior": 9}')
            self.assertTrue(match.ia.mode_available('refuerzo'))
            match.cambiar_modo_ia('refuerzo')
            self.assertEqual(match.network.temperature, .3)
            self.assertEqual(match.network.profile.prior_strength, 9)
            self.assertEqual(match.network.neutral_profile.prior_strength, 9)
            self.assertEqual(match.opponent_profile_phase2.prior_strength, 9)
            match.adapt_to_opponent = False
            match.jugar_carta_humano(match.cartas_humano_disponibles[0])
            self.assertEqual(match.network.profile.prior_strength, 9)

    def test_truco_after_robot_plays_all_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            match = PartidaTruco(historial_path=str(Path(tmp) / 'history.json'), rng=random.Random(7))
            match.cambiar_modo_ia('neuroevolucion')
            match.irse_al_mazo_humano()
            match.nueva_mano()
            while match.cartas_robot_disponibles and not match.mano_terminada:
                if match.canto_pendiente:
                    match.responder_canto_humano('QUIERO')
                else:
                    match.jugar_carta_humano(match.cartas_humano_disponibles[0])
            self.assertFalse(match.mano_terminada)
            response = match.cantar_truco_humano('TRUCO')
            self.assertNotIn('error', response)
            trace = match.ia.traza_razonamiento[-1]
            self.assertTrue({'acciones_probables', 'inferencia', 'es_farol', 'perfil_rival'} <= trace.keys())

    def test_network_mode_plays_and_explains_after_hand(self):
        with tempfile.TemporaryDirectory() as tmp:
            match = PartidaTruco(historial_path=str(Path(tmp) / 'history.json'), rng=random.Random(7))
            match.cambiar_modo_ia('neuroevolucion')
            match.jugar_carta_humano(match.cartas_humano_disponibles[0])
            self.assertTrue(match.ia.traza_razonamiento)
            trace = match.ia.traza_razonamiento[-1]
            self.assertIn('acciones_probables', trace)
            self.assertIn('inferencia', trace)
            self.assertIn('es_farol', trace)
            self.assertIn('perfil_rival', trace)
    def test_adaptation_and_reset_survive_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = str(Path(tmp) / 'history.json')
            match = PartidaTruco(historial_path=history, rng=random.Random(2))
            match.irse_al_mazo_humano()
            self.assertGreater(match.opponent_profile.rates()['fold'], .5)
            match = PartidaTruco(historial_path=history, rng=random.Random(2))
            server = crear_servidor(match, puerto=0, sin_robot=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
                connection.request('POST', '/truco/adaptacion', json.dumps({'activa': False}))
                response = connection.getresponse(); self.assertEqual(response.status, 200); response.read()
                self.assertFalse(match.estado_publico()['aprendizaje_ia']['adaptacion_activa'])
                connection.request('POST', '/truco/perfil/reset', '{}')
                response = connection.getresponse(); self.assertEqual(response.status, 200); response.read()
                self.assertEqual(PartidaTruco(historial_path=history).opponent_profile.rates()['fold'], .5)
                connection.close()
            finally:
                server.shutdown(); thread.join(); server.server_close()
    def test_missing_champion_is_unavailable_without_changing_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            match = PartidaTruco(historial_path=str(Path(tmp) / 'history.json'), champions_dir=tmp)
            server = crear_servidor(match, puerto=0, sin_robot=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
                connection.request('GET', '/truco/estado')
                state = json.loads(connection.getresponse().read())
                mode = next(m for m in state['aprendizaje_ia']['modos_disponibles'] if m['codigo'] == 'neuroevolucion')
                self.assertFalse(mode['disponible'])
                connection.request('POST', '/truco/modo_ia', json.dumps({'modo': 'neuroevolucion'}))
                response = connection.getresponse(); self.assertEqual(response.status, 409); response.read()
                self.assertEqual(match.estado_publico()['aprendizaje_ia']['modo_agente'], 'utilidad_adaptativa')
                connection.close()
            finally:
                server.shutdown(); thread.join(); server.server_close()

    def test_completed_hand_trace_from_server_includes_network_explanation(self):
        with tempfile.TemporaryDirectory() as tmp:
            match = PartidaTruco(historial_path=str(Path(tmp) / 'history.json'), rng=random.Random(7))
            server = crear_servidor(match, puerto=0, sin_robot=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
                def post(route, body):
                    connection.request('POST', route, json.dumps(body))
                    response = connection.getresponse(); self.assertEqual(response.status, 200)
                    return json.loads(response.read())
                post('/truco/modo_ia', {'modo': 'neuroevolucion'})
                post('/truco/jugar', {'carta': match.cartas_humano_disponibles[0]})
                post('/truco/mazo', {})
                connection.request('GET', '/truco/historial')
                trace = json.loads(connection.getresponse().read())['traza_razonamiento'][0]
                self.assertEqual(set(('acciones_probables', 'inferencia', 'es_farol', 'perfil_rival')) - set(trace), set())
                self.assertEqual(len(trace['acciones_probables']), 3)
                connection.close()
            finally:
                server.shutdown(); thread.join(); server.server_close()
    def test_switch_changes_belief_and_profile_used_by_network(self):
        from opponent_profile import OpponentProfile
        with tempfile.TemporaryDirectory() as tmp:
            history = str(Path(tmp) / 'history.json')
            profile = OpponentProfile(Path(tmp) / 'opponent_profile.json')
            event = {'player': 0, 'action': 'truco', 'observation': {'pending': None}, 'revealed': True, 'bluff': True}
            profile.observe([event] * 100, 0)
            traces = []
            for active in (True, False):
                match = PartidaTruco(historial_path=history, rng=random.Random(19))
                server = crear_servidor(match, puerto=0, sin_robot=True)
                thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
                try:
                    conn = http.client.HTTPConnection('127.0.0.1', server.server_port)
                    def post(path, payload):
                        conn.request('POST', path, json.dumps(payload))
                        r = conn.getresponse(); self.assertEqual(r.status, 200); return json.loads(r.read())
                    post('/truco/modo_ia', {'modo': 'neuroevolucion'})
                    post('/truco/adaptacion', {'activa': active})
                    post('/truco/truco', {'nivel': 'TRUCO'})
                    if not match.mano_terminada:
                        post('/truco/mazo', {})
                    conn.request('GET', '/truco/historial')
                    trace = json.loads(conn.getresponse().read())['traza_razonamiento'][-1]
                    traces.append(trace)
                    conn.close()
                finally:
                    server.shutdown(); thread.join(); server.server_close()
            self.assertGreater(traces[0]['perfil_rival']['bluff'], .9)
            self.assertIsNone(traces[1]['perfil_rival'])
            self.assertNotEqual(traces[0]['inferencia']['expected_strength'], traces[1]['inferencia']['expected_strength'])

    def test_shipped_champion_and_trace_match_explicit_v1(self):
        from unittest.mock import patch
        from phase1_network import NetworkPolicy
        with tempfile.TemporaryDirectory() as tmp:
            def play(name, explicit):
                match = PartidaTruco(historial_path=str(Path(tmp) / name), rng=random.Random(7))
                if explicit:
                    original = NetworkPolicy.load
                    with patch.object(NetworkPolicy, 'load', side_effect=lambda path, **kwargs: original(path, encoder='v1', **kwargs)):
                        match.cambiar_modo_ia('neuroevolucion')
                else:
                    match.cambiar_modo_ia('neuroevolucion')
                self.assertEqual(match.network.encoder, 'v1')
                match.jugar_carta_humano(match.cartas_humano_disponibles[0])
                # Separate runs can cross a wall-clock second boundary.
                return [{key: value for key, value in row.items() if key != 'hora'}
                        for row in match.ia.traza_razonamiento]
            self.assertEqual(play('default.json', False), play('explicit.json', True))
