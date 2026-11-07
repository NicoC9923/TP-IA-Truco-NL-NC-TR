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
                self.assertFalse(modes['refuerzo']['disponible'])
                connection.request('POST', '/truco/modo_ia', json.dumps({'modo': 'foo'}))
                response = connection.getresponse(); self.assertEqual(response.status, 400); response.read()
                connection.request('GET', '/truco/estado')
                self.assertEqual(json.loads(connection.getresponse().read())['aprendizaje_ia']['modo_agente'], 'utilidad_adaptativa')
                connection.request('POST', '/truco/modo_ia', json.dumps({'modo': 'refuerzo'}))
                response = connection.getresponse(); self.assertEqual(response.status, 409); response.read()
                self.assertEqual(match.ia.modo_agente, 'utilidad_adaptativa')
                connection.request('POST', '/truco/modo_ia', json.dumps({'modo': ' NEUROEVOLUCION '}))
                response = connection.getresponse(); self.assertEqual(response.status, 200); response.read()
                self.assertEqual(match.ia.modo_agente, 'neuroevolucion')
                connection.close()
            finally:
                server.shutdown(); thread.join(); server.server_close()
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
