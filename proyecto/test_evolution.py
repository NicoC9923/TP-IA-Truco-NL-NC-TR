import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen

from partida_truco import PartidaTruco
from servidor_truco import crear_servidor


class EvolutionServerTests(unittest.TestCase):
    def test_page_and_live_logs_with_committed_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            committed = root / 'champions' / 'phase1'
            committed.mkdir(parents=True)
            (committed / 'generations.jsonl').write_text('{"generation": 1}\n', encoding='utf8')
            (committed / 'reference_evaluation.jsonl').write_text('{"generation": 1}\n', encoding='utf8')
            server = crear_servidor(PartidaTruco(None, historial_path=str(root / 'history.json')), puerto=0, sin_robot=True, evolution_root=root)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                base = f'http://127.0.0.1:{server.server_port}'
                with urlopen(base + '/evolucion') as response:
                    self.assertIn('Evolución', response.read().decode())
                with urlopen(base + '/evolucion/phase1/generations') as response:
                    self.assertEqual(json.load(response), [{'generation': 1}])
                live = root / 'training_output' / 'phase1'
                live.mkdir(parents=True)
                (live / 'generations.jsonl').write_text('{"generation": 2}\n{"generation":', encoding='utf8')
                with urlopen(base + '/evolucion/phase1/generations') as response:
                    self.assertEqual(json.load(response), [{'generation': 2}])
                with urlopen(base + '/evolucion/phase1/reference') as response:
                    self.assertEqual(json.load(response), [])
                reference = live / 'reference_evaluation.jsonl'
                reference.write_text('{"generation": 2, "win_rate": 0.6}\n', encoding='utf8')
                with urlopen(base + '/evolucion/phase1/reference') as response:
                    self.assertEqual(json.load(response)[0]['win_rate'], 0.6)
                with reference.open('a', encoding='utf8') as stream:
                    stream.write('{"generation": 3, "win_rate": 0.7}\n')
                with urlopen(base + '/evolucion/phase1/reference') as response:
                    self.assertEqual(len(json.load(response)), 2)
                with urlopen(base + '/truco/estado') as response:
                    self.assertIsInstance(json.load(response), dict)
            finally:
                server.shutdown()
                thread.join()
                server.server_close()

    def test_missing_logs_return_empty_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = crear_servidor(PartidaTruco(historial_path=str(root / 'history.json')), puerto=0, sin_robot=True, evolution_root=root)
            thread = threading.Thread(target=server.serve_forever); thread.start()
            try:
                for name in ('generations', 'reference'):
                    with urlopen(f'http://127.0.0.1:{server.server_port}/evolucion/phase1/{name}') as response:
                        self.assertEqual(json.load(response), [])
            finally:
                server.shutdown(); thread.join(); server.server_close()

    def test_existing_http_contract(self):
        import http.client
        import random
        with tempfile.TemporaryDirectory() as tmp:
            match = PartidaTruco(historial_path=str(Path(tmp) / 'history.json'), rng=random.Random(7))
            server = crear_servidor(match, puerto=0, sin_robot=True)
            thread = threading.Thread(target=server.serve_forever); thread.start()
            try:
                conn = http.client.HTTPConnection('127.0.0.1', server.server_port)
                expected = {
                    '/truco/estado': set('aprendizaje_ia apuesta_truco bluffs_mano_actual canto_pendiente cartas_humano cartas_humano_disponibles cartas_robot_restantes cartas_robot_reveladas envido_disponible envido_resuelto es_mano estadisticas_partida ganador_mano ganador_partido ganadores_ronda log mano_terminada mesa partido_terminado por_el_partido puntos_truco_actuales resultado_envido ronda siguiente_truco_humano tanteador tantos_humano tantos_robot turno_carta ultima_frase_robot robot_conectado'.split()),
                    '/truco/historial': set('adaptacion_activa arquitectura_nombre arquitectura_subtitulo bluffs_humano_detectados manos_totales modo_agente modos_disponibles partidas partidas_jugadas perfil_humano_detectado perfil_rival prob_farol_envido_pct prob_farol_truco_pct tasa_bluff_humano_pct tasa_fold_envido_humano_pct tasa_fold_truco_humano_pct traza_razonamiento ultimo_razonamiento umbral_envido_offset umbral_truco_offset victorias_humano victorias_robot'.split()),
                }
                for route, keys in expected.items():
                    conn.request('GET', route); response = conn.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(set(json.loads(response.read())), keys)
                posts = [('/comando', {}, 400), ('/truco/detectar', {}, 200),
                         ('/truco/jugar', {}, 400), ('/truco/envido', {}, 200),
                         ('/truco/truco', {}, 200), ('/truco/responder', {}, 200),
                         ('/truco/mazo', {}, 200), ('/truco/voz', {}, 200),
                         ('/truco/nuevo', {}, 200), ('/truco/reiniciar', {}, 200),
                         ('/truco/modo_ia', {}, 200)]
                for route, body, status in posts:
                    with self.subTest(route=route):
                        conn.request('POST', route, json.dumps(body)); response = conn.getresponse()
                        self.assertEqual(response.status, status); response.read()
                conn.close()
            finally:
                server.shutdown(); thread.join(); server.server_close()

    def test_server_can_open_evolution_page_without_robot(self):
        from unittest.mock import patch
        from servidor_truco import main
        with patch('sys.argv', ['servidor_truco.py', '--sin-robot', '--abrir-evolucion', '--puerto=0']), \
             patch('webbrowser.open') as browser, \
             patch('servidor_truco.ServidorTruco.serve_forever'):
            main()
        browser.assert_called_once()
        self.assertTrue(browser.call_args.args[0].endswith('/evolucion'))


class EvolutionPageContentTests(unittest.TestCase):
    page = (Path(__file__).resolve().parent / 'evolucion.html').read_text(encoding='utf8')

    def test_has_three_tabs_and_required_texts(self):
        for text in ('Fase 1 · Neuroevolución', 'Fase 2 · Aprendizaje por refuerzo', 'Comparación',
                     'Neuroevolución vs Aprendizaje por refuerzo', 'Ver una mano de ejemplo de la campeona',
                     'Disponible cuando la fase 2 tenga campeona evaluada',
                     'Mejora sobre la campeona contra el panel fijo', 'Resultado del control contra la campeona',
                     'la fase 2 entrenó también contra redes de la fase 1'):
            self.assertIn(text, self.page)

    def test_polls_every_five_seconds_only_while_training(self):
        self.assertIn('POLL_MS = 5000', self.page)
        self.assertIn("status === 'training'", self.page)

    def test_reads_the_phase_routes(self):
        for route in ('/evolucion/phase1/generations', '/evolucion/phase1/reference', '/evolucion/phase2/generations',
                      '/evolucion/phase2/reference', '/evolucion/phase2/progress', '/evolucion/phase2/example-hand'):
            self.assertIn(route, self.page)

    def test_old_page_panels_and_hidden_metrics_are_gone(self):
        for text in ('Aptitud de la población', 'Conducta emergente', 'Diversidad', 'Campeones reinantes',
                     'league_mix', 'behavior', 'diversity', 'Sofia'):
            self.assertNotIn(text, self.page)

    def test_phone_layout_rules_and_touch_targets(self):
        self.assertIn('name="viewport"', self.page)
        self.assertIn('@media(max-width:640px)', self.page)
        self.assertGreaterEqual(self.page.count('min-height:44px'), 1)

    def test_empty_states_and_keyboard_tabs(self):
        self.assertIn('Todavía no hay una ejecución de la fase 2.', self.page)
        self.assertIn('Disponible cuando la fase 2 tenga campeona evaluada', self.page)
        self.assertIn('ArrowRight', self.page)
        self.assertIn('tabIndex', self.page)

    def test_review_fixes_are_present(self):
        self.assertIn('Panel de medición (provisorio)', self.page)
        self.assertIn('/evolucion/phase1/source', self.page)
        self.assertIn('phase1Training()', self.page)
        self.assertIn('trainingNow()', self.page)
        self.assertNotIn('state.failed) pollTimer', self.page)
        self.assertNotIn('.reverse().find(', self.page)
        self.assertIn('No hay una mano de ejemplo registrada para la campeona vigente.', self.page)
