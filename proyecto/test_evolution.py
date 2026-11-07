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

    def test_server_can_open_evolution_page_without_robot(self):
        from unittest.mock import patch
        from servidor_truco import main
        with patch('sys.argv', ['servidor_truco.py', '--sin-robot', '--abrir-evolucion', '--puerto=0']), \
             patch('webbrowser.open') as browser, \
             patch('servidor_truco.ServidorTruco.serve_forever'):
            main()
        browser.assert_called_once()
        self.assertTrue(browser.call_args.args[0].endswith('/evolucion'))
