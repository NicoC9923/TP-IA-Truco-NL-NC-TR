import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

from partida_truco import PartidaTruco
from servidor_truco import crear_servidor

REPO = Path(__file__).resolve().parent.parent


@contextmanager
def running(root, phase='phase2'):
    server = crear_servidor(PartidaTruco(historial_path=str(root / 'history.json')), puerto=0, sin_robot=True, evolution_root=root)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        base = f'http://127.0.0.1:{server.server_port}/evolucion/{phase}/'
        yield lambda name: json.load(urlopen(base + name))
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def put(root, where, name, text):
    folder = root / where / 'phase2'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding='utf8')


class Phase2RoutesTest(unittest.TestCase):
    def test_empty_states_are_clear_json_not_404(self):
        with tempfile.TemporaryDirectory() as tmp, running(Path(tmp)) as get:
            self.assertEqual(get('generations'), [])
            self.assertEqual(get('reference'), [])
            self.assertEqual(get('progress'), {'status': 'empty'})
            self.assertEqual(get('example-hand'), {'available': False, 'reason': 'no_champion'})

    def test_committed_copies_are_served_when_no_live_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root, 'champions', 'generations.jsonl', '{"generation": 1}\n')
            put(root, 'champions', 'reference_evaluation.jsonl', '{"opponent": "a"}\n')
            put(root, 'champions', 'progress.json', '{"status": "ended"}')
            with running(root) as get:
                self.assertEqual(get('generations'), [{'generation': 1}])
                self.assertEqual(get('reference'), [{'opponent': 'a'}])
                self.assertEqual(get('progress'), {'status': 'ended'})

    def test_live_run_wins_over_committed_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root, 'champions', 'generations.jsonl', '{"generation": 1}\n')
            put(root, 'champions', 'reference_evaluation.jsonl', '{"opponent": "old"}\n')
            put(root, 'champions', 'progress.json', '{"status": "ended"}')
            put(root, 'training_output', 'generations.jsonl', '{"generation": 5}\n{"generation":')
            put(root, 'training_output', 'progress.json', '{"status": "training"}')
            with running(root) as get:
                self.assertEqual(get('generations'), [{'generation': 5}])
                self.assertEqual(get('progress'), {'status': 'training'})
                self.assertEqual(get('reference'), [])

    def test_live_progress_alone_counts_as_live_and_bad_progress_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root, 'champions', 'progress.json', '{"status": "ended"}')
            put(root, 'training_output', 'progress.json', '{"status": "tr')
            with running(root) as get:
                self.assertEqual(get('progress'), {'status': 'empty'})

    def test_phase1_routes_still_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'champions' / 'phase1'
            folder.mkdir(parents=True)
            (folder / 'generations.jsonl').write_text('{"generation": 3}\n')
            with running(root, 'phase1') as get:
                self.assertEqual(get('generations'), [{'generation': 3}])
                self.assertEqual(get('reference'), [])


class ExampleHandTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest('torch not installed')

    def make_root(self, tmp):
        from phase2_torch import Phase2Network
        root = Path(tmp)
        (root / 'champions/phase2').mkdir(parents=True)
        (root / 'champions/phase1').mkdir(parents=True)
        Phase2Network(seed=1).save(root / 'champions/phase2/reigning_champion.pt')
        shutil.copy(REPO / 'champions/phase1/reigning_champion.npy', root / 'champions/phase1/reigning_champion.npy')
        return root

    def test_hand_is_deterministic_with_top3_percentages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_root(tmp)
            with running(root) as get:
                first, second = get('example-hand'), get('example-hand')
        self.assertEqual(first, second)
        self.assertTrue(first['available'])
        self.assertEqual(len(first['champion_hand']), 3)
        champion_steps = [s for s in first['steps'] if s['by'] == 'champion']
        self.assertTrue(champion_steps)
        for step in champion_steps:
            self.assertTrue(1 <= len(step['top']) <= 3)
            pcts = [t['pct'] for t in step['top']]
            self.assertEqual(pcts, sorted(pcts, reverse=True))
            self.assertTrue(all(0 <= p <= 100 for p in pcts))
        self.assertTrue(all('top' not in s for s in first['steps'] if s['by'] == 'rival'))

    def test_live_champion_wins_over_committed(self):
        from phase2_example_hand import find_champion
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_root(tmp)
            self.assertEqual(find_champion(root), root / 'champions/phase2/reigning_champion.pt')
            live = root / 'training_output/phase2'
            live.mkdir(parents=True)
            (live / 'reigning_champion.pt').write_bytes(b'x')
            self.assertEqual(find_champion(root), live / 'reigning_champion.pt')


class NoTorchTest(unittest.TestCase):
    def test_server_and_example_hand_work_without_torch(self):
        script = """
import sys, tempfile, shutil, json
sys.modules['torch'] = None
from pathlib import Path
import servidor_truco, phase2_example_hand
root = Path(tempfile.mkdtemp())
(root / 'champions/phase2').mkdir(parents=True)
(root / 'champions/phase2/reigning_champion.pt').write_bytes(b'x')
(root / 'champions/phase1').mkdir(parents=True)
shutil.copy(sys.argv[1], root / 'champions/phase1/reigning_champion.npy')
print(json.dumps(phase2_example_hand.example_hand_payload(root)))
"""
        result = subprocess.run([sys.executable, '-c', script, str(REPO / 'champions/phase1/reigning_champion.npy')],
                                cwd=os.path.dirname(__file__), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'available': False, 'reason': 'torch'})


if __name__ == '__main__':
    unittest.main()
