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

    def test_selected_runs_are_isolated_and_partial_files_safe(self):
        from urllib.error import HTTPError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root, 'champions', 'generations.jsonl', '{"generation": 28}\n')
            live = root / 'training_output/phase2-run2'
            live.mkdir(parents=True)
            (live / 'generations.jsonl').write_text('{"generation": 2}\n{"generation":')
            (live / 'progress.json').write_text('{"status":')
            with running(root) as get:
                self.assertEqual(get('runs')['default'], 'run2')
                self.assertEqual(get('generations?run=run1'), [{'generation': 28}])
                self.assertEqual(get('generations?run=run2'), [{'generation': 2}])
                self.assertEqual(get('progress?run=run2'), {'status': 'empty'})
                self.assertEqual(get('acceptance?run=run2'), {'status': 'pending'})
                self.assertEqual(get('example-hand?run=run2')['reason'], 'awaiting_stable_example')
                for value in ('../phase1', '', 'run1&run=run2'):
                    with self.assertRaises(HTTPError):
                        get('progress?run=' + value)
                acceptance = root / 'experiments/phase2-run2/acceptance'
                acceptance.mkdir(parents=True)
                (acceptance / 'results.json').write_text('{"decision":"ships","direct":{},"held_out":{}}')
                self.assertEqual(get('acceptance?run=run2')['decision'], 'ships')
                self.assertEqual(get('acceptance?run=run1')['status'], 'pending')

    def test_phase1_routes_still_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'champions' / 'phase1'
            folder.mkdir(parents=True)
            (folder / 'generations.jsonl').write_text('{"generation": 3}\n')
            with running(root, 'phase1') as get:
                self.assertEqual(get('generations'), [{'generation': 3}])
                self.assertEqual(get('reference'), [])


class Phase1SourceTest(unittest.TestCase):
    def test_source_tells_live_committed_or_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with running(root, 'phase1') as get:
                self.assertEqual(get('source'), {'source': 'empty'})
                folder = root / 'champions' / 'phase1'
                folder.mkdir(parents=True)
                (folder / 'generations.jsonl').write_text('{"generation": 1}')
                self.assertEqual(get('source'), {'source': 'committed'})
                live = root / 'training_output' / 'phase1'
                live.mkdir(parents=True)
                (live / 'generations.jsonl').write_text('{"generation": 2}')
                self.assertEqual(get('source'), {'source': 'live'})


class ExampleHandTest(unittest.TestCase):
    def test_invalid_sidecar_temperature_is_unavailable(self):
        import json
        from phase2_example_hand import example_hand_payload
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_root(tmp)
            folder = root / 'champions/phase2'
            (folder / 'progress.json').write_text('{"temperature": 0.4}')
            for value in (0, -1, float('nan'), float('inf'), None, '0.1'):
                with self.subTest(value=value):
                    (folder / 'reigning_champion.json').write_text(json.dumps({'temperature': value}))
                    self.assertEqual(example_hand_payload(root)['reason'], 'invalid_champion_metadata')

    def test_auxiliary_champion_is_playable(self):
        from phase2_example_hand import example_hand_payload
        from phase2_torch import Phase2Network
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_root(tmp)
            Phase2Network(seed=3, auxiliary=True).save(root / 'champions/phase2/reigning_champion.pt')
            self.assertTrue(example_hand_payload(root)['available'])

    def test_progress_prior_fallback_and_missing_sidecar_guard(self):
        from unittest.mock import patch
        from phase2_example_hand import example_hand_payload
        from phase2_torch import Phase2Policy
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_root(tmp)
            folder = root / 'champions/phase2'
            sidecar = folder / 'reigning_champion.json'
            progress = folder / 'progress.json'
            sidecar.write_text('{"temperature": 0.3}')
            progress.write_text('{"temperature": 0.4, "profile_prior": 5}')
            with patch('phase2_torch.Phase2Policy', wraps=Phase2Policy) as policy:
                self.assertTrue(example_hand_payload(root)['available'])
                self.assertEqual(policy.call_args.kwargs['profile_prior'], 5)
                self.assertEqual(policy.call_args.kwargs['temperature'], .3)
                progress.write_text('{"temperature": 0.4, "profile_prior": 7}')
                self.assertTrue(example_hand_payload(root)['available'])
                self.assertEqual(policy.call_args.kwargs['profile_prior'], 7)
                sidecar.write_text('{"profile_prior": 5}')
                self.assertTrue(example_hand_payload(root)['available'])
                self.assertEqual(policy.call_args.kwargs['profile_prior'], 5)
                sidecar.unlink()
                self.assertEqual(example_hand_payload(root)['reason'], 'invalid_champion_metadata')
                progress.write_text('{"temperature": 0.4, "profile_prior": 20}')
                self.assertTrue(example_hand_payload(root)['available'])
                self.assertEqual(policy.call_args.kwargs['profile_prior'], 20)

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

    def test_trace_matches_synthetic_model_forward_and_policy_mask(self):
        import numpy as np
        import torch
        from phase1_network import ACTIONS
        from phase2_example_hand import decision_trace
        from phase2_torch import Phase2Network
        model = Phase2Network(seed=19)
        inputs = np.linspace(-1, 1, 1628, dtype=np.float32)
        probabilities = {ACTIONS[0]: 0.25, ACTIONS[3]: 0.75}
        trace = decision_trace(model, inputs, probabilities, ACTIONS[3])
        with torch.no_grad():
            x = torch.from_numpy(inputs)
            first = model.layers[:2](x)
            second = model.layers[2:4](first)
            values = model(x)
        np.testing.assert_allclose(trace['input'], inputs)
        np.testing.assert_allclose(trace['hidden_256'], first.numpy())
        for layer_index, (layer, source, output) in enumerate(((model.layers[0], x, first), (model.layers[2], first, second))):
            for index, calculation in enumerate(trace['calculations'][layer_index]):
                expected = float(((layer.weight[index] * source).sum() + layer.bias[index]).detach())
                self.assertAlmostEqual(sum(calculation['groups']) + calculation['bias'], expected, places=5)
                self.assertAlmostEqual(calculation['sum'], expected, places=5)
                self.assertAlmostEqual(max(0, expected), float(output[index]), places=5)
        self.assertEqual(len(trace['calculations'][0][0]['groups']), 6)
        from phase2_network import INDEX
        spans = (slice(0, 70), INDEX['own_cards'], INDEX['legal_actions'],
                 INDEX['events'], INDEX['exact_belief'], INDEX['weighted_belief'])
        for index, calculation in enumerate(trace['calculations'][0]):
            for span, contribution in zip(spans, calculation['groups']):
                expected_group = float((model.layers[0].weight[index, span] * x[span]).sum().detach())
                self.assertAlmostEqual(contribution, expected_group, places=5)
        np.testing.assert_allclose(trace['hidden_128'], second.numpy())
        np.testing.assert_allclose([item['value'] for item in trace['actions']], values.numpy())
        self.assertEqual([len(trace[k]) for k in ('input', 'hidden_256', 'hidden_128', 'actions')], [1628, 256, 128, 12])
        self.assertEqual(trace['chosen'], ACTIONS[3])
        self.assertEqual([a['action'] for a in trace['actions'] if a['legal']], [ACTIONS[0], ACTIONS[3]])
        self.assertTrue(all(a['probability'] is None for a in trace['actions'] if not a['legal']))

    def test_sample_traces_belong_to_recorded_decisions(self):
        from phase2_example_hand import example_hand_payload
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_root(tmp)
            (root / 'champions/phase2/reigning_champion.json').write_text('{"generation": 9, "temperature": 0.05, "profile_prior": 5}')
            result = example_hand_payload(root, directory=root / 'champions/phase2')
            self.assertEqual(result['generation'], 9)
            self.assertEqual(len(result['model_sha256']), 64)
            self.assertNotIn('opponent_hand', result)
            import numpy as np
            import torch
            from phase2_torch import Phase2Network
            model = Phase2Network.load(root / 'champions/phase2/reigning_champion.pt')
            for step in result['steps']:
                if step['by'] == 'champion':
                    trace = step['trace']
                    with torch.no_grad():
                        x = torch.tensor(trace['input'], dtype=torch.float32)
                        np.testing.assert_allclose(trace['hidden_256'], model.layers[:2](x).numpy(), atol=1e-6)
                        np.testing.assert_allclose(trace['hidden_128'], model.layers[:4](x).numpy(), atol=1e-6)
                        np.testing.assert_allclose([a['value'] for a in trace['actions']], model(x).numpy(), atol=1e-6)
                    legal = [a for a in trace['actions'] if a['legal']]
                    scores = np.array([a['value'] for a in legal]) / result['temperature']
                    probs = np.exp(scores - scores.max()); probs /= probs.sum()
                    np.testing.assert_allclose([a['probability'] for a in legal], probs, atol=1e-7)
                    self.assertEqual(trace['chosen'], step['action'])
                    self.assertEqual(len(trace['input']), 1628)
                    self.assertAlmostEqual(sum(a['probability'] for a in trace['actions'] if a['legal']), 1)
                    self.assertTrue(next(a['legal'] for a in trace['actions'] if a['action'] == trace['chosen']))

    def test_temperature_comes_from_the_same_run_as_the_champion(self):
        from phase2_example_hand import find_run
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_root(tmp)
            committed, live = root / 'champions/phase2', root / 'training_output/phase2'
            self.assertEqual(find_run(root)[1], 0.1)
            (committed / 'progress.json').write_text('{"temperature": 0.4}')
            self.assertEqual(find_run(root)[1], 0.4)
            (committed / 'reigning_champion.json').write_text('{"temperature": 0.3}')
            self.assertEqual(find_run(root)[1], 0.3)
            # a live progress file without a live champion must not leak into the committed champion
            put(root, 'training_output', 'progress.json', '{"temperature": 0.9}')
            self.assertEqual(find_run(root), (committed / 'reigning_champion.pt', 0.3))
            # a live champion with no recorded temperature does not borrow the committed one
            (live / 'reigning_champion.pt').write_bytes(b'x')
            self.assertEqual(find_run(root), (live / 'reigning_champion.pt', 0.9))
            (live / 'progress.json').unlink()
            self.assertEqual(find_run(root), (live / 'reigning_champion.pt', 0.1))
            (live / 'reigning_champion.json').write_text('{"temperature": -2}')
            self.assertEqual(find_run(root)[1], 0.1)

    def test_payload_reports_temperature_and_is_cached_by_identity(self):
        from unittest.mock import patch
        import symmetric_engine
        from phase2_example_hand import example_hand_payload
        real = symmetric_engine.play_match
        calls = []
        def counting(*a, **k):
            calls.append(1)
            return real(*a, **k)
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_root(tmp)
            (root / 'champions/phase2/reigning_champion.json').write_text('{"temperature": 0.3}')
            with patch('symmetric_engine.play_match', counting):
                first = example_hand_payload(root)
                second = example_hand_payload(root)
                self.assertEqual((first, len(calls)), (second, 1))
                self.assertEqual(first['temperature'], 0.3)
                (root / 'champions/phase2/reigning_champion.json').write_text('{"temperature": 0.5}')
                self.assertEqual(example_hand_payload(root)['temperature'], 0.5)
                self.assertEqual(len(calls), 2)
                (root / 'champions/phase2/reigning_champion.pt').write_bytes(b'broken')
                self.assertEqual(example_hand_payload(root), {'available': False, 'reason': 'unreadable_champion'})

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
