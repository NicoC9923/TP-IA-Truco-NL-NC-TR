import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from phase2_torch import Phase2Network
from phase2_trainer import train

class SelectionIntegrationTest(unittest.TestCase):
    def test_check_snapshot_export_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opponent = root / 'opponent.pt'
            Phase2Network(seed=9).save(opponent)
            panel = root / 'panel.json'
            panel.write_text(json.dumps({'version': 1, 'members': [{'name': 'standin', 'path': 'opponent.pt', 'kind': 'phase2', 'encoder': 'v1', 'sha256': hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
            with patch('phase2_trainer.promotion_decision', return_value={'gain': .1, 'lower_bound': .05, 'upper_bound': .15, 'head_to_head': .5, 'promoted': True}):
                train(root / 'run', generations=2, workers=2, matches_per_generation=1, matches_per_round=1, target=1, panel_path=panel, champions_dir=root/'champions', panel_pairs=2, head_to_head_pairs=2)
            rows = [json.loads(line) for line in (root/'run/generations.jsonl').read_text().splitlines()]
            self.assertTrue(rows[1]['promoted'])
            self.assertEqual(rows[1]['check']['panel_version'], 1)
            self.assertIn('challenger', rows[1]['check']['members']['standin'])
            self.assertTrue((root/'run/generation_000002.pt').exists())
            self.assertTrue((root/'champions/reigning_champion.pt').exists())
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in torch.load(root/'champions/reigning_champion.pt', weights_only=True).values()))

    def test_excluded_modules_not_imported(self):
        import subprocess, sys
        script = "import sys; import phase2_trainer; assert 'truco_ia' not in sys.modules; assert 'engine_policies' not in sys.modules"
        subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).parent, check=True)


class CheckProtocolTest(unittest.TestCase):
    def test_phase1_policy_is_native_and_mirror_seats_are_credited(self):
        import numpy as np
        from types import SimpleNamespace
        from phase1_network import GENOME_SIZE, NetworkPolicy
        from phase2_trainer import _check_pair
        weights = np.zeros(GENOME_SIZE)
        spec = {'kind': 'phase1', 'weights': weights, 'encoder': 'v1'}
        seen = []
        def play(a, b, **kwargs):
            seen.append((a, b))
            return SimpleNamespace(winner=0), SimpleNamespace(winner=1)
        with patch('phase2_trainer.play_mirror_pair', side_effect=play):
            self.assertEqual(_check_pair((spec, spec, 3, 1, 8, .1)), 1)
        self.assertEqual(seen[0][1].__dict__.keys(), NetworkPolicy(weights, seed=9).__dict__.keys())
        self.assertFalse(hasattr(seen[0][1], 'temperature'))
        from phase1_network import INPUTS
        with patch('phase1_network.encode_observation', return_value=np.zeros(INPUTS)):
            plain = NetworkPolicy(weights, seed=9, encoder='v1')
            self.assertEqual([seen[0][1]({}, ['quiero', 'no_quiero']) for _ in range(100)], [plain({}, ['quiero', 'no_quiero']) for _ in range(100)])
        with patch('phase2_trainer.play_mirror_pair', return_value=(SimpleNamespace(winner=1), SimpleNamespace(winner=0))):
            self.assertEqual(_check_pair((spec, spec, 3, 1, 8, .1)), 0)

    def test_patience_and_time_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opponent = root/'opponent.pt'; Phase2Network(seed=9).save(opponent)
            panel = root/'panel.json'
            panel.write_text(json.dumps({'version': 1, 'members': [{'name':'x','path':'opponent.pt','kind':'phase2','encoder':'v1','sha256':hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
            config = dict(generations=4, matches_per_generation=1, matches_per_round=1, target=1, panel_path=panel, champions_dir=root/'champ', panel_pairs=2, head_to_head_pairs=2)
            with patch('phase2_trainer.promotion_decision', return_value={'gain': 0, 'lower_bound': 0, 'upper_bound': 0, 'head_to_head': .5, 'promoted': False}):
                train(root/'patience', patience=1, **config)
            rows = [json.loads(x) for x in (root/'patience/generations.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['stop_reason'], 'patience')
            self.assertEqual(len(rows), 2)
            counter = iter(range(10000))
            train(root/'time', clock=lambda: next(counter), time_cap_seconds=1, **config)
            rows = [json.loads(x) for x in (root/'time/generations.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['stop_reason'], 'time_cap')

    def test_panel_boundary_and_resume(self):
        import numpy as np
        from phase1_network import GENOME_SIZE
        def fake_check(model, champion, panel, *args):
            marker = float(next(iter(model.state_dict().values())).sum())
            names = [m['name'] for m in panel['members']]
            members = {name: {'challenger': [(.5 + (marker % .1)) if i % 2 else .5]*2, 'champion': [.8]*2} for i, name in enumerate(names)}
            return {'gain': marker, 'lower_bound': -1, 'upper_bound': 1, 'head_to_head': .5, 'promoted': False, 'members': members, 'direct': [.5]*2, 'panel_version': panel['version'], 'schedule_seed': args[0]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = []
            for i in range(4):
                path = root / f'm{i}.npy'
                np.save(path, np.zeros(GENOME_SIZE) + i)
                members.append({'name':f'm{i}', 'path':path.name, 'kind':'phase1', 'encoder':'v1', 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            panel = root/'panel.json'; panel.write_text(json.dumps({'version': 1, 'members':members}))
            config = dict(generations=4, matches_per_generation=1, matches_per_round=1, target=1, panel_path=panel, panel_pairs=2, head_to_head_pairs=2, saturation_checks=1, saturation_pairs=2, clock=lambda: 0, champions_dir=root/'champ')
            with patch('phase2_trainer._selection_check', side_effect=fake_check), patch('phase2_trainer._champion_overlap', return_value={}):
                train(root/'full', **config)
                train(root/'resumed', stop_after_round=3, **config)
                train(root/'resumed', resume=root/'resumed/checkpoint_000003.pt', **config)
            def read(folder):
                return [json.loads(x) for x in (root/folder/'generations.jsonl').read_text().splitlines()]
            full, resumed = read('full'), read('resumed')
            self.assertEqual(full, resumed)
            self.assertEqual(full[-1]['panel_boundary']['to'], 2)
            self.assertEqual(full[-1]['check']['panel_version'], 2)
            self.assertTrue((root/'resumed/panel-v2.json').exists())
            from phase2_promotion import load_panel
            self.assertEqual(load_panel(root/'resumed/panel-v2.json')['version'], 2)

    def test_real_check_matches_across_workers_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opponent = root/'opponent.pt'; Phase2Network(seed=9).save(opponent)
            panel = root/'panel.json'
            panel.write_text(json.dumps({'version': 0, 'measuring_only': True, 'members': [{'name':'x','path':'opponent.pt','kind':'phase2','encoder':'v1','sha256':hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
            config = dict(generations=2, matches_per_generation=1, matches_per_round=1, target=1, panel_path=panel, champions_dir=root/'champ', panel_pairs=2, head_to_head_pairs=2, clock=lambda: 0)
            train(root/'one', workers=1, **config)
            train(root/'two', workers=2, stop_after_round=1, **config)
            train(root/'two', workers=2, resume=root/'two/checkpoint_000001.pt', **config)
            def read(name): return [json.loads(x) for x in (root/name/'generations.jsonl').read_text().splitlines()]
            self.assertEqual(read('one'), read('two'))
            self.assertEqual(read('one')[-1]['check']['panel_version'], 0)

    def test_time_cap_at_round_boundary_does_not_start_a_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opponent = root/'opponent.pt'; Phase2Network(seed=9).save(opponent)
            panel = root/'panel.json'
            panel.write_text(json.dumps({'version': 0, 'members': [{'name':'x', 'path':'opponent.pt', 'kind':'phase2', 'encoder':'v1', 'sha256':hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
            ticks = iter([0, 0, 0, 0, 0, 10, 10, 10, 10])
            with patch('phase2_trainer._selection_check', side_effect=AssertionError('check started after cap')):
                train(root/'run', generations=2, matches_per_generation=1, matches_per_round=1, target=1, panel_path=panel, champions_dir=root/'champ', panel_pairs=2, head_to_head_pairs=2, time_cap_seconds=5, clock=lambda: next(ticks))
            entries = [json.loads(x) for x in (root/'run/generations.jsonl').read_text().splitlines()]
            self.assertEqual(entries[-1]['stop_reason'], 'time_cap')
            self.assertEqual(entries[-1]['generation'], 2)

    def test_stopped_run_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); opponent=root/'x.pt'; Phase2Network(seed=9).save(opponent)
            panel=root/'panel.json'; panel.write_text(json.dumps({'version':1, 'members':[{'name':'x','path':'x.pt','kind':'phase2','encoder':'v1','sha256':hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
            config=dict(generations=4, matches_per_generation=1, matches_per_round=1, target=1, panel_path=panel, champions_dir=root/'champ', panel_pairs=2, head_to_head_pairs=2)
            for reason in ('patience', 'time_cap'):
                options={'patience':1} if reason=='patience' else {'time_cap_seconds':1, 'clock':lambda: 100}
                if reason == 'time_cap':
                    ticks=iter(range(10000)); options['clock']=lambda: next(ticks)
                with patch('phase2_trainer.promotion_decision', return_value={'gain':0,'lower_bound':0,'upper_bound':0,'head_to_head':.5,'promoted':False}):
                    train(root/reason, **config, **options)
                    log=root/reason/'generations.jsonl'; before=log.read_bytes()
                    checkpoint=sorted((root/reason).glob('checkpoint_*.pt'))[-1]
                    state=torch.load(checkpoint, weights_only=False)
                    self.assertTrue(state['phase_ended']); self.assertEqual(state['stop_reason'],reason)
                    with patch('phase2_trainer._matches', side_effect=AssertionError('trained after stop')):
                        train(root/reason, resume=checkpoint, **config, **options)
                    self.assertEqual(log.read_bytes(),before)

    def test_real_panel_change_workers_and_resume(self):
        import numpy as np
        from phase1_network import GENOME_SIZE
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); members=[]
            for i in range(4):
                path=root/f'm{i}.npy'; np.save(path, np.random.default_rng(i).normal(0, .1, GENOME_SIZE))
                members.append({'name':f'm{i}','path':path.name,'kind':'phase1','encoder':'v1','sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            panel=root/'panel.json'; panel.write_text(json.dumps({'version':1,'members':members}))
            cfg=dict(generations=6, matches_per_generation=1, matches_per_round=1, target=1, panel_path=panel, champions_dir=root/'champ', panel_pairs=2, head_to_head_pairs=2, min_gain=2, saturation_checks=1, saturation_pairs=2, saturation_threshold=0, clock=lambda:0)
            train(root/'one', workers=1, **cfg)
            train(root/'two', workers=2, **cfg)
            train(root/'resume', workers=1, stop_after_round=3, **cfg)
            train(root/'resume', workers=2, resume=root/'resume/checkpoint_000003.pt', **cfg)
            def records(name): return [json.loads(x) for x in (root/name/'generations.jsonl').read_text().splitlines()]
            self.assertEqual(records('one'),records('two')); self.assertEqual(records('one'),records('resume'))
            self.assertTrue(any('panel_boundary' in x for x in records('one')))
            states=[torch.load(root/name/'checkpoint_000006.pt',weights_only=False)['model'] for name in ('one','two','resume')]
            for key in states[0]:
                self.assertTrue(torch.equal(states[0][key],states[1][key])); self.assertTrue(torch.equal(states[0][key],states[2][key]))
            checkpoints=[torch.load(root/name/'checkpoint_000006.pt',weights_only=False) for name in ('one','two','resume')]
            for key, values in checkpoints[0]['optimizer']['state'].items():
                for name, value in values.items():
                    for state in checkpoints[1:]:
                        self.assertTrue(torch.equal(value, state['optimizer']['state'][key][name]))
            import shutil
            shutil.copytree(root/'one', root/'moved')
            train(root/'moved', workers=1, resume=root/'moved/checkpoint_000006.pt', **{**cfg, 'generations':7})
            moved_state = torch.load(root/'moved/checkpoint_000006.pt', weights_only=False)
            member = next(m for m in moved_state['panel']['members'] if m['kind'] == 'phase2')
            (root/'moved'/member['path']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                train(root/'moved', workers=1, resume=root/'moved/checkpoint_000006.pt', **{**cfg, 'generations':7})

    def test_rotating_seeds_and_resume_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); opponent=root/'x.pt'; Phase2Network(seed=9).save(opponent)
            panel=root/'panel.json'; panel.write_text(json.dumps({'version':0,'measuring_only':True,'members':[{'name':'x','path':'x.pt','kind':'phase2','encoder':'v1','sha256':hashlib.sha256(opponent.read_bytes()).hexdigest()}]}))
            cfg=dict(generations=3,matches_per_generation=1,matches_per_round=1,target=1,panel_path=panel,champions_dir=root/'champ',panel_pairs=2,head_to_head_pairs=2,clock=lambda:0)
            with self.assertRaisesRegex(ValueError,'reserved'):
                train(root/'bad', **{**cfg,'check_seed_base':9300})
            train(root/'run', **cfg)
            rows=[json.loads(x) for x in (root/'run/generations.jsonl').read_text().splitlines()]
            self.assertEqual([r['check']['schedule_seed'] for r in rows[1:]],[10000,10001])
            self.assertTrue(rows[-1]['measuring_only']); self.assertNotIn('promotion_check',rows[-1])
            state=torch.load(root/'run/checkpoint_000002.pt',weights_only=False)
            self.assertNotIn('resolved_path',state['panel']['members'][0])
            opponent.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                train(root/'run', resume=root/'run/checkpoint_000002.pt', **cfg)

    def test_check_uses_one_pool_submission(self):
        import numpy as np
        from phase1_network import GENOME_SIZE
        from phase2_trainer import _selection_check
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'opponent.npy'; np.save(path,np.zeros(GENOME_SIZE))
            panel={'version':1,'members':[{'name':'x','resolved_path':str(path),'kind':'phase1','encoder':'v1'}]}
            class Pool:
                _processes=2
                def __init__(self): self.calls=0
                def map(self, function, jobs, chunksize):
                    self.calls+=1
                    return [[.5]*len(job[2]) for job in jobs]
            pool=Pool(); model=Phase2Network()
            _selection_check(model, model.state_dict(), panel,10000,2,2,1,.1,pool,.01,.475)
            self.assertEqual(pool.calls,1)
