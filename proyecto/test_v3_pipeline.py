import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import panel_evaluation as panel
from opponent_profile import OpponentProfile
from phase1_network import GENOME_SIZE, NetworkPolicy
from phase1_trainer import train, _evaluate
from partida_truco import PartidaTruco

class PipelineV3Tests(unittest.TestCase):
    def test_sidecar_game_and_existing_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); champion = root / 'v3.npy'
            NetworkPolicy(np.zeros(GENOME_SIZE), encoder='v3').save(champion)
            self.assertEqual(json.loads(champion.with_suffix('.encoder.json').read_text()), {'encoder':'v3'})
            self.assertEqual(NetworkPolicy.load(champion).encoder, 'v3')
            with self.assertRaisesRegex(ValueError, 'encoder'):
                NetworkPolicy.load(champion, encoder='v2')
            old = OpponentProfile(root / 'opponent_profile.json')
            old.observe([{'player':0, 'action':'fold', 'observation':{'pending':None}}], 0)
            old_bytes = (root / 'opponent_profile.json').read_bytes()
            game = PartidaTruco(historial_path=str(root / 'history.json'))
            with patch.object(game.ia, 'mode_available', return_value=True), patch.object(game.ia, 'set_modo_agente', return_value={'nombre':'test'}), patch('partida_truco.PHASE1_CHAMPION', 'v3.npy'):
                game.ia.champions_dir = root
                game.cambiar_modo_ia('neuroevolucion')
            self.assertEqual(game.network.encoder, 'v3')
            self.assertEqual(game.opponent_profile.counts, old.counts)
            self.assertEqual((root / 'opponent_profile.json').read_bytes(), old_bytes)
            self.assertEqual(game.opponent_profile_v3.counts['fold'], [0, 0])

    def test_cli_versions(self):
        for name in ('phase1_trainer.py', 'run_phase1.py', 'reference_evaluation.py', 'panel_evaluation.py'):
            result = subprocess.run([sys.executable, str(Path(__file__).with_name(name)), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn('v3', result.stdout)

    def test_resume_rejects_v2_checkpoint_for_v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'state.npz'
            np.savez(checkpoint, population=np.zeros((2, GENOME_SIZE)), reigning=np.zeros(GENOME_SIZE), hall=np.empty((0,GENOME_SIZE)), state=json.dumps({'encoder':'v2'}))
            with self.assertRaisesRegex(ValueError, 'Checkpoint encoder'):
                train(tmp, population_size=2, generations=1, pairs=1, workers=1, validation_pairs=1, resume=checkpoint, encoder='v3')

    def test_run_config_records_v3_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('phase1_trainer.league_schedule', side_effect=RuntimeError('stop before matches')):
                with self.assertRaisesRegex(RuntimeError, 'stop before matches'):
                    train(tmp, population_size=2, generations=1, pairs=1, workers=1, validation_pairs=1, encoder='v3')
            self.assertEqual(json.loads((Path(tmp)/'run_config.json').read_text())['encoder'], 'v3')

    def test_reserved_seeds_and_selected_seeds(self):
        for seed in (9001, 9002, 9101):
            with self.assertRaisesRegex(ValueError, 'reserved'):
                panel.evaluate('missing', 'missing', baseline_seed=seed, panel_seed=9202)
            with self.assertRaisesRegex(ValueError, 'reserved'):
                panel.evaluate('missing', 'missing', baseline_seed=9201, panel_seed=seed)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in (('a',1),('b',2),('c',3)):
                NetworkPolicy(np.full(GENOME_SIZE, value), encoder='v3' if name == 'a' else 'v2').save(root / (name+'.npy'))
            (root / 'panel.json').write_text(json.dumps([{'name':'c','file':'c.npy','sha256':panel.digest(root/'c.npy')}]))
            with patch.object(panel, '_measure', return_value={'win_rate':.5,'margin_of_error':0,'pairs':2}) as measure:
                report = panel.evaluate(root/'a.npy', root/'b.npy', panel_dir=root, baseline_pairs=2, panel_pairs=2, workers=1, baseline_seed=9201, panel_seed=9202)
                self.assertEqual(report['schedule_seeds'], {'baseline':9201, 'panel':9202})
                self.assertEqual([(c.args[3], c.args[5], c.args[6]) for c in measure.call_args_list], [(9201,'v3','v2'), (9202,'v3','v2'), (9202,'v2','v2')])
                panel.evaluate(root/'a.npy', root/'b.npy', panel_dir=root, baseline_pairs=2, panel_pairs=2, workers=1, allow_reused_seeds=True)

    def test_reference_configuration_dispatches_v3(self):
        import reference_evaluation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root/'champion_0001.npy', np.zeros(GENOME_SIZE))
            (root/'run_config.json').write_text(json.dumps({'encoder':'v3'}))
            with patch.object(reference_evaluation, '_evaluate_pair', return_value=.5) as pair:
                reference_evaluation.evaluate(root, pairs=1, workers=1)
                self.assertEqual({call.args[0][4] for call in pair.call_args_list}, {'v3'})

    def test_mixed_job_creates_encoder_specific_policies(self):
        captured = []
        def match(a, b, **kwargs):
            captured.append((a.encoder, b.encoder))
            return SimpleNamespace(winner=0)
        with patch('phase1_trainer.play_match', side_effect=match):
            _evaluate((np.zeros(GENOME_SIZE), np.zeros(GENOME_SIZE), 1, 15, (0,0), None, 1.0, 'v3', 'v2'))
        self.assertEqual(captured, [('v3','v2'), ('v2','v3')])
