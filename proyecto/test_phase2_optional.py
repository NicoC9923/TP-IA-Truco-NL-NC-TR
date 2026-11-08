import os
import subprocess
import sys
import unittest

class OptionalTorchTests(unittest.TestCase):
    def test_phase1_plays_without_torch(self):
        script = """
import sys
sys.modules['torch'] = None
import numpy as np
import symmetric_engine, phase1_network, phase1_trainer, phase2_network, partida_truco
obs = {'player': 0, 'hand': ((1, 'Espada'), (2, 'Oro'), (3, 'Copa')), 'available': (0, 1, 2), 'score': (0, 0), 'target': 15, 'mano': 0, 'table': (), 'tricks': (), 'truco': 1, 'pending': None, 'history': (), 'envido_done': False, 'envido_reveal': None, 'raise_owner': None}
p = phase1_network.NetworkPolicy(np.zeros(phase1_network.GENOME_SIZE))
assert p(obs, ('card_0', 'card_1')) in ('card_0', 'card_1')
"""
        result = subprocess.run([sys.executable, '-c', script], cwd=os.path.dirname(__file__), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_policy_tests_skip_when_torch_is_unavailable(self):
        script = "import sys, unittest; sys.modules['torch']=None; import test_phase2_policy; r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(test_phase2_policy)); assert r.wasSuccessful() and len(r.skipped) == r.testsRun and r.testsRun > 0"
        result = subprocess.run([sys.executable, '-c', script], cwd=os.path.dirname(__file__), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
