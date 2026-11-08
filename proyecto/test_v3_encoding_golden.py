import hashlib
import unittest
from opponent_inference import IncrementalInference
from opponent_profile import OpponentProfile
from phase1_network import encode_observation
from symmetric_engine import play_match

class Recorder:
    def __init__(self, version):
        self.version = version
        self.profile = OpponentProfile()
        self.inference = IncrementalInference()
        self.digest = hashlib.sha256()
        self.count = 0
    def __call__(self, obs, actions):
        self.digest.update(encode_observation(obs, self.profile, self.inference, encoder=self.version).tobytes())
        self.count += 1
        if obs['pending']:
            return 'quiero'
        return next((a for a in actions if a.startswith('card_')), 'fold')
    def observe_hand(self, decisions, opponent):
        self.inference.reset()

class EncodingGoldenTests(unittest.TestCase):
    def test_stateless_and_incremental_matches_are_byte_identical_to_goldens(self):
        for seed, count, versions in GOLDENS:
            for version, expected in versions:
                with self.subTest(seed=seed, encoder=version):
                    recorder = Recorder(version)
                    result = play_match(recorder, recorder, seed=seed, target=3)
                    digest = hashlib.sha256()
                    for decision in result.decisions:
                        digest.update(encode_observation(decision['observation'], OpponentProfile(), encoder=version).tobytes())
                    self.assertEqual((len(result.decisions), digest.hexdigest(), recorder.digest.hexdigest()),
                                     (count, expected, expected))

GOLDENS = [
    (16, 20, [('v1', 'b8cb35063ae93e1b7918f0084f19e705d524d475e37ed711ed62e667bab99790'),
              ('v2', 'e8c300ef11fc27cf1f4f6e3a193dea72e00c1f38e0e7da704912c342782721cb'),
              ('v3', 'c04830e85571479b43920e549d6203badc148387f7fe28b3bc9277174d20b234')]),
    (3, 28, [('v1', '58763127e8367020bfc987839351d1a0e2fefb61b43722ab99473e94478bd118'),
             ('v2', 'd60bb51f2dc329294cfbef1beb4fa29ca0d2efcec59a3e50d54a24d538749fef'),
             ('v3', '16cf85edbaf13d0d5ab6279ec7fc06fcf90d68398d062ab161c396302cfd9538')]),
]
