import tempfile
import unittest
from pathlib import Path

from phase2_torch import Phase2Network
from phase2_reference_evaluation import paired_interval, evaluate
from phase2_loader import load_phase2_policy

class ReferenceEvaluationTests(unittest.TestCase):
    def test_paired_interval(self):
        result = paired_interval([0, 1, 0.5, 0.5])
        self.assertEqual(result['mean'], 0.5)
        self.assertAlmostEqual(result['margin'], 1.96 * (1 / 24) ** 0.5)

    def test_metadata_required(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'weights.pt'
            Phase2Network().save(path)
            with self.assertRaises(ValueError):
                load_phase2_policy(path, seed=1)
            policy = load_phase2_policy(path, seed=1, metadata={'temperature': .2, 'profile_prior': 7})
            self.assertEqual((policy.temperature, policy.profile.prior_strength), (.2, 7))

    def test_reserved_seed(self):
        with self.assertRaises(ValueError):
            evaluate({}, pairs=2, schedule_seed=777, target=None, workers=1, metadata={})

    def test_sidecar_and_explicit_metadata(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'weights.pt'
            Phase2Network(auxiliary=True).save(path)
            (Path(d) / 'reigning_champion.json').write_text(json.dumps({'temperature': .3, 'profile_prior': 9}))
            policy = load_phase2_policy(path, seed=1)
            self.assertEqual((policy.temperature, policy.profile.prior_strength), (.3, 9))
            self.assertTrue(hasattr(policy.model, 'card_head'))
            policy = load_phase2_policy(path, seed=1, metadata={'temperature': .4, 'profile_prior': 11})
            self.assertEqual((policy.temperature, policy.profile.prior_strength), (.4, 11))

    def test_neutral_profile_when_adaptation_off(self):
        from unittest.mock import patch
        from phase2_torch import Phase2Policy
        policy = Phase2Policy(adapt=False)
        with patch('phase2_torch.encode_observation', return_value=None) as encode:
            policy.encode({}, ())
            self.assertIs(encode.call_args.args[1], policy.neutral_profile)
            self.assertIsNot(encode.call_args.args[1], policy.profile)

    def test_smoke_determinism_and_validation_partition(self):
        from unittest.mock import patch
        from phase2_reference_evaluation import _chunk
        from symmetric_engine import MatchResult
        with tempfile.TemporaryDirectory() as d:
            paths = {name: str(Path(d) / (name + '.pt')) for name in ('a', 'b')}
            for seed, path in enumerate(paths.values()):
                Phase2Network(seed=seed).save(path)
            metadata = {'temperature': .1, 'profile_prior': 20}
            kwargs = dict(pairs=2, schedule_seed=12345, target=15, metadata=metadata)
            first = evaluate(paths, workers=1, **kwargs)
            second = evaluate(paths, workers=2, **kwargs)
            self.assertEqual(first, second)
            self.assertEqual(len(first['scores']['a']['reactive']['on']), 2)
            result = MatchResult(0, (15, 0), [], [])
            with patch('phase2_reference_evaluation.play_mirror_pair', return_value=(result, result)) as play:
                _chunk((paths['a'], metadata, 12345, [(0, 24690, 15)]))
                self.assertTrue(all(call.kwargs['deal_partition'] == 0 for call in play.call_args_list))
                self.assertEqual(play.call_count, 4)
