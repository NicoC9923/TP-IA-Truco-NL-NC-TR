import unittest

class AcceptanceStatsTest(unittest.TestCase):
    def test_interval_and_strict_decision(self):
        from phase2_acceptance import interval, decision
        result = interval([0, .5, 1])
        self.assertAlmostEqual(result['mean'], .5)
        self.assertAlmostEqual(result['lower'], .5 - 1.96 * .5 / 3**.5)
        self.assertEqual(decision({'lower': .5, 'upper': .6}, {'lower': 0, 'upper': .1}), 'inconclusive')
        self.assertEqual(decision({'lower': .51, 'upper': .6}, {'lower': -.03, 'upper': 0}), 'inconclusive')
        self.assertEqual(decision({'lower': .51, 'upper': .6}, {'lower': -.029, 'upper': 0}), 'ships')
        self.assertEqual(decision({'lower': .4, 'upper': .49}, {'lower': 0, 'upper': .1}), 'regression')
        self.assertEqual(decision({'lower': .4, 'upper': .6}, {'lower': -.1, 'upper': -.031}), 'regression')

    def test_fresh_match_policies_and_seat_seed_formula(self):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import patch
        from phase2_acceptance import play_pair
        calls = []
        def factory(spec, seed):
            value = SimpleNamespace(spec=spec, seed=seed)
            calls.append(value)
            return value
        def play(a, b, **kwargs):
            self.assertEqual(kwargs['deal_partition'], 0)
            self.assertEqual(kwargs['starting_score'], (0, 0))
            return SimpleNamespace(winner=0)
        with patch('phase2_acceptance.make_policy', side_effect=factory), patch('phase2_acceptance.play_match', side_effect=play):
            self.assertEqual(play_pair('candidate', 'baseline', 123, 15, 987, 4), .5)
        self.assertEqual([x.spec for x in calls], ['candidate', 'baseline', 'baseline', 'candidate'])
        expected = [int(np.random.SeedSequence([987, 4, match, seat]).generate_state(1)[0])
                    for match in range(2) for seat in range(2)]
        self.assertEqual([x.seed for x in calls], expected)
        self.assertEqual(len({id(x) for x in calls}), 4)

    def test_end_to_end_resume_and_workers(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        import numpy as np
        from phase1_network import GENOME_SIZE
        from phase2_torch import Phase2Network
        from phase2_acceptance import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('candidate', 'baseline'):
                folder = root / name
                folder.mkdir()
                Phase2Network(seed=3 if name == 'candidate' else 4).save(folder / 'weights.pt')
                (folder / 'reigning_champion.json').write_text(json.dumps({'temperature': .05, 'profile_prior': 5}))
            members = []
            for i in range(2):
                path = root / f'{i}.npy'
                np.save(path, np.zeros(GENOME_SIZE) + i / 100)
                members.append({'name': str(i), 'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            manifest = root / 'held.json'
            manifest.write_text(json.dumps({'members': members, 'sha256': [m['sha256'] for m in members]}))
            kwargs = dict(candidate=root/'candidate/weights.pt', baseline=root/'baseline/weights.pt',
                          held_out_manifest=manifest, direct_seed=12345, held_out_seed=12346, pairs=2)
            evaluate(output=root/'one', workers=1, **kwargs)
            evaluate(output=root/'two', workers=2, **kwargs)
            a = json.loads((root/'one/results.json').read_text())
            b = json.loads((root/'two/results.json').read_text())
            self.assertEqual(a, b)
            original = (root/'one/results.json').read_bytes()
            partial = root/'one/partial.jsonl'
            partial.write_text(''.join(partial.read_text().splitlines(keepends=True)[:-2]))
            evaluate(output=root/'one', workers=2, resume=True, **kwargs)
            self.assertEqual(original, (root/'one/results.json').read_bytes())
            with self.assertRaises(FileExistsError):
                evaluate(output=root/'one', workers=1, **kwargs)
            (root/'candidate/reigning_champion.json').unlink()
            with self.assertRaises(ValueError):
                evaluate(output=root/'missing', workers=1, **kwargs)
            members[0]['sha256'] = '0' * 64
            manifest.write_text(json.dumps({'members': members, 'sha256': [m['sha256'] for m in members]}))
            with self.assertRaises(ValueError):
                evaluate(output=root/'bad', workers=1, candidate_temperature=.05, candidate_prior=5, **kwargs)

    def test_clustered_statistics_use_pair_indices_and_equal_weights(self):
        from phase2_acceptance import summarize
        rows = [
            {'direct': 0, 'held_out': {'a': {'candidate': 1, 'baseline': 0}, 'b': {'candidate': 0, 'baseline': 1}}},
            {'direct': .5, 'held_out': {'a': {'candidate': 1, 'baseline': .5}, 'b': {'candidate': .5, 'baseline': .5}}},
            {'direct': 1, 'held_out': {'a': {'candidate': 0, 'baseline': 1}, 'b': {'candidate': 1, 'baseline': 0}}},
        ]
        result = summarize(rows, ['a', 'b'])
        self.assertAlmostEqual(result['direct']['mean'], .5)
        self.assertAlmostEqual(result['held_out']['mean'], 1/12)
        self.assertAlmostEqual(result['held_out']['margin'], 1.96 / 12)
        self.assertAlmostEqual(result['held_out_individual']['a']['mean'], 1/6)
        self.assertAlmostEqual(result['held_out_individual']['b']['mean'], 0)

    def test_network_policies_start_with_fresh_profiles(self):
        from phase2_acceptance import make_policy
        from phase2_torch import Phase2Network
        spec = {'kind': 'phase2', 'model': Phase2Network(seed=4), 'temperature': .05, 'profile_prior': 5}
        first, second = make_policy(spec, 11), make_policy(spec, 12)
        self.assertIsNot(first.profile, second.profile)
        self.assertIsNot(first.rng, second.rng)
        self.assertEqual(first.profile.prior_strength, 5)
