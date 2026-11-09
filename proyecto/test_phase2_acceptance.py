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
        from unittest.mock import patch
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
                (folder / 'reigning_champion.json').write_text(json.dumps({'temperature': .05, 'profile_prior': 5, 'generation': 3}))
            run = root / 'run'
            run.mkdir()
            (run / 'generation_000003.pt').write_bytes((root/'candidate/weights.pt').read_bytes())
            (run / 'generations.jsonl').write_text(json.dumps({'generation': 4, 'champion': 3}) + '\n')
            members = []
            for i in range(2):
                path = root / f'{i}.npy'
                np.save(path, np.zeros(GENOME_SIZE) + i / 100)
                members.append({'name': str(i), 'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            manifest = root / 'held.json'
            manifest.write_text(json.dumps({'members': members, 'sha256': [m['sha256'] for m in members]}))
            kwargs = dict(candidate=root/'candidate/weights.pt', baseline=root/'baseline/weights.pt',
                          held_out_manifest=manifest, direct_seed=12345, held_out_seed=12346, pairs=2, run_dir=run)
            baseline = root/'baseline/weights.pt'
            with patch('phase2_acceptance.BASELINE_SHA256', hashlib.sha256(baseline.read_bytes()).hexdigest()), \
                 patch('phase2_acceptance.BASELINE_SNAPSHOT', baseline), patch('phase2_acceptance._verify_source', return_value='fixture-source-hash'):
                evaluate(output=root/'one', workers=1, **kwargs)
                evaluate(output=root/'two', workers=2, **kwargs)
                a = json.loads((root/'one/results.json').read_text())
                b = json.loads((root/'two/results.json').read_text())
                self.assertEqual(a, b)
                self.assertEqual(a['config']['candidate']['generation'], 3)
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
                (root/'candidate/reigning_champion.json').write_text(json.dumps({'temperature': .05, 'profile_prior': 5, 'generation': 3}))
                members[0]['sha256'] = '0' * 64
                manifest.write_text(json.dumps({'members': members, 'sha256': [m['sha256'] for m in members]}))
                with self.assertRaises(ValueError):
                    evaluate(output=root/'bad', workers=1, **kwargs)

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

    def test_champion_identities_and_frozen_sidecars(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from phase2_torch import Phase2Network
        from phase2_acceptance import _champions
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root/'candidate/reigning_champion.pt'
            baseline = root/'baseline/reigning_champion.pt'
            candidate.parent.mkdir(); baseline.parent.mkdir()
            Phase2Network(seed=3).save(candidate); Phase2Network(seed=4).save(baseline)
            run = root/'run'; run.mkdir()
            snapshot = run/'generation_000003.pt'; snapshot.write_bytes(candidate.read_bytes())
            (run/'generations.jsonl').write_text(json.dumps({'champion': 3}) + '\n')
            for path in (candidate, baseline):
                (path.parent/'reigning_champion.json').write_text(json.dumps({'generation': 3, 'temperature': .05, 'profile_prior': 5}))
            with patch('phase2_acceptance.BASELINE_SHA256', hashlib.sha256(baseline.read_bytes()).hexdigest()), patch('phase2_acceptance.BASELINE_SNAPSHOT', baseline):
                info = _champions(candidate, baseline, run)
                self.assertEqual(info['candidate']['generation'], 3)
                sidecar = candidate.parent/'reigning_champion.json'
                for value in ({'generation': 2, 'temperature': .05, 'profile_prior': 5},
                              {'generation': 3, 'temperature': .1, 'profile_prior': 5},
                              {'generation': 3, 'temperature': .05, 'profile_prior': 20}):
                    sidecar.write_text(json.dumps(value))
                    with self.assertRaises(ValueError): _champions(candidate, baseline, run)
                sidecar.unlink()
                with self.assertRaises(ValueError): _champions(candidate, baseline, run)
                sidecar.write_text(json.dumps({'generation': 3, 'temperature': .05, 'profile_prior': 5}))
                baseline_sidecar = baseline.parent/'reigning_champion.json'
                saved = baseline_sidecar.read_text()
                baseline_sidecar.unlink()
                with self.assertRaises(ValueError): _champions(candidate, baseline, run)
                baseline_sidecar.write_text(saved)
                Phase2Network(seed=7).save(snapshot)
                with self.assertRaises(ValueError): _champions(candidate, baseline, run)
                snapshot.write_bytes(candidate.read_bytes())
                wrong_baseline_snapshot = root/'wrong.pt'
                Phase2Network(seed=9).save(wrong_baseline_snapshot)
                with patch('phase2_acceptance.BASELINE_SNAPSHOT', wrong_baseline_snapshot):
                    with self.assertRaisesRegex(ValueError, 'baseline identity'):
                        _champions(candidate, baseline, run)
                Phase2Network(seed=8).save(baseline)
                with self.assertRaises(ValueError): _champions(candidate, baseline, run)

    def test_seat_scores_for_both_parts(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from phase2_acceptance import play_pair
        for opponent in ('baseline', 'held_out'):
            for winners, score in [([0, 1], 1), ([1, 0], 0)]:
                with self.subTest(opponent=opponent, score=score), \
                     patch('phase2_acceptance.make_policy', side_effect=lambda spec, seed: spec), \
                     patch('phase2_acceptance.play_match', side_effect=[SimpleNamespace(winner=w) for w in winners]):
                    self.assertEqual(play_pair('candidate', opponent, 1, 15, 123, 0), score)

    def test_dirty_source_is_rejected(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from phase2_acceptance import _verify_source
        with patch('subprocess.run', return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(ValueError, 'uncommitted'):
                _verify_source()
        with patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=b'MM proyecto/phase2_acceptance.py')):
            with self.assertRaisesRegex(ValueError, 'uncommitted'):
                _verify_source()
