import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from proyecto.phase2_league import draw_opponent, load_phase1_manifest

class LeagueTest(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(draw_opponent(7, 3, pool=['a'], phase1=['b']), draw_opponent(7, 3, pool=['a'], phase1=['b']))

    def test_proportions_and_neutral(self):
        draws = [draw_opponent(7, i, pool=['a'], phase1=['b']) for i in range(20000)]
        counts = Counter(d['style'] or d['category'] for d in draws)
        for category, share in [('self', .5), ('phase1', .1), ('snapshot', .2), ('bluffer', .05), ('cautious', .05), ('aggressive', .05), ('calculating', .05)]:
            self.assertAlmostEqual(counts[category] / len(draws), share, delta=.012)
        self.assertAlmostEqual(sum(d['neutral_profile'] for d in draws) / len(draws), .05, delta=.01)

    def test_empty_pool_and_exclusions(self):
        draws = [draw_opponent(4, i, pool=['reserved'], phase1=['reserved', 'allowed'], exclusions={'reserved'}) for i in range(5000)]
        self.assertFalse(any(d['identity'] == 'reserved' or d['category'] == 'snapshot' for d in draws))
        self.assertAlmostEqual(sum(d['category'] == 'self' for d in draws) / len(draws), .7, delta=.025)
        self.assertTrue(all(d['identity'] == 'current' for d in draws if d['category'] in ('styled', 'self')))

    def test_ramp(self):
        shares = dict.fromkeys(('self', 'phase1', 'snapshot', 'bluffer', 'cautious', 'aggressive', 'calculating'), 0)
        shares['bluffer'] = 1
        for matches, expected in [(0, 0), (10000, .05), (20000, .1), (30000, .1)]:
            self.assertAlmostEqual(draw_opponent(1, 1, shares=shares, matches_so_far=matches)['offset'], expected)

    def test_manifest_hash(self):
        members = load_phase1_manifest('experiments/phase2-selection-panel/league-phase1.json')
        self.assertEqual(len(members), 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'weights').write_bytes(b'wrong')
            (path / 'manifest.json').write_text(json.dumps({'members': [{'name': 'x', 'path': 'weights', 'sha256': 'bad'}]}))
            with self.assertRaises(ValueError):
                load_phase1_manifest(path / 'manifest.json')


class SnapshotPoolTest(unittest.TestCase):
    def test_recent_diverse_excluded_and_ordered(self):
        from proyecto.phase2_league import retain_snapshots
        snapshots = [dict(generation=i, sha256=str(i), scores={'a': score}) for i, score in enumerate([0, .8, .2, .4, .5], 1)]
        kept = retain_snapshots(snapshots, recent=2, diverse=1, exclusions={'2'})
        self.assertEqual([s['sha256'] for s in kept], ['4', '5', '1'])
        self.assertEqual(kept, retain_snapshots(list(reversed(snapshots)), recent=2, diverse=1, exclusions={'2'}))

class PanelExclusionsTest(unittest.TestCase):
    def test_all_panel_reservations(self):
        from proyecto.phase2_league import panel_exclusions
        current = {'members': [{'sha256': 'current'}], 'pending_member': {'sha256': 'pending'}}
        history = [{'members': [{'sha256': 'old'}], 'reserved': ['retired']}]
        self.assertEqual(panel_exclusions(current, history), {'current', 'pending', 'old', 'retired'})
