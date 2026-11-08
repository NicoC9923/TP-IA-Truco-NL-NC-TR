import unittest
from phase2_promotion import promotion_decision

class PromotionTests(unittest.TestCase):
    def test_promotion_and_direct_screen(self):
        a = {'x': [1, 1, 1], 'y': [1, 1, 1]}
        b = {'x': [0.5]*3, 'y': [0.5]*3}
        self.assertTrue(promotion_decision(a, b, [0.5]*2)['promoted'])
        self.assertFalse(promotion_decision(a, b, [0.0]*2)['promoted'])
        self.assertFalse(promotion_decision({'x': [1, 0, 0]}, {'x': [0, 0, 0]}, [0.5])['promoted'])

class PanelTests(unittest.TestCase):
    def test_saturation_resets_on_champion_change(self):
        from phase2_promotion import saturated
        row = {'champion': 'a', 'scores': {'x': [.7]*100}}
        self.assertTrue(saturated([row]*3, 'a', ['x']))
        self.assertFalse(saturated([row]*2, 'a', ['x']))
        self.assertFalse(saturated([row, {'champion': 'b'}, row, row], 'a', ['x']))
        self.assertFalse(saturated([row]*3, 'a', ['x'], threshold=.71))

    def test_panel_ranking_and_postponement(self):
        from phase2_promotion import build_panel
        members = [{'name': str(i), 'kind': 'phase1', 'sha256': str(i)} for i in range(3)]
        panel = {'version': 1, 'members': members}
        snapshots = [{'sha256': 'a', 'gain': .3, 'scores': {'x': .2}}, {'sha256': 'b', 'gain': .1, 'scores': {'x': .8}}, {'sha256': 'c', 'gain': .2, 'scores': {'x': .3}}]
        result = build_panel(panel, {'0': .9, '1': .7, '2': .8}, snapshots, 'c')
        self.assertEqual([m['sha256'] for m in result['members']], ['1', '2', 'a', 'b'])
        self.assertIsNone(build_panel(panel, {'0': .9, '1': .7, '2': .8}, snapshots[:1], 'c'))

    def test_hash_and_reserved_seed(self):
        import hashlib, json, tempfile
        from pathlib import Path
        from phase2_promotion import check_seed, load_panel
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'weights').write_bytes(b'weights')
            member = {'name': 'x', 'path': 'weights', 'kind': 'phase1', 'encoder': 'v2', 'sha256': hashlib.sha256(b'weights').hexdigest()}
            (p/'panel.json').write_text(json.dumps({'version': 1, 'members': [member]}))
            self.assertEqual(load_panel(p/'panel.json')['members'][0]['encoder'], 'v2')
            (p/'weights').write_bytes(b'changed')
            with self.assertRaises(ValueError): load_panel(p/'panel.json')
        with self.assertRaises(ValueError): check_seed(9301, 0)
        with self.assertRaises(ValueError): check_seed(9300, 1)
