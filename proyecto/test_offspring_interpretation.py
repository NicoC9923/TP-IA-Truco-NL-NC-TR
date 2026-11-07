import unittest
from offspring_diagnostic import interpretation

class InterpretationTests(unittest.TestCase):
    def test_no_rules(self):
        s = {'ci95': [-.1,.1], 'mean': 0., 'spread': .1, 'fraction_beating_parent': .5}
        off = {'pairs': 32, 'variants': {'mutation_0.050': s, 'mutation_0.025': s}, 'crossover_minus_mutation': s}
        beh = {'variants': {'mutation_0.050': {'mean_argmax_agreement': .9}}, 'population': {'mean_pairwise_tv': .2}, 'panel': {'mean_pairwise_tv': .3}}
        self.assertEqual(interpretation(off, beh)['recommendation'], 'no operator diagnosis; consider inputs (proposal 7)')
    def test_all_rules_can_fire(self):
        bad = {'ci95': [-.12, -.02], 'mean': -.07, 'spread': .01, 'fraction_beating_parent': .2}
        small = {'mean': -.01}
        off = {'pairs': 32, 'variants': {'mutation_0.050': bad, 'mutation_0.025': small}, 'crossover_minus_mutation': bad}
        beh = {'variants': {'mutation_0.050': {'mean_argmax_agreement': .99}}, 'population': {'mean_pairwise_tv': .01}, 'panel': {'mean_pairwise_tv': .3}}
        result = interpretation(off, beh)
        self.assertEqual(len(result['fired_rules']), 3)
        self.assertIn('crossover damage: proposal 4', result['fired_rules'])
        self.assertIn('mutation too destructive: smaller sigma', result['fired_rules'])
        self.assertIn('low behavioral diversity: proposal 6', result['fired_rules'])
        bad['ci95'] = [-.02, .02]
        self.assertIn('mutation too weak: larger sigma', interpretation(off, beh)['fired_rules'])
