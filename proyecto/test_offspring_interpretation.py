import unittest
from offspring_diagnostic import interpretation

class InterpretationTests(unittest.TestCase):
    def test_no_rules(self):
        s = {'ci95': [-.1,.1], 'mean': 0., 'spread': .1, 'fraction_beating_parent': .5}
        off = {'pairs': 32, 'variants': {'mutation_0.050': s, 'mutation_0.025': s}, 'crossover_minus_mutation': s}
        beh = {'variants': {'mutation_0.050': {'mean_argmax_agreement': .9}}, 'population': {'mean_pairwise_tv': .2}, 'panel': {'mean_pairwise_tv': .3}}
        self.assertEqual(interpretation(off, beh)['recommendation'], 'no operator diagnosis; consider inputs (proposal 7)')
