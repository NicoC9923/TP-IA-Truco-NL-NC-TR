import unittest

from input_audit import card_strength, determinable_winners
import truco_reglas as rules


class InputAuditTests(unittest.TestCase):
    def test_hierarchy_matches_rules_for_the_deck(self):
        for suit in rules.PALOS:
            for value in rules.VALORES:
                self.assertEqual(card_strength((value, suit)), rules.valor_carta_truco(value, suit))

    def test_opponent_card_on_table_determines_winners(self):
        observation = {'player': 0, 'hand': ((1, 'Espada'), (4, 'Oro'), (3, 'Copa')),
                       'available': (0, 1, 2), 'table': ((1, (2, 'Oro')),)}
        self.assertEqual(determinable_winners(observation), (0, 2))

    def test_no_opponent_card_means_unknown(self):
        observation = {'player': 0, 'hand': ((1, 'Espada'),), 'available': (0,), 'table': ()}
        self.assertIsNone(determinable_winners(observation))

    def test_mistakes_use_only_legal_actions_and_certain_facts(self):
        from input_audit import mistake_actions
        obs = {'player': 0, 'hand': ((7, 'Oro'), (6, 'Oro'), (4, 'Copa')),
               'available': (0, 1, 2), 'table': (), 'tricks': (),
               'pending': {'kind': 'envido'}, 'envido_reveal': None}
        self.assertEqual(mistake_actions(obs, ('quiero', 'no_quiero', 'fold'))['decline_high_envido'], ('no_quiero', 'fold'))

    def test_near_aliasing_respects_relevant_state(self):
        from input_audit import aliasing
        import numpy as np
        result = aliasing(np.array([[0., 0.], [0., 0.], [0.001, 0.], [1., 1.]]),
                          [('a',), ('b',), ('a',), ('c',)], radius=0.01)
        self.assertEqual(result['exact']['observations'], 2)
        self.assertEqual(result['near']['observations'], 3)

    def test_card_choice_reports_winning_and_cheapest_probability(self):
        from input_audit import card_choice_audit
        obs = {'player': 0, 'hand': ((1, 'Espada'), (4, 'Oro'), (3, 'Copa')),
               'available': (0, 1, 2), 'table': ((1, (2, 'Oro')),)}
        p = {'card_0': .2, 'card_1': .5, 'card_2': .3}
        result = card_choice_audit([(obs, p)])
        self.assertEqual(result['opportunities'], 1)
        self.assertAlmostEqual(result['winning_mass'], .5)
        self.assertAlmostEqual(result['cheapest_winning_mass'], .3)
        self.assertAlmostEqual(result['stronger_winning_mass'], .2)
        self.assertEqual(result['pairwise']['disagree']['pairs'], 2)

    def test_per_card_uses_table_when_opponent_has_played(self):
        from input_audit import per_card_win_probabilities
        obs = {'player': 0, 'hand': ((1, 'Espada'), (4, 'Oro'), (3, 'Copa')),
               'available': (0, 1, 2), 'table': ((1, (2, 'Oro')),)}
        self.assertEqual(per_card_win_probabilities(obs), {0: 1., 1: 0., 2: 1.})

    def test_per_card_comparison_separates_first_and_other(self):
        from input_audit import per_card_audit
        rows = [({'available': (0, 1)}, {'card_0': .2, 'card_1': .8}, {0: .8, 1: .1}),
                ({'available': (0, 1)}, {'card_0': .7, 'card_1': .3}, {0: .1, 1: .8})]
        result = per_card_audit(rows)
        self.assertEqual(result['first']['opportunities'], 1)
        self.assertEqual(result['other']['opportunities'], 1)
        self.assertEqual(result['first']['best_card_argmax_rate'], 0.)
        self.assertEqual(result['other']['best_card_argmax_rate'], 0.)
        self.assertAlmostEqual(result['other']['best_card_mass'], .3)


    def test_preference_probe_holds_other_features_fixed(self):
        import numpy as np
        from input_audit import preference_probe
        from phase1_network import ACTIONS, GENOME_SIZE, INPUTS, HIDDEN
        genome = np.zeros(GENOME_SIZE)
        genome[0] = 1.0
        genome[INPUTS * HIDDEN + HIDDEN] = 1.0
        result = preference_probe(genome, np.zeros(INPUTS), slot=0)
        self.assertGreater(result['face_value_correlation'], .99)
        self.assertGreater(result['face_value_correlation'], result['strength_correlation'])

    def test_inference_probabilities_are_for_each_available_card(self):
        from input_audit import per_card_win_probabilities
        obs = {'player': 0, 'mano': 0, 'hand': ((1, 'Espada'), (4, 'Oro'), (3, 'Copa')),
               'available': (0, 1, 2), 'table': (), 'history': (), 'envido_reveal': None}
        p = per_card_win_probabilities(obs)
        self.assertGreater(p[0], p[2])
        self.assertGreater(p[2], p[1])
