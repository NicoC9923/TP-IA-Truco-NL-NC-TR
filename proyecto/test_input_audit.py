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
