import unittest
from symmetric_engine import play_match, play_mirror_pair


class RaiseAcceptanceTests(unittest.TestCase):
    def match(self, actions, target):
        moves = iter(actions)
        def policy(obs, legal):
            return next(moves)
        return play_match(policy, policy, seed=0, target=target)

    def test_raise_accepts_lower_bet_and_disallows_envido(self):
        result = self.match(['truco', 'retruco', 'vale_cuatro', 'fold'], 3)
        for decision, value, owner in zip(result.decisions[2:], (2, 3), (1, 0)):
            self.assertEqual(decision['observation']['truco'], value)
            self.assertEqual(decision['observation']['raise_owner'], owner)
            self.assertTrue({'envido', 'real_envido', 'falta_envido'}.isdisjoint(decision['legal_actions']))

    def test_envido_after_retruco_is_illegal(self):
        with self.assertRaisesRegex(ValueError, 'Illegal action'):
            self.match(['truco', 'retruco', 'envido'], 2)

    def test_first_truco_can_be_interrupted_with_any_envido(self):
        result = self.match(['truco', 'envido', 'fold'], 2)
        self.assertTrue({'envido', 'real_envido', 'falta_envido'} <= set(result.decisions[1]['legal_actions']))
        self.assertEqual(result.points, (0, 2))
        self.assertEqual(result.decisions[2]['observation']['truco'], 1)

    def test_declining_raised_bets_awards_accepted_level(self):
        for raises, points, expected in ((['truco', 'retruco'], 2, (0, 2)), (['truco', 'retruco', 'vale_cuatro'], 3, (3, 0))):
            for decline in ('no_quiero', 'fold'):
                with self.subTest(raises=raises, decline=decline):
                    self.assertEqual(self.match(raises + [decline], points).points, expected)

    def test_acceptance_and_raise_right_after_chain(self):
        result = self.match(['truco', 'retruco', 'quiero', 'card_0', 'fold'], 3)
        self.assertEqual(result.decisions[3]['observation']['truco'], 3)
        self.assertEqual(result.decisions[3]['observation']['raise_owner'], 0)
        self.assertIn('vale_cuatro', result.decisions[3]['legal_actions'])
        self.assertNotIn('vale_cuatro', result.decisions[4]['legal_actions'])
        self.assertEqual(result.points, (3, 0))
        result = self.match(['truco', 'retruco', 'vale_cuatro', 'quiero', 'fold'], 4)
        self.assertEqual(result.decisions[4]['observation']['truco'], 4)
        self.assertEqual(result.decisions[4]['observation']['raise_owner'], 1)
        self.assertEqual(result.points, (0, 4))

    def test_mirror_pair_is_deterministic(self):
        def run():
            def policy(obs, legal):
                history = obs['history']
                return ('truco', 'retruco', 'no_quiero')[len(history)]
            return play_mirror_pair(policy, policy, seed=7, target=2)
        first, second = run(), run()
        self.assertEqual(first, second)
        self.assertEqual([r.points for r in first], [(0, 2), (0, 2)])
