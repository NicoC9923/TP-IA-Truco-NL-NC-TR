import unittest
from opponent_inference import Inference

class InferenceTests(unittest.TestCase):
    def test_initial_hands_exclude_own_cards(self):
        own = ((1, 'Espada'), (2, 'Copa'), (3, 'Oro'))
        belief = Inference(own)
        self.assertEqual(len(belief.distribution()), 7770)
        self.assertTrue(all(not set(hand) & set(own) for hand in belief.distribution()))
    def test_certain_evidence_filters_cards_and_tantos(self):
        belief = Inference(((1, 'Espada'), (2, 'Copa'), (3, 'Oro')))
        belief.declared_tantos(33)
        self.assertTrue(belief.distribution())
        self.assertTrue(all(any((7, suit) in h and (6, suit) in h for suit in ('Espada', 'Basto', 'Oro', 'Copa')) for h in belief.distribution()))
        belief.seen_cards(excluded=[(7, 'Espada')], opponent_played=[(7, 'Oro')])
        self.assertTrue(belief.distribution())
        self.assertTrue(all((7, 'Espada') not in h and (7, 'Oro') in h for h in belief.distribution()))

    def test_son_buenas_respects_mano(self):
        own = ((7, 'Espada'), (6, 'Espada'), (4, 'Oro'))
        pie = Inference(own)
        pie.son_buenas(27, opponent_is_mano=True)
        self.assertTrue(all(tantos(h) < 27 for h in pie.distribution()))
        mano = Inference(own)
        mano.son_buenas(27, opponent_is_mano=False)
        self.assertTrue(any(tantos(h) == 27 for h in mano.distribution()))

    def test_truco_weights_without_restoring_excluded_hands(self):
        own = ((1, 'Espada'), (2, 'Copa'), (3, 'Oro'))
        low, high = Inference(own), Inference(own)
        for belief in (low, high):
            belief.seen_cards(excluded=[(1, 'Basto')])
        before = set(low.distribution())
        low.truco_call(0.1)
        high.truco_call(0.9)
        self.assertEqual(before, set(low.distribution()))
        strong = lambda d: sum(w for h, w in d.items() if (7, 'Espada') in h)
        self.assertGreater(strong(low.distribution()), strong(high.distribution()))

    def test_summaries_on_one_possible_hand(self):
        belief = Inference(((1, 'Espada'), (2, 'Copa'), (3, 'Oro')))
        hand = ((1, 'Basto'), (4, 'Copa'), (5, 'Oro'))
        belief.seen_cards(excluded=[(v, p) for p in ('Espada', 'Basto', 'Oro', 'Copa') for v in (1, 2, 3, 4, 5, 6, 7, 10, 11, 12) if (v, p) not in hand])
        summary = belief.summaries((7, 'Espada'))
        self.assertEqual(summary['brava_probability'], 1)
        self.assertAlmostEqual(summary['expected_strength'], 12.6)
        self.assertAlmostEqual(summary['next_trick_win_probability'], 2 / 3)


def tantos(hand):
    from truco_reglas import calcular_envido
    return calcular_envido([{'valor': v, 'palo': p} for v, p in hand])


class EvidenceTests(unittest.TestCase):
    def test_accepted_envido_becomes_public_without_hidden_cards(self):
        from symmetric_engine import play_match
        seen = []
        def caller(obs, actions):
            seen.append(obs)
            return 'envido' if 'envido' in actions else 'fold'
        play_match(caller, lambda o, a: 'quiero' if 'quiero' in a else 'fold', seed=3, target=15)
        self.assertTrue(any(o['envido_reveal'] for o in seen))
        reveal = next(o['envido_reveal'] for o in seen if o['envido_reveal'])
        self.assertIn('declarations', reveal)
        self.assertNotIn('opponent_hand', seen[-1])

    def test_objective_bluff_uses_same_rule_for_both_players(self):
        from symmetric_engine import play_match
        from truco_reglas import calcular_envido
        def call(obs, legal):
            return 'envido' if 'envido' in legal else 'no_quiero' if 'no_quiero' in legal else 'fold'
        result = play_match(call, call, seed=21, target=10)
        calls = [d for d in result.decisions if d['action'] == 'envido']
        self.assertEqual({d['player'] for d in calls}, {0, 1})
        for d in calls:
            cards = [{'valor': v, 'palo': p} for v, p in d['observation']['hand']]
            self.assertEqual(d['bluff'], calcular_envido(cards) < 24)
    def test_truco_bluff_known_deals(self):
        from symmetric_engine import play_match
        def caller(obs, legal):
            return 'truco' if 'truco' in legal else 'no_quiero' if 'no_quiero' in legal else 'fold'
        self.assertFalse(play_match(caller, caller, seed=0, target=1).decisions[0]['bluff'])
        self.assertTrue(play_match(caller, caller, seed=3, target=1).decisions[0]['bluff'])
