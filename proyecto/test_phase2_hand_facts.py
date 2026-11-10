"""Run 3 encoder (ADR 0062): hand facts appended as new columns, checked against the rules module."""
import unittest
from itertools import combinations
import numpy as np
import truco_reglas as rules
import truco_strength
from opponent_inference import exact_inference_from_observation
from opponent_profile import OpponentProfile
import phase2_network as net

ALL_LEGAL = ('card_0', 'card_1', 'card_2', 'envido', 'real_envido', 'falta_envido', 'truco', 'fold')
V1 = net.INPUT_SIZE


def observation(hand, *, available=(0, 1, 2), mano=0, player=0, pending=None, history=(), table=(),
                reveal=None, score=(0, 0), target=15):
    return {'player': player, 'hand': hand, 'available': available, 'score': score, 'target': target,
            'mano': mano, 'table': table, 'tricks': (), 'truco': 1, 'pending': pending,
            'history': history, 'envido_done': False, 'envido_reveal': reveal, 'raise_owner': None}


def tanto(hand):
    return rules.calcular_envido([{'valor': v, 'palo': p} for v, p in hand])


def encode(obs, legal=ALL_LEGAL, version=2):
    return net.encode_observation(obs, OpponentProfile(prior_strength=5), legal, encoder_version=version)


STRONG = ((7, 'Espada'), (6, 'Espada'), (1, 'Oro'))


class LayoutTest(unittest.TestCase):
    def test_sizes_and_versions(self):
        self.assertEqual(net.INPUT_SIZES, {1: 1628, 2: 1634, 3: 1635})
        self.assertEqual(net.encoder_version_for_size(1634), 2)
        with self.assertRaises(ValueError):
            net.encoder_version_for_size(1700)

    def test_new_versions_keep_the_old_columns_unchanged(self):
        obs = observation(STRONG)
        old = encode(obs, version=1)
        for version, size in ((2, 1634), (3, 1635)):
            new = encode(obs, version=version)
            self.assertEqual(new.shape, (size,))
            self.assertEqual(new[:V1].tobytes(), old.tobytes())
        with self.assertRaises(ValueError):
            encode(obs, version=9)

    def test_default_version_is_the_old_encoder(self):
        plain = net.encode_observation(observation(STRONG), OpponentProfile(prior_strength=5), ALL_LEGAL)
        self.assertEqual(plain.shape, (V1,))


class OwnHandFactsTest(unittest.TestCase):
    def test_own_envido_total_uses_the_original_three_cards(self):
        full = encode(observation(STRONG))
        self.assertAlmostEqual(float(full[V1]), 33 / 33)
        # Two cards already played: the total still comes from all three.
        self.assertEqual(encode(observation(STRONG, available=(2,)))[V1], full[V1])
        low = ((12, 'Espada'), (11, 'Basto'), (4, 'Oro'))
        self.assertAlmostEqual(float(encode(observation(low))[V1]), tanto(low) / 33, places=6)

    def test_flush_pair_bit(self):
        self.assertEqual(encode(observation(((7, 'Espada'), (1, 'Espada'), (1, 'Oro'))))[V1 + 1], 1)
        self.assertEqual(encode(observation(((7, 'Espada'), (1, 'Basto'), (1, 'Oro'))))[V1 + 1], 0)
        self.assertEqual(encode(observation(((7, 'Espada'), (1, 'Espada'), (2, 'Espada'))))[V1 + 1], 1)
        self.assertEqual(encode(observation(((7, 'Espada'), (1, 'Espada'), (2, 'Oro')), available=(2,)))[V1 + 1], 1)

    def test_hand_percentile_matches_the_standalone_table(self):
        hand = ((3, 'Oro'), (2, 'Copa'), (12, 'Basto'))
        vector = encode(observation(hand))
        self.assertEqual(vector[V1 + 2], np.float32(truco_strength.percentile(list(hand))))
        self.assertEqual(vector[V1 + 2], encode(observation(hand, available=(0,)))[V1 + 2])

    def test_unplayed_percentile_follows_the_remaining_cards(self):
        hand = ((1, 'Espada'), (4, 'Copa'), (3, 'Oro'))
        for available in ((0, 1, 2), (0, 2), (1,), (2,)):
            vector = encode(observation(hand, available=available))
            expected = truco_strength.percentile([hand[i] for i in available])
            self.assertEqual(vector[V1 + 3], np.float32(expected))
        self.assertEqual(encode(observation(hand, available=()))[V1 + 3], 0)


class StakeTest(unittest.TestCase):
    def test_fresh_hand_envido_and_real_envido(self):
        vector = encode(observation(STRONG))
        self.assertAlmostEqual(float(vector[V1 + 4]), 2 / 30)
        self.assertAlmostEqual(float(vector[V1 + 5]), 3 / 30)

    def test_stakes_include_the_pending_envido_chain(self):
        pending = {'kind': 'envido', 'caller': 1, 'call': 'envido', 'accepted': 2, 'declined': 1}
        legal = ('quiero', 'no_quiero', 'fold', 'real_envido', 'falta_envido')
        vector = encode(observation(STRONG, pending=pending, history=((1, 'envido'),)), legal)
        self.assertEqual(vector[V1 + 4], 0)  # a bare envido is not a legal raise here
        self.assertAlmostEqual(float(vector[V1 + 5]), 5 / 30)
        chained = {'kind': 'envido', 'caller': 1, 'call': 'envido_envido', 'accepted': 4, 'declined': 2}
        vector = encode(observation(STRONG, pending=chained, history=((0, 'envido'), (1, 'envido'))), legal)
        self.assertAlmostEqual(float(vector[V1 + 5]), 7 / 30)

    def test_stakes_are_zero_when_envido_cannot_be_called(self):
        vector = encode(observation(STRONG, available=(0, 1)), ('card_0', 'card_1', 'fold', 'truco'))
        self.assertEqual((vector[V1 + 4], vector[V1 + 5]), (0, 0))

    def test_stake_after_a_pending_truco_starts_from_zero(self):
        pending = {'kind': 'truco', 'caller': 1, 'level': 2}
        legal = ('quiero', 'no_quiero', 'fold', 'retruco', 'envido', 'real_envido', 'falta_envido')
        vector = encode(observation(STRONG, pending=pending, history=((1, 'truco'),)), legal)
        self.assertAlmostEqual(float(vector[V1 + 4]), 2 / 30)
        self.assertAlmostEqual(float(vector[V1 + 5]), 3 / 30)


def brute_force_win_probability(own_hand, own_is_mano):
    others = [(v, p) for p in rules.PALOS for v in rules.VALORES if (v, p) not in own_hand]
    mine = tanto(own_hand)
    wins = total = 0
    for hand in combinations(others, 3):
        theirs = tanto(hand)
        total += 1
        wins += mine > theirs or (mine == theirs and own_is_mano)
    return wins / total


class TantoBeliefTest(unittest.TestCase):
    def test_probability_matches_enumeration_for_both_seats(self):
        for hand in (STRONG, ((12, 'Espada'), (11, 'Basto'), (5, 'Oro')), ((1, 'Copa'), (2, 'Copa'), (3, 'Oro'))):
            for mano in (0, 1):
                vector = encode(observation(hand, mano=mano), version=3)
                self.assertAlmostEqual(float(vector[V1 + 6]), brute_force_win_probability(hand, mano == 0), places=5)

    def test_a_declared_tanto_makes_it_certain(self):
        reveal = {'declarations': ((1, 25), (0, 'son buenas'))}
        beats = encode(observation(STRONG, mano=1, reveal=reveal), version=3)
        self.assertAlmostEqual(float(beats[V1 + 6]), 1.0)
        weak = ((12, 'Espada'), (11, 'Basto'), (5, 'Oro'))
        self.assertAlmostEqual(float(encode(observation(weak, mano=1, reveal=reveal), version=3)[V1 + 6]), 0.0)

    def test_inference_method_agrees_with_enumeration(self):
        belief = exact_inference_from_observation(observation(STRONG))
        self.assertAlmostEqual(belief.tanto_win_probability(33, ties_win=True),
                               brute_force_win_probability(STRONG, True), places=9)


if __name__ == '__main__':
    unittest.main()
