"""The array posterior must equal the original dict posterior bit for bit."""
from itertools import combinations
import random
import unittest
import numpy as np
import truco_reglas as rules
import opponent_inference as oi
from opponent_inference import Inference, _tantos, _strength


class DictInference(Inference):
    """Frozen copy of the pre-array implementation (ordered dict of hands)."""
    def __init__(self, own_cards):
        deck = [(v, p) for p in rules.PALOS for v in rules.VALORES]
        self.played = set()
        self._dict = {hand: 1.0 for hand in combinations((c for c in deck if c not in own_cards), 3)}
        self._summary_cache = None

    @property
    def weights(self):
        return self._dict

    def distribution(self):
        total = sum(self._dict.values())
        return {hand: weight / total for hand, weight in self._dict.items()}

    def declared_tantos(self, value):
        self._summary_cache = None
        self._dict = {h: w for h, w in self._dict.items() if _tantos(h) == value}

    def son_buenas(self, own_tantos, opponent_is_mano):
        self._summary_cache = None
        self._dict = {h: w for h, w in self._dict.items()
                      if _tantos(h) < own_tantos or (not opponent_is_mano and _tantos(h) == own_tantos)}

    def seen_cards(self, excluded=(), opponent_played=()):
        excluded, played = set(excluded), set(opponent_played)
        if not excluded and played.issubset(self.played):
            return
        self._summary_cache = None
        self.played.update(played)
        if excluded:
            self._dict = {h: w for h, w in self._dict.items() if not excluded.intersection(h) and played.issubset(h)}
        else:
            self._dict = {h: w for h, w in self._dict.items() if played.issubset(h)}

    def truco_call(self, bluff_rate):
        if not 0 <= bluff_rate <= 1:
            raise ValueError('Bluff rate must be between zero and one')
        self._summary_cache = None
        if bluff_rate == 1.0:
            return
        for hand in self._dict:
            strength = _strength(hand) / 24.7
            self._dict[hand] *= bluff_rate + (1 - bluff_rate) * (0.1 + strength)

    def _summary_arrays(self):
        if self._summary_cache is not None:
            return self._summary_cache
        ids = np.fromiter((oi._HAND_IDS[h] for h in self._dict), dtype=np.int64)
        weights = np.fromiter(self._dict.values(), dtype=np.float64)
        weights = weights / sum(self._dict.values())
        cards = oi._IDS[ids]
        remaining = np.ones(cards.shape, dtype=bool)
        for card in self.played:
            remaining &= cards != oi._CARD_IDS[card]
        ranks = np.where(remaining, oi._RANKS[cards], -1)
        counts = np.maximum(remaining.sum(axis=1), 1)
        brava = (oi._BRAVAS[cards] & remaining).any(axis=1)
        strength = oi._strength_table()[tuple((ranks + 1).T)]
        self._summary_cache = weights, ranks, counts, brava, strength
        return self._summary_cache

    def v3_summaries(self, cards, available, *, estimate_outcomes):
        weights, ranks, counts, brava, strength = self._summary_arrays()
        outcomes = [0.0, 0.0, 0.0]
        if estimate_outcomes:
            for i in available:
                rank = rules.valor_carta_truco(*cards[i])
                signs = ((rank > ranks).astype(int) - (rank < ranks)) * (ranks >= 0)
                outcomes[i] = oi._ordered_sum(weights * signs.sum(axis=1) / counts)
        return oi._ordered_sum(weights * brava), oi._ordered_sum(weights * strength), outcomes

    def summaries(self, card, *, ties_win=False):
        weights, ranks, counts, brava, strength = self._summary_arrays()
        rank = rules.valor_carta_truco(*card)
        wins = ((rank > ranks) | (ties_win & (rank == ranks))) & (ranks >= 0)
        return dict(brava_probability=oi._ordered_sum(weights * brava),
                    expected_strength=oi._ordered_sum(weights * strength),
                    next_trick_win_probability=oi._ordered_sum(weights * wins.sum(axis=1) / counts))


def first_difference(actual, expected):
    """Avoid difflib on 7,770-entry lists: report the first differing entry."""
    if actual == expected:
        return None
    index = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
    return index, len(actual), len(expected), actual[index:index + 1], expected[index:index + 1]


def same(test, actual, expected, own):
    test.assertIsNone(first_difference([(h, float(w).hex()) for h, w in actual.weights.items()],
                                       [(h, float(w).hex()) for h, w in expected.weights.items()]))
    test.assertEqual(actual.played, expected.played)
    if not expected.weights:
        return
    test.assertIsNone(first_difference([(h, float(w).hex()) for h, w in actual.distribution().items()],
                                       [(h, float(w).hex()) for h, w in expected.distribution().items()]))
    for a, b in zip(actual._summary_arrays(), expected._summary_arrays()):
        test.assertEqual((a.dtype, a.shape, a.tobytes()), (b.dtype, b.shape, b.tobytes()))
    for card in own:
        for ties in (False, True):
            test.assertEqual(repr(actual.summaries(card, ties_win=ties)), repr(expected.summaries(card, ties_win=ties)))
    for available in ((0, 1, 2), (1,), (0, 2), ()):
        for estimate in (True, False):
            test.assertEqual(repr(actual.v3_summaries(own, available, estimate_outcomes=estimate)),
                             repr(expected.v3_summaries(own, available, estimate_outcomes=estimate)))
    for card in oi._DECK[::7]:
        test.assertEqual(repr(actual.summaries(card)), repr(expected.summaries(card)))


class ArrayPosteriorTests(unittest.TestCase):
    def test_random_evidence_sequences_match_dict_posterior(self):
        rng = random.Random(2026)
        deck = list(oi._DECK)
        for trial in range(250):
            cards = rng.sample(deck, 9)
            log = []
            own, hidden = tuple(cards[:3]), cards[3:6]
            actual, expected = Inference(own), DictInference(own)
            same(self, actual, expected, own)
            played = []
            for step in range(rng.randint(1, 8)):
                kind = rng.random()
                if kind < .3 and len(played) < 3:
                    played.append(hidden[len(played)])
                    args = dict(opponent_played=list(played))
                    if rng.random() < .3:
                        args['excluded'] = [rng.choice(cards[6:])]
                    for belief in (actual, expected):
                        belief.seen_cards(**args)
                elif kind < .45:
                    for belief in (actual, expected):
                        belief.seen_cards(opponent_played=list(played))
                elif kind < .7:
                    rate = rng.choice((0.0, 1.0, .5, rng.random(), 1 / 3, np.float64(rng.random())))
                    for belief in (actual, expected):
                        belief.truco_call(rate)
                elif kind < .85:
                    value = _tantos(tuple(hidden)) if rng.random() < .8 else rng.randint(0, 33)
                    for belief in (actual, expected):
                        belief.declared_tantos(value)
                else:
                    own_tantos, mano = rng.randint(0, 33), rng.random() < .5
                    for belief in (actual, expected):
                        belief.son_buenas(own_tantos, mano)
                log.append(kind)
                with self.subTest(trial=trial, step=step, log=log[:]):
                    same(self, actual, expected, own)

    def test_assigning_weights_keeps_given_order(self):
        own = ((1, 'Espada'), (2, 'Copa'), (3, 'Oro'))
        belief, reference = Inference(own), DictInference(own)
        items = list(reference.weights.items())[::-3]
        belief.weights = dict(items)
        reference._dict = dict(items)
        belief.truco_call(.3)
        reference.truco_call(.3)
        same(self, belief, reference, own)

    def test_invalid_bluff_rate_still_raises(self):
        with self.assertRaises(ValueError):
            Inference(((1, 'Espada'), (2, 'Copa'), (3, 'Oro'))).truco_call(1.5)


if __name__ == '__main__':
    unittest.main()
