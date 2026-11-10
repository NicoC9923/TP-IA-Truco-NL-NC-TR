"""Public Bayesian inference over the opponent's original three cards."""
from itertools import combinations
from functools import lru_cache
import truco_reglas as rules
import numpy as np

_DECK = tuple((v, p) for p in rules.PALOS for v in rules.VALORES)
_CARD_IDS = {card: i for i, card in enumerate(_DECK)}
_HANDS = tuple(combinations(_DECK, 3))
_HAND_IDS = {hand: i for i, hand in enumerate(_HANDS)}
_IDS = np.array([[ _CARD_IDS[c] for c in hand] for hand in _HANDS])
_MEMBERSHIP = np.array([(_IDS == card).any(axis=1) for card in range(len(_DECK))])
_RANKS = np.array([rules.valor_carta_truco(*c) for c in _DECK])
_BRAVAS = np.array([c in {(1, 'Espada'), (1, 'Basto'), (7, 'Espada'), (7, 'Oro')} for c in _DECK])

@lru_cache(maxsize=1)
def _strength_table():
    table = np.empty((15, 15, 15))
    for indexes in np.ndindex(table.shape):
        ranks = sorted((i - 1 for i in indexes if i), reverse=True)
        table[indexes] = round(sum(r * factor for r, factor in zip(ranks, (1.0, .6, .3))), 2)
    return table

def _ordered_sum(values):
    # np.sum uses pairwise reduction: preserve Python's left-to-right additions.
    return float(np.cumsum(values)[-1]) if len(values) else 0.0


class Inference:
    """Ordered posterior over the opponent's original hands.

    Hands are kept as parallel arrays (hand ids, float64 weights) in the original
    dict order: ascending hand id after construction, preserved by every filter.
    Each mutator performs the same IEEE operations, element by element, as the
    original dict code, and totals still use Python's ``sum`` over the same
    values in the same order (Python 3.12 compensated summation). A NumPy
    scalar bluff rate used to turn every weight into ``np.float64``, which makes
    ``sum`` add naively; ``_scalars`` records that so totals stay identical.
    """
    def __init__(self, own_cards):
        own = [_CARD_IDS[c] for c in set(own_cards) if c in _CARD_IDS]
        self.played = set()
        self._ids = np.flatnonzero(~_MEMBERSHIP[own].any(axis=0)) if own else np.arange(len(_HANDS))
        self._weights = np.ones(len(self._ids))
        self._scalars = False
        self._summary_cache = None

    def _values(self):
        # Python floats exactly as the dict held them, or np.float64 objects.
        return list(self._weights) if self._scalars else self._weights.tolist()

    def _total(self):
        return sum(self._values())

    @property
    def weights(self):
        """Ordered {hand: weight} view; build it only outside hot paths."""
        return dict(zip((_HANDS[i] for i in self._ids.tolist()), self._values()))

    @weights.setter
    def weights(self, value):
        self._summary_cache = None
        self._scalars = any(type(w) is not float for w in value.values())
        self._ids = np.fromiter((_HAND_IDS[h] for h in value), dtype=np.int64, count=len(value))
        self._weights = np.fromiter(value.values(), dtype=np.float64, count=len(value))

    def _keep(self, mask):
        self._ids = self._ids[mask]
        self._weights = self._weights[mask]

    def distribution(self):
        total = self._total()
        values = self._weights / total
        return dict(zip((_HANDS[i] for i in self._ids.tolist()), list(values) if self._scalars else values.tolist()))

    def declared_tantos(self, value):
        self._summary_cache = None
        self._keep(_hand_tantos()[self._ids] == value)

    def son_buenas(self, own_tantos, opponent_is_mano):
        self._summary_cache = None
        tantos = _hand_tantos()[self._ids]
        self._keep((tantos < own_tantos) | ((not opponent_is_mano) & (tantos == own_tantos)))

    def seen_cards(self, excluded=(), opponent_played=()):
        excluded, played = set(excluded), set(opponent_played)
        # Earlier filters already enforce these cards; later evidence and calls
        # only remove hands or change weights, never restore excluded hands.
        if not excluded and played.issubset(self.played):
            return
        self._summary_cache = None
        self.played.update(played)
        if any(card not in _CARD_IDS for card in played):
            self._keep(np.zeros(len(self._ids), dtype=bool))
            return
        mask = np.ones(len(self._ids), dtype=bool)
        for card in played:
            mask &= _MEMBERSHIP[_CARD_IDS[card]][self._ids]
        for card in excluded:
            if card in _CARD_IDS:
                mask &= ~_MEMBERSHIP[_CARD_IDS[card]][self._ids]
        self._keep(mask)

    def truco_call(self, bluff_rate):
        if not 0 <= bluff_rate <= 1:
            raise ValueError('Bluff rate must be between zero and one')
        self._summary_cache = None
        if bluff_rate == 1.0:
            return
        # Same two roundings per hand as the scalar loop: 0.1 + strength, then
        # bluff_rate + (1 - bluff_rate) * that, then the in-place product.
        if type(bluff_rate) is np.float64:
            self._scalars = True
        elif isinstance(bluff_rate, np.generic) or not isinstance(bluff_rate, (float, int)):
            # Other scalar types promoted differently in the original loop.
            raise TypeError('Bluff rate must be a Python float or numpy.float64')
        strength = _hand_strength()[self._ids]
        self._weights = self._weights * (bluff_rate + (1 - bluff_rate) * (0.1 + strength))

    def v3_summaries(self, cards, available, *, estimate_outcomes):
        """Compute v3 public summaries and slot outcomes in one posterior pass."""
        weights, ranks, counts, brava, strength = self._summary_arrays()
        outcomes = [0.0, 0.0, 0.0]
        if estimate_outcomes:
            for i in available:
                rank = rules.valor_carta_truco(*cards[i])
                signs = ((rank > ranks).astype(int) - (rank < ranks)) * (ranks >= 0)
                outcomes[i] = _ordered_sum(weights * signs.sum(axis=1) / counts)
        return _ordered_sum(weights * brava), _ordered_sum(weights * strength), outcomes

    def _summary_arrays(self):
        # All evidence and weight mutators invalidate this derived state.
        if self._summary_cache is not None:
            return self._summary_cache
        # Dict iteration preserves the original posterior order, including after filters.
        weights = self._weights / self._total()
        cards = _IDS[self._ids]
        remaining = np.ones(cards.shape, dtype=bool)
        for card in self.played:
            remaining &= cards != _CARD_IDS[card]
        ranks = np.where(remaining, _RANKS[cards], -1)
        counts = np.maximum(remaining.sum(axis=1), 1)
        brava = (_BRAVAS[cards] & remaining).any(axis=1)
        strength = _strength_table()[tuple((ranks + 1).T)]
        self._summary_cache = weights, ranks, counts, brava, strength
        return self._summary_cache

    def summaries(self, card, *, ties_win=False):
        weights, ranks, counts, brava, strength = self._summary_arrays()
        rank = rules.valor_carta_truco(*card)
        wins = ((rank > ranks) | (ties_win & (rank == ranks))) & (ranks >= 0)
        return dict(brava_probability=_ordered_sum(weights * brava),
                    expected_strength=_ordered_sum(weights * strength),
                    next_trick_win_probability=_ordered_sum(weights * wins.sum(axis=1) / counts))



@lru_cache(maxsize=20000)
def _strength(hand):
    return rules.poder_mano_truco([{'valor': v, 'palo': p} for v, p in hand])


@lru_cache(maxsize=20000)
def _tantos(hand):
    return rules.calcular_envido([{'valor': v, 'palo': p} for v, p in hand])


def compact_evidence(obs):
    opponent = 1 - obs['player']
    own = tuple(_CARD_IDS[c] for c in obs['hand'])
    played = tuple(_CARD_IDS[c] for player, c in obs['table'] if player == opponent)
    reveal = obs.get('envido_reveal')
    declared, upper = -1, 34
    if reveal:
        declarations = dict(reveal['declarations'])
        value = declarations[opponent]
        if value == 'son buenas':
            upper = declarations[obs['player']] - int(opponent == obs['mano'])
        else:
            declared = value
    return own, played, declared, upper


@lru_cache(maxsize=1)
def _hand_strength():
    return np.array([_strength(hand) / 24.7 for hand in _HANDS])


@lru_cache(maxsize=1)
def _hand_tantos():
    return np.array([_tantos(hand) for hand in _HANDS])


def compatible_mask(evidence):
    masks = np.ones((len(evidence), len(_HANDS)), dtype=bool)
    tantos = _hand_tantos()
    for row, (own, played, declared, upper) in zip(masks, evidence):
        for card in own:
            row &= ~_MEMBERSHIP[card]
        for card in played:
            row &= _MEMBERSHIP[card]
        if declared >= 0:
            row &= tantos == declared
        row &= tantos <= upper
    return masks


def exact_inference_from_observation(obs):
    belief = Inference(obs['hand'])
    opponent = 1 - obs['player']
    belief.seen_cards(opponent_played=[card for player, card in obs['table'] if player == opponent])
    reveal = obs.get('envido_reveal')
    if reveal:
        declarations = dict(reveal['declarations'])
        value = declarations[opponent]
        if value == 'son buenas':
            belief.son_buenas(declarations[obs['player']], opponent == obs['mano'])
        else:
            belief.declared_tantos(value)
    return belief


def inference_from_observation(obs, bluff_rate):
    belief = exact_inference_from_observation(obs)
    opponent = 1 - obs['player']
    for player, action in obs['history']:
        if player == opponent and action in ('truco', 'retruco', 'vale_cuatro'):
            belief.truco_call(bluff_rate)
    return belief



class IncrementalInference:
    """Keep the exact ordered posterior for decisions within one hand."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.belief = None
        self.hand = None
        self.reveal = None
        self.calls = 0
        self.bluff_rate = None

    def update(self, obs, bluff_rate):
        opponent = 1 - obs['player']
        calls = sum(player == opponent and action in ('truco', 'retruco', 'vale_cuatro')
                    for player, action in obs['history'])
        reveal = obs.get('envido_reveal')
        if (self.belief is None or self.hand != obs['hand'] or
                self.bluff_rate != bluff_rate or calls < self.calls or
                (self.reveal is not None and reveal != self.reveal)):
            self.belief = Inference(obs['hand'])
            self.hand, self.bluff_rate = obs['hand'], bluff_rate
            self.calls, self.reveal = 0, None
        played = [card for player, card in obs['table'] if player == opponent]
        self.belief.seen_cards(opponent_played=played)
        if reveal and self.reveal is None:
            declarations = dict(reveal['declarations'])
            value = declarations[opponent]
            if value == 'son buenas':
                self.belief.son_buenas(declarations[obs['player']], opponent == obs['mano'])
            else:
                self.belief.declared_tantos(value)
            self.reveal = reveal
        for _ in range(calls - self.calls):
            self.belief.truco_call(bluff_rate)
        self.calls = calls
        return self.belief
