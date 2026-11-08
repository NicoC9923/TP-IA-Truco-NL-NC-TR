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
    def __init__(self, own_cards):
        deck = [(v, p) for p in rules.PALOS for v in rules.VALORES]
        self.played = set()
        self.weights = {hand: 1.0 for hand in combinations((c for c in deck if c not in own_cards), 3)}
        self._summary_cache = None

    def distribution(self):
        total = sum(self.weights.values())
        return {hand: weight / total for hand, weight in self.weights.items()}

    def declared_tantos(self, value):
        self._summary_cache = None
        self.weights = {h: w for h, w in self.weights.items() if _tantos(h) == value}

    def son_buenas(self, own_tantos, opponent_is_mano):
        self._summary_cache = None
        self.weights = {h: w for h, w in self.weights.items()
                        if _tantos(h) < own_tantos or (not opponent_is_mano and _tantos(h) == own_tantos)}

    def seen_cards(self, excluded=(), opponent_played=()):
        excluded, played = set(excluded), set(opponent_played)
        # Earlier filters already enforce these cards; later evidence and calls
        # only remove hands or change weights, never restore excluded hands.
        if not excluded and played.issubset(self.played):
            return
        self._summary_cache = None
        self.played.update(played)
        self.weights = {h: w for h, w in self.weights.items() if not excluded.intersection(h) and played.issubset(h)}

    def truco_call(self, bluff_rate):
        if not 0 <= bluff_rate <= 1:
            raise ValueError('Bluff rate must be between zero and one')
        self._summary_cache = None
        for hand in self.weights:
            strength = _strength(hand) / 24.7
            self.weights[hand] *= bluff_rate + (1 - bluff_rate) * (0.1 + strength)

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
        ids = np.fromiter((_HAND_IDS[h] for h in self.weights), dtype=np.int64)
        weights = np.fromiter(self.weights.values(), dtype=np.float64)
        weights = weights / sum(self.weights.values())
        cards = _IDS[ids]
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
