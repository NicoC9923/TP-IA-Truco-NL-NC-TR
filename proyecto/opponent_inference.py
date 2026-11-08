"""Public Bayesian inference over the opponent's original three cards."""
from itertools import combinations
from functools import lru_cache
import truco_reglas as rules

class Inference:
    def __init__(self, own_cards):
        deck = [(v, p) for p in rules.PALOS for v in rules.VALORES]
        self.played = set()
        self.weights = {hand: 1.0 for hand in combinations((c for c in deck if c not in own_cards), 3)}

    def distribution(self):
        total = sum(self.weights.values())
        return {hand: weight / total for hand, weight in self.weights.items()}

    def declared_tantos(self, value):
        self.weights = {h: w for h, w in self.weights.items() if _tantos(h) == value}

    def son_buenas(self, own_tantos, opponent_is_mano):
        self.weights = {h: w for h, w in self.weights.items()
                        if _tantos(h) < own_tantos or (not opponent_is_mano and _tantos(h) == own_tantos)}

    def seen_cards(self, excluded=(), opponent_played=()):
        excluded, played = set(excluded), set(opponent_played)
        self.played.update(played)
        self.weights = {h: w for h, w in self.weights.items() if not excluded.intersection(h) and played.issubset(h)}

    def truco_call(self, bluff_rate):
        if not 0 <= bluff_rate <= 1:
            raise ValueError('Bluff rate must be between zero and one')
        for hand in self.weights:
            strength = _strength(hand) / 24.7
            self.weights[hand] *= bluff_rate + (1 - bluff_rate) * (0.1 + strength)

    def v3_summaries(self, cards, available, *, estimate_outcomes):
        """Compute v3 public summaries and slot outcomes in one posterior pass."""
        distribution = self.distribution()
        bravas = {(1, 'Espada'), (1, 'Basto'), (7, 'Espada'), (7, 'Oro')}
        ranks = [rules.valor_carta_truco(*card) for card in cards]
        brava_probability = 0.0
        expected_strength = 0.0
        outcomes = [0.0, 0.0, 0.0]
        for hand, weight in distribution.items():
            remaining = [c for c in hand if c not in self.played]
            brava_probability += weight * bool(bravas.intersection(remaining))
            expected_strength += weight * _strength(tuple(remaining))
            if estimate_outcomes and remaining:
                other_ranks = [rules.valor_carta_truco(*c) for c in remaining]
                for i in available:
                    rank = ranks[i]
                    outcomes[i] += weight * sum((rank > other) - (rank < other)
                                                for other in other_ranks) / len(remaining)
        return brava_probability, expected_strength, outcomes

    def summaries(self, card, *, ties_win=False):
        distribution = self.distribution()
        rank = rules.valor_carta_truco(*card)
        bravas = {(1, 'Espada'), (1, 'Basto'), (7, 'Espada'), (7, 'Oro')}
        result = dict(brava_probability=0.0, expected_strength=0.0, next_trick_win_probability=0.0)
        for hand, weight in distribution.items():
            remaining = [c for c in hand if c not in self.played]
            result['brava_probability'] += weight * bool(bravas.intersection(remaining))
            result['expected_strength'] += weight * _strength(tuple(remaining))
            if remaining:
                # Opponent plays each remaining card with equal probability.
                result['next_trick_win_probability'] += weight * sum(rank > rules.valor_carta_truco(*c) or (ties_win and rank == rules.valor_carta_truco(*c)) for c in remaining) / len(remaining)
        return result


@lru_cache(maxsize=20000)
def _strength(hand):
    return rules.poder_mano_truco([{'valor': v, 'palo': p} for v, p in hand])


@lru_cache(maxsize=20000)
def _tantos(hand):
    return rules.calcular_envido([{'valor': v, 'palo': p} for v, p in hand])


def inference_from_observation(obs, bluff_rate):
    belief = Inference(obs['hand'])
    opponent = 1 - obs['player']
    played = [card for player, card in obs['table'] if player == opponent]
    belief.seen_cards(opponent_played=played)
    reveal = obs.get('envido_reveal')
    if reveal:
        declarations = dict(reveal['declarations'])
        value = declarations[opponent]
        if value == 'son buenas':
            belief.son_buenas(declarations[obs['player']], opponent == obs['mano'])
        else:
            belief.declared_tantos(value)
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
