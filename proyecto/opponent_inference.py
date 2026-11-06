"""Public Bayesian inference over the opponent's original three cards."""
from itertools import combinations
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
            strength = rules.poder_mano_truco([{'valor': v, 'palo': p} for v, p in hand]) / 24.7
            self.weights[hand] *= bluff_rate + (1 - bluff_rate) * (0.1 + strength)

    def summaries(self, card, *, ties_win=False):
        distribution = self.distribution()
        rank = rules.valor_carta_truco(*card)
        bravas = {(1, 'Espada'), (1, 'Basto'), (7, 'Espada'), (7, 'Oro')}
        result = dict(brava_probability=0.0, expected_strength=0.0, next_trick_win_probability=0.0)
        for hand, weight in distribution.items():
            remaining = [c for c in hand if c not in self.played]
            result['brava_probability'] += weight * bool(bravas.intersection(remaining))
            result['expected_strength'] += weight * rules.poder_mano_truco([{'valor': v, 'palo': p} for v, p in remaining])
            if remaining:
                # Opponent plays each remaining card with equal probability.
                result['next_trick_win_probability'] += weight * sum(rank > rules.valor_carta_truco(*c) or (ties_win and rank == rules.valor_carta_truco(*c)) for c in remaining) / len(remaining)
        return result


def _tantos(hand):
    return rules.calcular_envido([{'valor': v, 'palo': p} for v, p in hand])
