"""Seeded two-player Truco match engine."""
from dataclasses import dataclass
import random
import truco_reglas as rules

@dataclass
class MatchResult:
    winner: int
    points: tuple
    decisions: list
    deals: list


def play_match(policy_a, policy_b, *, seed=0, target=15, starting_score=(0, 0)):
    rng = random.Random(seed)
    score = list(starting_score)
    policies = (policy_a, policy_b)
    decisions, deals = [], []
    mano = 0
    while max(score) < target:
        deck = [(v, p) for p in rules.PALOS for v in rules.VALORES]
        rng.shuffle(deck)
        hands = [deck[:3], deck[3:6]]
        deals.append(tuple(tuple(h) for h in hands))
        actor = mano
        obs = {'player': actor, 'hand': tuple(hands[actor]), 'score': tuple(score)}
        actions = ('fold',)
        action = policies[actor](obs, actions)
        if action not in actions:
            raise ValueError('Illegal action: %r' % (action,))
        decisions.append({'player': actor, 'observation': obs, 'action': action})
        score[1-actor] += 2
        mano = 1-mano
    return MatchResult(score.index(max(score)), tuple(score), decisions, deals)
