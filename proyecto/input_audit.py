"""Audit the information and decisions of phase 1 network policies."""
import truco_reglas as rules


def card_strength(card):
    """Return the rules-defined Truco rank of a card."""
    return rules.valor_carta_truco(*card)


def determinable_winners(observation):
    """Return card indices winning the current trick, or None if unknown."""
    table = observation['table']
    if len(table) % 2 != 1 or table[-1][0] == observation['player']:
        return None
    opponent_rank = card_strength(table[-1][1])
    return tuple(i for i in observation['available']
                 if card_strength(observation['hand'][i]) > opponent_rank)


def mistake_actions(obs, legal):
    """Identify actions provably mistaken under the documented narrow checks."""
    result = {}
    pending = obs.get('pending') or {}
    points = rules.calcular_envido([{'valor': v, 'palo': s} for v, s in obs['hand']])
    if pending.get('kind') == 'envido' and points >= 31:
        result['decline_high_envido'] = tuple(a for a in ('no_quiero', 'fold') if a in legal)
    reveal = obs.get('envido_reveal')
    if pending.get('kind') == 'envido' and reveal:
        other = dict(reveal['declarations']).get(1 - obs['player'])
        if isinstance(other, int) and other > points and 'quiero' in legal:
            result['accept_known_losing_envido'] = ('quiero',)
    winning = determinable_winners(obs)
    if len(obs['tricks']) == 2 and winning:
        result['lose_last_trick'] = tuple('card_%d' % i for i in obs['available']
                                          if i not in winning and 'card_%d' % i in legal)
    return result


def aliasing(features, facts, radius=0.01):
    """Count observations with a strategically different exact or nearby peer."""
    import numpy as np
    from collections import defaultdict
    exact = defaultdict(list)
    for i, vector in enumerate(features):
        exact[tuple(vector)].append(i)
    exact_pairs = [(g[i], g[j]) for g in exact.values() for i in range(len(g))
                   for j in range(i + 1, len(g)) if facts[g[i]] != facts[g[j]]]
    near_pairs = []
    for i, vector in enumerate(features):
        distance = np.linalg.norm(features[i + 1:] - vector, axis=1)
        near_pairs.extend((i, i + 1 + int(j)) for j in np.flatnonzero(distance <= radius)
                          if facts[i] != facts[i + 1 + int(j)])
    def summarize(pairs):
        members = {i for pair in pairs for i in pair}
        return {'observations': len(members), 'rate': len(members) / len(features) if len(features) else 0,
                'pairs': len(pairs), 'examples': [list(pair) for pair in pairs[:5]]}
    return {'exact': summarize(exact_pairs), 'near': summarize(near_pairs)}
