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


def card_choice_audit(rows, *, bootstrap_samples=2000, seed=73141):
    """Summarize legal-action probability mass and pairwise trick errors.

    Rows contain (observation, action-probability mapping). Pairwise errors
    compare a winning card against a losing card, conditional on choosing
    either; bootstrap resamples decisions, preserving within-decision pairs.
    """
    import itertools
    import numpy as np
    wins, cheap, costly = [], [], []
    pairs = {'agree': [], 'disagree': []}
    for obs, probs in rows:
        winners = determinable_winners(obs)
        if not winners:
            continue
        available = [i for i in obs['available'] if 'card_%d' % i in probs]
        winners = [i for i in winners if i in available]
        if not winners:
            continue
        minimum = min(card_strength(obs['hand'][i]) for i in winners)
        wins.append(sum(probs['card_%d' % i] for i in winners))
        cheap.append(sum(probs['card_%d' % i] for i in winners if card_strength(obs['hand'][i]) == minimum))
        costly.append(wins[-1] - cheap[-1])
        per_decision = {'agree': [], 'disagree': []}
        for i, j in itertools.combinations(available, 2):
            if (i in winners) == (j in winners):
                continue
            winner, loser = (i, j) if i in winners else (j, i)
            if obs['hand'][winner][0] == obs['hand'][loser][0]:
                continue
            category = 'agree' if obs['hand'][winner][0] > obs['hand'][loser][0] else 'disagree'
            wp, lp = probs['card_%d' % winner], probs['card_%d' % loser]
            if wp + lp > 0:
                per_decision[category].append(lp / (wp + lp))
        for category in pairs:
            pairs[category].append(per_decision[category])
    def average(values):
        return float(np.mean(values)) if values else None
    summary = {}
    for category, grouped in pairs.items():
        values = [v for group in grouped for v in group]
        summary[category] = {'pairs': len(values), 'error_rate': average(values)}
    ratio = None
    if summary['agree']['error_rate'] and summary['disagree']['error_rate'] is not None:
        ratio = summary['disagree']['error_rate'] / summary['agree']['error_rate']
    ratios = []
    rng = np.random.default_rng(seed)
    if wins:
        for _ in range(bootstrap_samples):
            indices = rng.integers(0, len(wins), len(wins))
            means = [average([v for i in indices for v in pairs[c][i]]) for c in ('agree', 'disagree')]
            if means[0] and means[1] is not None:
                ratios.append(means[1] / means[0])
    return {'opportunities': len(wins), 'winning_mass': average(wins),
            'cheapest_winning_mass': average(cheap), 'stronger_winning_mass': average(costly),
            'pairwise': summary, 'error_rate_ratio': ratio,
            'error_rate_ratio_ci95': list(map(float, np.percentile(ratios, [2.5, 97.5]))) if ratios else None,
            'bootstrap_unit': 'decision'}


def per_card_win_probabilities(obs, bluff_rate=0.5):
    """Return strict trick-win chances, exact against a played card or inferred."""
    winners = determinable_winners(obs)
    if winners is not None:
        return {i: float(i in winners) for i in obs['available']}
    from opponent_inference import inference_from_observation
    belief = inference_from_observation(obs, bluff_rate)
    return {i: belief.summaries(obs['hand'][i], ties_win=False)['next_trick_win_probability']
            for i in obs['available']}


def per_card_audit(rows, *, gap=0.3):
    """Compare card-restricted policy preferences for clearly better cards.

    Rows contain observation, action probability mapping, and card win chances.
    A unique best card must beat the next best by at least gap. Probabilities
    are conditional on choosing a card; argmax ties split credit evenly.
    """
    groups = {'first': [], 'other': []}
    for obs, probs, chances in rows:
        available = [i for i in obs['available'] if 'card_%d' % i in probs and i in chances]
        if len(available) < 2:
            continue
        ordered = sorted(available, key=lambda i: chances[i], reverse=True)
        if chances[ordered[0]] - chances[ordered[1]] < gap - 1e-12:
            continue
        total = sum(probs['card_%d' % i] for i in available)
        if total <= 0:
            continue
        top = max(probs['card_%d' % i] for i in available)
        tied = [i for i in available if probs['card_%d' % i] == top]
        best = ordered[0]
        group = 'first' if best == obs['available'][0] else 'other'
        groups[group].append((float(best in tied) / len(tied), probs['card_%d' % best] / total))
    return {group: {'opportunities': len(items),
                    'best_card_argmax_rate': sum(x[0] for x in items) / len(items) if items else None,
                    'best_card_mass': sum(x[1] for x in items) / len(items) if items else None}
            for group, items in groups.items()}


def preference_probe(genome, features, *, slot, legal_cards=(0, 1, 2)):
    """Correlate card-choice probability with rank across a one-slot deck sweep.

    Other encoded inputs, including inference, remain fixed; this is an
    artificial intervention and may generate impossible duplicate cards.
    """
    import numpy as np
    from phase1_network import ACTIONS, INPUTS, HIDDEN
    if slot not in legal_cards or not 0 <= slot < 3:
        raise ValueError('slot must be a legal card slot')
    cards = [(v, suit) for suit in rules.PALOS for v in rules.VALORES]
    x = np.repeat(np.asarray(features, dtype=float)[None, :], len(cards), axis=0)
    for row, (value, suit) in enumerate(cards):
        x[row, 4 * slot] = value / 12
        x[row, 4 * slot + 1] = rules.PALOS.index(suit) / 3
        x[row, 4 * slot + 3] = min(value, 10) % 10 / 7
    a = INPUTS * HIDDEN
    b = a + HIDDEN
    c = b + HIDDEN * len(ACTIONS)
    hidden = np.tanh(x @ genome[:a].reshape(INPUTS, HIDDEN) + genome[a:b])
    logits = hidden @ genome[b:c].reshape(HIDDEN, len(ACTIONS)) + genome[c:]
    logits = logits[:, list(legal_cards)]
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    preference = weights[:, list(legal_cards).index(slot)] / weights.sum(axis=1)
    def correlation(values):
        if np.std(preference) < 1e-15 or np.std(values) < 1e-15:
            return None
        return float(np.corrcoef(preference, values)[0, 1])
    return {'face_value_correlation': correlation([card[0] for card in cards]),
            'strength_correlation': correlation([card_strength(card) for card in cards]),
            'cards': len(cards)}
