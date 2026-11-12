"""Audit the information and decisions of phase 1 network policies."""
import truco_reglas as rules


def card_strength(card):
    """Return the rules-defined Truco rank of a card."""
    return rules.valor_carta_truco(*card)


def cannot_lose_hand(obs):
    """Conservatively detect a secured first trick plus unbeatable remaining cards."""
    player = obs['player']
    if not obs['tricks'] or obs['tricks'][0] != player or not obs['available']:
        return False
    known = set(obs['hand']) | {card for seat, card in obs['table'] if seat != player}
    possible = [(value, suit) for suit in rules.PALOS for value in rules.VALORES
                if (value, suit) not in known]
    if len(obs['table']) % 2 and obs['table'][-1][0] != player:
        possible.append(obs['table'][-1][1])
    return bool(possible) and min(card_strength(obs['hand'][i]) for i in obs['available']) > max(map(card_strength, possible))


def mistake_audit(rows):
    """Average mistake probability over decisions satisfying each exact check."""
    types = ('fold_guaranteed_hand', 'decline_high_envido',
             'accept_known_losing_envido', 'lose_last_trick')
    collected = {name: [] for name in types}
    for obs, probabilities in rows:
        for name, actions in mistake_actions(obs, probabilities).items():
            if actions:
                collected[name].append(sum(probabilities[action] for action in actions))
    return {name: {'opportunities': len(values),
                   'mean_probability': sum(values) / len(values) if values else None}
            for name, values in collected.items()}


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
        opponent_rank = card_strength(obs['table'][-1][1])
        result['lose_last_trick'] = tuple('card_%d' % i for i in obs['available']
                                          if card_strength(obs['hand'][i]) < opponent_rank
                                          and 'card_%d' % i in legal)
    if cannot_lose_hand(obs):
        actions = ('no_quiero', 'fold') if pending.get('kind') == 'truco' else ('fold',)
        result['fold_guaranteed_hand'] = tuple(a for a in actions if a in legal)
    return result


def aliasing(features, facts, radius=0.01):
    """Count observations and connected groups with strategically different peers."""
    import numpy as np
    if len(features) != len(facts) or radius < 0:
        raise ValueError('invalid alias inputs')
    def summarize(limit):
        parents = list(range(len(features)))
        members, pairs, examples = set(), 0, []
        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i
        for i, vector in enumerate(features):
            distances = np.linalg.norm(features[i + 1:] - vector, axis=1)
            for j in np.flatnonzero(distances <= limit) + i + 1:
                j = int(j)
                if facts[i] == facts[j]:
                    continue
                pairs += 1
                members.update((i, j))
                parents[root(i)] = root(j)
                if len(examples) < 5:
                    examples.append({'indices': [i, j], 'distance': float(distances[j-i-1]),
                                     'facts': [facts[i], facts[j]]})
        return {'observations': len(members), 'rate': len(members) / len(features) if len(features) else 0,
                'groups': len({root(i) for i in members}), 'pairs': pairs, 'examples': examples}
    return {'exact': summarize(0), 'near': summarize(radius), 'radius': radius}


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


BANK_SEED = 10**20
DEFAULT_CHECKPOINT = 'E:/Tools/GitHub-Repositories/TP-IA-Truco-training-archive/seeds-phase1-seeds/training_output/phase1_seed1/checkpoint_0070.npz'
CHAMPIONS = ('champions/phase1/reigning_champion.npy',
             'experiments/phase1-seed1/reigning_champion.npy',
             'experiments/phase1-gate53-seed1/reigning_champion.npy')


def bank_seed(index):
    if index < 0 or index >= 10**9:
        raise ValueError('bank index outside reserved range')
    return 2 * (BANK_SEED + index) + 1


def strategic_facts(obs, legal):
    """Return card strengths, legal actions, and envido points for alias checks."""
    return (tuple(card_strength(card) for card in obs['hand']), tuple(sorted(legal)),
            rules.calcular_envido([{'valor': v, 'palo': s} for v, s in obs['hand']]))


def interpret(h1, h2, probe, aliases):
    """Apply fixed thresholds, with missing opportunities treated as unsupported."""
    import math
    ratio, interval = h1.get('error_rate_ratio'), h1.get('error_rate_ratio_ci95')
    face, strength = probe.get('mean_face_value_correlation'), probe.get('mean_strength_correlation')
    h1_supported = bool((ratio is not None and ratio >= 2 and interval and interval[0] >= 2)
                        or (face is not None and strength is not None and face > strength))
    first, other = h2['first'], h2['other']
    p, q = first['best_card_argmax_rate'], other['best_card_argmax_rate']
    n, m = first['opportunities'], other['opportunities']
    h2_supported = False
    if n and m and p is not None and q is not None:
        difference = p - q
        standard_error = math.sqrt(p * (1-p) / n + q * (1-q) / m)
        h2_supported = difference >= .10 and difference - 1.96 * standard_error > 0
    return {'H1': h1_supported, 'H2': h2_supported, 'H3': aliases['near']['rate'] >= .05}


def _prepare_observation(item):
    from phase1_network import encode_observation
    from opponent_profile import OpponentProfile
    obs, legal = item
    features = encode_observation(obs, OpponentProfile())
    chances = per_card_win_probabilities(obs) if any(a.startswith('card_') for a in legal) else {}
    return features, chances


def build_bank(population, *, card_decisions, workers):
    """Use behavior_diversity's match builder and odd training-partition seeds."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from behavior_diversity import _bank_match
    if card_decisions < 1 or workers < 1 or len(population) < 2:
        raise ValueError('counts must be positive and population must have at least two members')
    rng = np.random.default_rng(770184)
    bank, cards, index = [], 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while cards < card_decisions:
            jobs = []
            for _ in range(8):
                a, b = rng.choice(len(population), 2, replace=False)
                jobs.append((population[a], population[b], bank_seed(index)))
                index += 1
            for match in pool.map(_bank_match, jobs):
                for item in match:
                    bank.append(item)
                    cards += any(a.startswith('card_') for a in item[1])
                    if cards >= card_decisions:
                        break
                if cards >= card_decisions:
                    break
    return bank, index


def audit(checkpoint, *, card_decisions=2000, workers=12, members=16, radius=.01,
          champions=None):
    """Build a shared observation bank and audit frozen policies, without training."""
    import hashlib
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    from time import perf_counter
    from behavior_diversity import probabilities
    from phase1_network import ACTIONS
    start = perf_counter()
    if min(card_decisions, workers, members) < 1 or radius < 0:
        raise ValueError('counts must be positive and radius nonnegative')
    root = Path(__file__).resolve().parent.parent
    checkpoint = Path(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as data:
        population = data['population'].copy()
    if len(population) < max(2, members):
        raise ValueError('population is too small')
    indices = np.random.default_rng(770185).choice(len(population), members, replace=False)
    selected = [(str(p), np.load(p, allow_pickle=False)) for p in
                (champions if champions is not None else [root / p for p in CHAMPIONS])]
    selected += [('population_%d' % i, population[i]) for i in indices]
    bank, matches_scheduled = build_bank(population, card_decisions=card_decisions, workers=workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        prepared = list(pool.map(_prepare_observation, bank, chunksize=8))
    features = np.array([item[0] for item in prepared])
    mask = np.array([[a in legal for a in ACTIONS] for _, legal in bank])
    facts = [strategic_facts(obs, legal) for obs, legal in bank]
    aliases = aliasing(features, facts, radius)
    for group in ('exact', 'near'):
        for example in aliases[group]['examples']:
            example['observations'] = [bank[i][0] for i in example['indices']]
    networks = {}
    for name, genome in selected:
        probs = probabilities(genome, features, mask)
        rows = [(obs, {action: float(probs[i, j]) for j, action in enumerate(ACTIONS) if mask[i, j]})
                for i, (obs, _) in enumerate(bank)]
        card_rows = [(obs, p, prepared[i][1]) for i, (obs, p) in enumerate(rows)
                     if prepared[i][1]]
        probes = []
        for i, (obs, legal) in enumerate(bank):
            slots = tuple(slot for slot in obs['available'] if 'card_%d' % slot in legal)
            if len(slots) >= 2:
                probes.extend(preference_probe(genome, features[i], slot=slot, legal_cards=slots) for slot in slots)
        def mean_probe(key):
            values = [v[key] for v in probes if v[key] is not None]
            return sum(values) / len(values) if values else None
        networks[name] = {'sha256': hashlib.sha256(np.asarray(genome, dtype=np.float64).tobytes()).hexdigest(),
                          'h1': card_choice_audit(rows), 'h2': per_card_audit(card_rows),
                          'preference_probe': {'sweeps': len(probes),
                                               'mean_face_value_correlation': mean_probe('face_value_correlation'),
                                               'mean_strength_correlation': mean_probe('strength_correlation')},
                          'mistakes': mistake_audit(rows)}
        report = networks[name]
        report['interpretation'] = interpret(report['h1'], report['h2'], report['preference_probe'], aliases)
    return {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'population_indices': list(map(int, indices)), 'card_decisions': card_decisions,
            'observations': len(bank), 'matches_scheduled': matches_scheduled,
            'bank_seed_namespace': 'odd 2*(10^20 + match_index)+1, index < 10^9',
            'profile': 'fresh default profile', 'inference': 'uniform opponent card, strict win',
            'near_alias_groups': 'connected components; endpoints may exceed radius',
            'aliasing': aliases, 'networks': networks, 'seconds': perf_counter() - start}


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', required=True)
    parser.add_argument('--card-decisions', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--members', type=int, default=16)
    parser.add_argument('--radius', type=float, default=.01)
    args = parser.parse_args(argv)
    if min(args.card_decisions, args.workers, args.members) < 1 or not 0 <= args.radius < float('inf'):
        parser.error('counts must be positive and radius finite and nonnegative')
    try:
        result = audit(args.checkpoint, card_decisions=args.card_decisions, workers=args.workers,
                       members=args.members, radius=args.radius)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
