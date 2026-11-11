"""Seeded symmetric Truco engine; policies receive only their own cards."""
from dataclasses import dataclass
import random
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import truco_reglas as rules

# Version 2 records acceptance when raising a pending truco.
ENGINE_VERSION = 2

@lru_cache(maxsize=1)
def _percentiles():
    with Path(__file__).with_name('truco_stats.json').open(encoding='utf-8') as file:
        return {float(k): float(v) for k, v in json.load(file)['truco_poder_percentil'].items()}


def bluff(action, hand):
    cards = [{'valor': v, 'palo': p} for v, p in hand]
    if action in ('envido', 'real_envido', 'falta_envido'):
        return rules.calcular_envido(cards) < 24
    if action in ('truco', 'retruco', 'vale_cuatro'):
        power = round(min(23.5, max(0.0, rules.poder_mano_truco(cards))), 1)
        percentiles = _percentiles()
        return percentiles[min(percentiles, key=lambda k: abs(k - power))] < 0.44
    return None

_bluff = bluff  # Backwards-compatible alias.

@dataclass
class MatchResult:
    winner: int
    points: tuple
    decisions: list
    deals: list


def play_match(policy_a, policy_b, *, seed=0, target=15, starting_score=(0, 0), deal_partition=None,
               max_hands=None, on_event=None):
    """Play one match. It ends when a player reaches `target`, or, with `max_hands`,
    after that many hands as soon as the points are not level (a short match: extra
    hands are played one at a time until they are). `on_event`, if given, is called with
    a dict at each deal and at each hand end; it never changes the play."""
    if deal_partition not in (None, 0, 1):
        raise ValueError("Invalid deal partition")
    if max_hands is not None and (isinstance(max_hands, bool) or not isinstance(max_hands, int) or max_hands < 1):
        raise ValueError('Invalid max_hands')
    if target <= 0 or len(starting_score) != 2 or min(starting_score) < 0 or max(starting_score) >= target:
        raise ValueError('Invalid target or starting score')
    rng = random.Random(seed)
    score = list(starting_score)
    policies = (policy_a, policy_b)
    decisions, deals = [], []
    mano = 0
    while max(score) < target and not (max_hands is not None and len(deals) >= max_hands and score[0] != score[1]):
        deck = [(v, p) for p in rules.PALOS for v in rules.VALORES]
        while True:
            rng.shuffle(deck)
            hands = [deck[:3], deck[3:6]]
            canonical = tuple(sorted(tuple(sorted(hand)) for hand in hands))
            if deal_partition is None or hashlib.sha256(repr(canonical).encode('utf-8')).digest()[0] % 2 == deal_partition:
                break
        deals.append(tuple(tuple(h) for h in hands))
        if on_event is not None:
            on_event({'kind': 'deal', 'hand_index': len(deals) - 1, 'hands': deals[-1], 'mano': mano, 'score': tuple(score)})
        available = [list(range(3)), list(range(3))]
        table, tricks, history = [], [], []
        turn = mano
        truco = 1
        raise_owner = None
        pending = None
        suspended = None
        envido_done = False
        envido_reveal = None
        hand_start = len(decisions)
        hand_done = False
        while not hand_done and max(score) < target:
            can_envido = not envido_done and len(tricks) == 0 and truco == 1
            if pending:
                actor = 1 - pending['caller']
                actions = ['quiero', 'no_quiero', 'fold']
                if pending['kind'] == 'truco':
                    if pending['level'] < 4:
                        actions.append(('retruco', 'vale_cuatro')[pending['level']-2])
                    if can_envido:
                        actions.extend(['envido', 'real_envido', 'falta_envido'])
                else:
                    actions.extend({'envido': ['envido', 'real_envido', 'falta_envido'], 'envido_envido': ['real_envido', 'falta_envido'], 'real_envido': ['falta_envido'], 'falta_envido': []}[pending['call']])
            else:
                actor = turn
                actions = ['card_%d' % i for i in available[actor]] + ['fold']
                if can_envido:
                    actions += ['envido', 'real_envido', 'falta_envido']
                if truco < 4 and (raise_owner is None or raise_owner == actor):
                    actions.append(('truco', 'retruco', 'vale_cuatro')[truco-1])
            obs = {'player': actor, 'hand': tuple(hands[actor]), 'available': tuple(available[actor]), 'score': tuple(score), 'target': target, 'mano': mano, 'table': tuple(table), 'tricks': tuple(tricks), 'truco': truco, 'pending': None if pending is None else dict(pending), 'history': tuple(history), 'envido_done': envido_done, 'envido_reveal': envido_reveal, 'raise_owner': raise_owner}
            legal = tuple(actions)
            action = policies[actor](obs, legal)
            if action not in legal:
                raise ValueError('Illegal action: %r' % (action,))
            history.append((actor, action))
            decisions.append({'player': actor, 'observation': obs, 'legal_actions': legal, 'action': action, 'bluff': _bluff(action, hands[actor])})
            if action in ('envido', 'real_envido', 'falta_envido'):
                if pending and pending['kind'] == 'truco':
                    suspended, pending = pending, None
                previous = pending['accepted'] if pending else 0
                call = 'envido_envido' if pending and pending['call'] == 'envido' and action == 'envido' else action
                accepted = previous + (2 if action == 'envido' else 3 if action == 'real_envido' else 0)
                pending = {'kind': 'envido', 'caller': actor, 'call': call, 'accepted': accepted, 'declined': max(1, previous)}
            elif action in ('truco', 'retruco', 'vale_cuatro'):
                if pending and pending['kind'] == 'truco':
                    truco = pending['level']
                    raise_owner = actor
                level = {'truco': 2, 'retruco': 3, 'vale_cuatro': 4}[action]
                pending = {'kind': 'truco', 'caller': actor, 'level': level}
            elif action in ('quiero', 'no_quiero'):
                if pending['kind'] == 'truco':
                    if action == 'no_quiero':
                        score[pending['caller']] += pending['level'] - 1
                        hand_done = True
                    else:
                        truco = pending['level']
                        raise_owner = actor
                    pending = None
                else:
                    if action == 'no_quiero':
                        winner, points = pending['caller'], pending['declined']
                    else:
                        values = [rules.calcular_envido([{'valor': v, 'palo': p} for v, p in hand]) for hand in hands]
                        winner = mano if values[0] == values[1] else values.index(max(values))
                        pie = 1 - mano
                        declarations = [(mano, values[mano])]
                        declarations.append((pie, values[pie] if values[pie] > values[mano] else 'son buenas'))
                        envido_reveal = {'declarations': tuple(declarations)}
                        if pending['call'] == 'falta_envido':
                            points = rules.puntos_falta_envido(score[0], score[1], target, ('humano', 'robot')[winner])[0]
                        else:
                            points = pending['accepted']
                    score[winner] += points
                    envido_done = True
                    pending, suspended = suspended, None
            elif action == 'fold':
                score[1-actor] += (pending['level'] - 1 if pending and pending['kind'] == 'truco' else truco + pending['declined'] if pending and pending['kind'] == 'envido' else 2 if not table and not envido_done and truco == 1 else truco)
                hand_done = True
            else:
                index = int(action[-1])
                available[actor].remove(index)
                table.append((actor, hands[actor][index]))
                if len(table) % 2:
                    turn = 1-actor
                else:
                    first, second = table[-2:]
                    powers = [rules.valor_carta_truco(*entry[1]) for entry in (first, second)]
                    winner = None if powers[0] == powers[1] else (first, second)[powers.index(max(powers))][0]
                    tricks.append(winner)
                    envido_done = True
                    hand_winner = None
                    if tricks.count(0) == 2 or tricks.count(1) == 2:
                        hand_winner = 0 if tricks.count(0) == 2 else 1
                    elif len(tricks) >= 2:
                        if tricks[0] is None and tricks[1] is not None:
                            hand_winner = tricks[1]
                        elif tricks[0] is not None and tricks[1] is None:
                            hand_winner = tricks[0]
                        elif len(tricks) == 3:
                            hand_winner = tricks[2] if tricks[2] is not None else (tricks[0] if tricks[0] is not None else mano)
                    if hand_winner is not None:
                        score[hand_winner] += truco
                        hand_done = True
                    else:
                        turn = winner if winner is not None else (tricks[0] if tricks[0] is not None else mano)
        for decision in decisions[hand_start:]:
            decision['hand_index'] = len(deals) - 1
            decision['revealed'] = len(available[decision['player']]) == 0
        for seat, policy in enumerate(policies):
            observer = getattr(policy, 'observe_hand', None)
            if observer is not None:
                observer([{'player': d['player'], 'action': d['action'],
                           'legal_actions': d['legal_actions'],
                           'observation': {'pending': d['observation']['pending']},
                           'hand_index': d['hand_index'], 'revealed': d['revealed'],
                           'bluff': d['bluff'] if d['revealed'] else None}
                          for d in decisions[hand_start:]], 1-seat)
        if on_event is not None:
            on_event({'kind': 'hand_end', 'hand_index': len(deals) - 1, 'mano': mano, 'score': tuple(score)})
        mano = 1-mano
    return MatchResult(score.index(max(score)), tuple(score), decisions, deals)


def play_mirror_pair(policy_a, policy_b, *, seed=0, target=15, starting_score=(0, 0), deal_partition=None):
    return (play_match(policy_a, policy_b, seed=seed, target=target, starting_score=starting_score, deal_partition=deal_partition),
            play_match(policy_b, policy_a, seed=seed, target=target, starting_score=starting_score[::-1], deal_partition=deal_partition))







