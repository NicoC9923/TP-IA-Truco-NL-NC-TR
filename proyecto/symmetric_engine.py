"""Seeded symmetric Truco engine; policies receive only their own cards."""
from dataclasses import dataclass
import random
import truco_reglas as rules

@dataclass
class MatchResult:
    winner: int
    points: tuple
    decisions: list
    deals: list


def play_match(policy_a, policy_b, *, seed=0, target=15, starting_score=(0, 0), swap_deals=False):
    if target <= 0 or len(starting_score) != 2 or min(starting_score) < 0 or max(starting_score) >= target:
        raise ValueError('Invalid target or starting score')
    rng = random.Random(seed)
    score = list(starting_score)
    policies = (policy_a, policy_b)
    decisions, deals = [], []
    mano = 1 if swap_deals else 0
    while max(score) < target:
        deck = [(v, p) for p in rules.PALOS for v in rules.VALORES]
        rng.shuffle(deck)
        hands = [deck[:3], deck[3:6]]
        if swap_deals:
            hands.reverse()
        deals.append(tuple(tuple(h) for h in hands))
        available = [list(range(3)), list(range(3))]
        table, tricks = [], []
        turn = mano
        truco = 1
        raise_owner = None
        pending = None
        suspended = None
        envido_done = False
        hand_done = False
        while not hand_done and max(score) < target:
            can_envido = not envido_done and len(tricks) == 0 and truco == 1
            if pending:
                actor = 1 - pending['caller']
                actions = ['quiero', 'no_quiero']
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
            obs = {'player': actor, 'hand': tuple(hands[actor]), 'available': tuple(available[actor]), 'score': tuple(score), 'target': target, 'mano': mano, 'table': tuple(table), 'tricks': tuple(tricks), 'truco': truco, 'pending': None if pending is None else dict(pending)}
            legal = tuple(actions)
            action = policies[actor](obs, legal)
            if action not in legal:
                raise ValueError('Illegal action: %r' % (action,))
            decisions.append({'player': actor, 'observation': obs, 'legal_actions': legal, 'action': action})
            if action in ('envido', 'real_envido', 'falta_envido'):
                if pending and pending['kind'] == 'truco':
                    suspended, pending = pending, None
                previous = pending['accepted'] if pending else 0
                call = 'envido_envido' if pending and pending['call'] == 'envido' and action == 'envido' else action
                accepted = previous + (2 if action == 'envido' else 3 if action == 'real_envido' else 0)
                pending = {'kind': 'envido', 'caller': actor, 'call': call, 'accepted': accepted, 'declined': max(1, previous)}
            elif action in ('truco', 'retruco', 'vale_cuatro'):
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
                        if pending['call'] == 'falta_envido':
                            points = rules.puntos_falta_envido(score[0], score[1], target, ('humano', 'robot')[winner])[0]
                        else:
                            points = pending['accepted']
                    score[winner] += points
                    envido_done = True
                    pending, suspended = suspended, None
            elif action == 'fold':
                score[1-actor] += 2 if not table and not envido_done and truco == 1 else truco
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
        mano = 1-mano
    return MatchResult(score.index(max(score)), tuple(score), decisions, deals)


def play_mirror_pair(policy_a, policy_b, *, seed=0, target=15, starting_score=(0, 0)):
    return (play_match(policy_a, policy_b, seed=seed, target=target, starting_score=starting_score),
            play_match(policy_b, policy_a, seed=seed, target=target, starting_score=starting_score[::-1], swap_deals=True))

