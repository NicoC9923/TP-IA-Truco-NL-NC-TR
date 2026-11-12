"""Run 3 blended per-decision target (ADR 0062).

target = (1 - blend) * match result + blend * (2 * equity - 1), both parts in [-1, 1].
``result`` is +1 for a win and -1 for a loss from the acting seat. ``equity`` is the match equity
(phase2_equity) of the state after the decision's hand has been scored, read at the first decision
of the next hand: its scores already include the whole hand, so no points are counted twice, and its
mano is the next hand's mano. The last hand of a match, including one ended by an accepted falta
envido, has no next state: its target is the match result.
"""


def decision_targets(result, seat, table, blend):
    """Targets for ``seat``'s decisions of a MatchResult, in the order they were made."""
    if not 0 <= blend <= 1:
        raise ValueError('blend must be between 0 and 1')
    outcome = 1.0 if result.winner == seat else -1.0
    starts = {}
    for decision in result.decisions:
        starts.setdefault(decision['hand_index'], decision['observation'])
    targets = []
    for decision in result.decisions:
        if decision['player'] != seat:
            continue
        following = starts.get(decision['hand_index'] + 1)
        if following is None:
            targets.append(outcome)
            continue
        own, opp = following['score'][seat], following['score'][1 - seat]
        value = table.equity(own, opp, following['target'], following['mano'] == seat)
        targets.append((1 - blend) * outcome + blend * (2 * value - 1))
    return targets
