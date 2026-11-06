"""Smoothed opponent statistics computed from observed decisions."""

class OpponentProfile:
    def __init__(self):
        self.counts = {name: [0, 0] for name in ('bluff', 'fold', 'truco_accepted', 'envido_called')}

    def observe(self, decisions, opponent):
        hands = {}
        for index, decision in enumerate(decisions):
            if decision['player'] == opponent:
                hands.setdefault(decision.get('hand_index', ('event', index)), []).append(decision)
        for hand in hands.values():
            actions = [d['action'] for d in hand]
            self._record('fold', 'fold' in actions)
            self._record('envido_called', any(a in ('envido', 'real_envido', 'falta_envido') for a in actions))
            for decision in hand:
                pending = decision['observation'].get('pending')
                if pending and pending['kind'] == 'truco':
                    self._record('truco_accepted', decision['action'] in ('quiero', 'retruco', 'vale_cuatro'))
            calls = [d for d in hand if d.get('bluff') is not None]
            if calls and all(d.get('revealed') for d in calls):
                self._record('bluff', any(d['bluff'] for d in calls))

    def _record(self, name, success):
        self.counts[name][0] += int(success)
        self.counts[name][1] += 1

    def rates(self, adapt=True):
        return {name: (10 + successes) / (20 + total) if adapt else 0.5
                for name, (successes, total) in self.counts.items()}
