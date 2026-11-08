"""Smoothed opponent statistics computed from observed decisions."""
import json
from pathlib import Path

TRUCO_RESOLUTION_ACTIONS = frozenset(('quiero', 'no_quiero', 'fold', 'retruco', 'vale_cuatro'))

class OpponentProfile:
    def __init__(self, path=None, *, prior_strength=20):
        if prior_strength <= 0:
            raise ValueError('prior_strength must be positive')
        self.prior_strength = prior_strength
        self.path = Path(path) if path is not None else None
        self.counts = {name: [0, 0] for name in ('bluff', 'fold', 'truco_accepted', 'envido_called')}
        if self.path is not None and self.path.is_file():
            data = json.loads(self.path.read_text(encoding='utf-8'))
            for name in self.counts:
                successes, total = data[name]
                if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
                    raise ValueError('Invalid profile counts')
                self.counts[name] = [successes, total]

    def _save(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + '.tmp')
            temporary.write_text(json.dumps(self.counts), encoding='utf-8')
            temporary.replace(self.path)

    def reset(self):
        self.counts = {name: [0, 0] for name in self.counts}
        self._save()

    def observe(self, decisions, opponent, *, semantics="legacy"):
        # The former v2 label remains an alias for the v3 resolution semantics.
        hands = {}
        for index, decision in enumerate(decisions):
            if decision['player'] == opponent:
                hands.setdefault(decision.get('hand_index', ('event', index)), []).append(decision)
        for hand in hands.values():
            actions = [d['action'] for d in hand]
            if semantics == 'phase2':
                for decision in hand:
                    pending = decision['observation'].get('pending')
                    action = decision['action']
                    if pending and pending['kind'] == 'truco' and action in TRUCO_RESOLUTION_ACTIONS:
                        self._record('fold', action == 'fold')
                envido_actions = ('envido', 'real_envido', 'falta_envido')
                if any(any(a in d['legal_actions'] for a in envido_actions) for d in hand):
                    self._record('envido_called', any(a in envido_actions for a in actions))
            else:
                self._record('fold', 'fold' in actions)
                self._record('envido_called', any(a in ('envido', 'real_envido', 'falta_envido') for a in actions))
            for decision in hand:
                pending = decision['observation'].get('pending')
                if (pending and pending['kind'] == 'truco' and
                        (semantics == 'legacy' or decision['action'] in TRUCO_RESOLUTION_ACTIONS)):
                    self._record('truco_accepted', decision['action'] in ('quiero', 'retruco', 'vale_cuatro'))
            calls = [d for d in hand if d.get('bluff') is not None]
            if calls and all(d.get('revealed') for d in calls):
                self._record('bluff', any(d['bluff'] for d in calls))
        self._save()

    def _record(self, name, success):
        self.counts[name][0] += int(success)
        self.counts[name][1] += 1

    def confidence(self):
        return {name: total for name, (_, total) in self.counts.items()}

    def rates(self, adapt=True):
        return {name: (self.prior_strength * 0.5 + successes) / (self.prior_strength + total) if adapt else 0.5
                for name, (successes, total) in self.counts.items()}

