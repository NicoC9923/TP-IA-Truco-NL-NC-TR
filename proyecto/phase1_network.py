"""Fixed-topology phase 1 policy and observation encoder."""
import numpy as np
import truco_reglas as rules
from opponent_inference import inference_from_observation, IncrementalInference
from opponent_profile import OpponentProfile

ACTIONS = ('card_0', 'card_1', 'card_2', 'envido', 'real_envido', 'falta_envido', 'quiero', 'no_quiero', 'truco', 'retruco', 'vale_cuatro', 'fold')
INPUTS, HIDDEN = 64, 24
GENOME_SIZE = INPUTS * HIDDEN + HIDDEN + HIDDEN * len(ACTIONS) + len(ACTIONS)

def encode_observation(obs, profile, inference=None):
    values = []
    for i, (value, suit) in enumerate(obs['hand']):
        values.extend((value / 12, rules.PALOS.index(suit) / 3, float(i in obs['available']), min(value, 10) % 10 / 7))
    for i in range(6):
        if i < len(obs['table']):
            player, (value, suit) = obs['table'][i]
            values.extend((value / 12, rules.PALOS.index(suit) / 3, 1 if player == obs['player'] else -1))
        else:
            values.extend((0, 0, 0))
    values.extend(0 if winner is None else (1 if winner == obs['player'] else -1) for winner in obs['tricks'])
    values.extend([0] * (3 - len(obs['tricks'])))
    pending = obs['pending']
    values.extend((obs['truco'] / 4, float(obs['envido_done']), float(bool(pending and pending['kind'] == 'envido')), float(bool(pending and pending['kind'] == 'truco')), float(obs['mano'] == obs['player']), float(obs['raise_owner'] == obs['player'])))
    player = obs['player']
    values.extend(((obs['target'] - obs['score'][player]) / 30, (obs['target'] - obs['score'][1-player]) / 30, rules.puntos_falta_envido(*obs['score'], obs['target'], ('humano', 'robot')[player])[0] / 30))
    for action in ('envido', 'real_envido', 'falta_envido', 'truco', 'retruco', 'vale_cuatro'):
        values.append(sum(1 if actor == player else -1 for actor, call in obs['history'] if call == action))
    reveal = obs.get('envido_reveal')
    declarations = dict(reveal['declarations']) if reveal else {}
    for seat in (player, 1-player):
        value = declarations.get(seat)
        values.append(value / 33 if isinstance(value, int) else -1 if value == 'son buenas' else 0)
    rates = profile.rates()
    belief = (inference.update(obs, rates['bluff']) if inference is not None
              else inference_from_observation(obs, rates['bluff']))
    card = obs['hand'][obs['available'][0]] if obs['available'] else obs['hand'][0]
    summaries = belief.summaries(card, ties_win=obs['mano'] == player)
    values.extend((summaries['brava_probability'], summaries['expected_strength'] / 24.7, summaries['next_trick_win_probability']))
    values.extend(rates.values())
    values.extend([0] * (INPUTS - len(values)))
    return np.asarray(values, dtype=np.float64)

class NetworkPolicy:
    def __init__(self, genome, seed=0, bias=None, profile=None, style=None, style_strength=1.0, explain=False):
        self.genome = np.asarray(genome, dtype=np.float64).copy()
        if self.genome.shape != (GENOME_SIZE,):
            raise ValueError('Invalid genome shape')
        self.rng = np.random.default_rng(seed)
        self.bias = dict(bias or {})
        self.profile = profile or OpponentProfile()
        self.style = style
        self.style_strength = style_strength
        self.explain = explain
        self.inference = IncrementalInference()

    def __call__(self, observation, legal_actions):
        x = encode_observation(observation, self.profile, self.inference)
        a = INPUTS * HIDDEN
        b = a + HIDDEN
        c = b + HIDDEN * len(ACTIONS)
        hidden = np.tanh(x @ self.genome[:a].reshape(INPUTS, HIDDEN) + self.genome[a:b])
        logits = hidden @ self.genome[b:c].reshape(HIDDEN, len(ACTIONS)) + self.genome[c:]
        from symmetric_engine import _bluff
        scores = []
        for action in legal_actions:
            adjustment = self.bias.get(action, 0)
            if self.style in ('bluffer', 'calculating') and _bluff(action, observation['hand']):
                adjustment += 3 * self.style_strength if self.style == 'bluffer' else -3 * self.style_strength
            if self.style == 'cautious' and action == 'quiero':
                adjustment -= 3 * self.style_strength
            if self.style == 'aggressive' and action in ('retruco', 'vale_cuatro', 'real_envido', 'falta_envido'):
                adjustment += 3 * self.style_strength
            scores.append(logits[ACTIONS.index(action)] + adjustment)
        scores = np.array(scores)
        weights = np.exp(scores - scores.max())
        probabilities = weights / weights.sum()
        if self.explain:
            self.last_probabilities = dict(zip(legal_actions, map(float, probabilities)))
            card = observation['hand'][observation['available'][0] if observation['available'] else 0]
            self.last_belief = self.inference.belief.summaries(card, ties_win=observation['mano'] == observation['player'])
        return legal_actions[self.rng.choice(len(legal_actions), p=probabilities)]

    def observe_hand(self, decisions, opponent):
        self.profile.observe(decisions, opponent)
        self.inference.reset()

    def save(self, path):
        np.save(path, self.genome)

    @classmethod
    def load(cls, path, **kwargs):
        return cls(np.load(path, allow_pickle=False), **kwargs)


