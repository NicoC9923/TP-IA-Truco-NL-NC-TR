"""Fixed-topology phase 1 policy and observation encoder."""
import numpy as np
import truco_reglas as rules
from opponent_inference import inference_from_observation
from opponent_profile import OpponentProfile

ACTIONS = ('card_0', 'card_1', 'card_2', 'envido', 'real_envido', 'falta_envido', 'quiero', 'no_quiero', 'truco', 'retruco', 'vale_cuatro', 'fold')
INPUTS, HIDDEN = 64, 24
GENOME_SIZE = INPUTS * HIDDEN + HIDDEN + HIDDEN * len(ACTIONS) + len(ACTIONS)

def encode_observation(obs, profile):
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
    values.extend(((obs['target'] - obs['score'][player]) / 30, (obs['target'] - obs['score'][1-player]) / 30, (obs['target'] - max(obs['score'])) / 30))
    rates = profile.rates()
    belief = inference_from_observation(obs, rates['bluff'])
    card = obs['hand'][obs['available'][0]] if obs['available'] else obs['hand'][0]
    summaries = belief.summaries(card, ties_win=obs['mano'] == player)
    values.extend((summaries['brava_probability'], summaries['expected_strength'] / 24.7, summaries['next_trick_win_probability']))
    values.extend(rates.values())
    values.extend([0] * (INPUTS - len(values)))
    return np.asarray(values, dtype=np.float64)

class NetworkPolicy:
    def __init__(self, genome, seed=0, bias=None, profile=None):
        self.genome = np.asarray(genome, dtype=np.float64).copy()
        if self.genome.shape != (GENOME_SIZE,):
            raise ValueError('Invalid genome shape')
        self.rng = np.random.default_rng(seed)
        self.bias = dict(bias or {})
        self.profile = profile or OpponentProfile()

    def __call__(self, observation, legal_actions):
        x = encode_observation(observation, self.profile)
        a = INPUTS * HIDDEN
        b = a + HIDDEN
        c = b + HIDDEN * len(ACTIONS)
        hidden = np.tanh(x @ self.genome[:a].reshape(INPUTS, HIDDEN) + self.genome[a:b])
        logits = hidden @ self.genome[b:c].reshape(HIDDEN, len(ACTIONS)) + self.genome[c:]
        scores = np.array([logits[ACTIONS.index(action)] + self.bias.get(action, 0) for action in legal_actions])
        weights = np.exp(scores - scores.max())
        return legal_actions[self.rng.choice(len(legal_actions), p=weights / weights.sum())]

    def save(self, path):
        np.save(path, self.genome)

    @classmethod
    def load(cls, path, **kwargs):
        return cls(np.load(path, allow_pickle=False), **kwargs)
