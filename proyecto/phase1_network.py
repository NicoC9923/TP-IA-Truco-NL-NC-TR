"""Fixed-topology phase 1 policy and observation encoder."""
import numpy as np
import json
from pathlib import Path
import truco_reglas as rules
from opponent_inference import inference_from_observation, IncrementalInference
from opponent_profile import OpponentProfile

ACTIONS = ('card_0', 'card_1', 'card_2', 'envido', 'real_envido', 'falta_envido', 'quiero', 'no_quiero', 'truco', 'retruco', 'vale_cuatro', 'fold')
INPUTS, HIDDEN = 64, 24
GENOME_SIZE = INPUTS * HIDDEN + HIDDEN + HIDDEN * len(ACTIONS) + len(ACTIONS)

def encode_observation(obs, profile, inference=None, encoder='v1'):
    """Encode a public observation. V3 zero-based indexes:

    0-3, 4-7, 8-11: each held slot's rank / 13, suit / 3,
    availability, and tantos digit / 7, respectively.
    12-14, 15-17, 18-20, 21-23, 24-26, 27-29: table slot rank
    / 13, suit / 3, and owner (+1 self, -1 opponent, 0 absent).
    30-32: completed trick winners (+1 self, -1 opponent, 0 tie or absent).
    33: truco / 4; 34: envido done; 35: pending envido;
    36: pending truco; 37: self is mano; 38: self owns raise.
    39-40: points to target / 30 for self and opponent; 41: falta / 30.
    42-47: signed call counts for envido, real envido, falta envido,
    truco, retruco, vale cuatro; 48-49: self and opponent declarations.
    50: brava probability; 51: expected opponent strength / 24.7;
    52-54: slot-aligned signed trick outcomes; 55-58: bluff, fold,
    truco accepted and envido called profile rates; 59: completed tricks / 2.
    60: accepted points won / 30; 61: negative accepted points lost / 30;
    62: negative declined points / 30; 63: reserved zero. Unavailable card outcomes are zero;
    ties are zero. The individual slot indexes are:
    0 rank, 1 suit, 2 available, 3 tantos; 4 rank, 5 suit, 6 available,
    7 tantos; 8 rank, 9 suit, 10 available, 11 tantos; 12 rank, 13 suit,
    14 owner; 15 rank, 16 suit, 17 owner; 18 rank, 19 suit, 20 owner;
    21 rank, 22 suit, 23 owner; 24 rank, 25 suit, 26 owner; 27 rank,
    28 suit, 29 owner; 30, 31, 32 trick winners; 42 envido, 43 real
    envido, 44 falta envido, 45 truco, 46 retruco, 47 vale cuatro;
    48 self declaration, 49 opponent declaration; 52, 53, 54 card
    outcomes; 55 bluff, 56 fold, 57 truco accepted, 58 envido called.
    """
    if encoder not in ('v1', 'v2', 'v3'):
        raise ValueError('Unknown encoder')
    values = []
    for i, (value, suit) in enumerate(obs['hand']):
        strength = value / 12 if encoder == 'v1' else rules.valor_carta_truco(value, suit) / 13
        values.extend((strength, rules.PALOS.index(suit) / 3, float(i in obs['available']), min(value, 10) % 10 / 7))
    for i in range(6):
        if i < len(obs['table']):
            player, (value, suit) = obs['table'][i]
            values.extend(((rules.valor_carta_truco(value, suit) / 13 if encoder == 'v3' else value / 12), rules.PALOS.index(suit) / 3, 1 if player == obs['player'] else -1))
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
    if encoder == 'v3':
        current = obs['table'][2 * len(obs['tricks']):]
        opposing = next((card for actor, card in current if actor != player), None)
        brava, strength, outcomes = belief.v3_summaries(
            obs['hand'], obs['available'], estimate_outcomes=opposing is None)
        values.extend((brava, strength / 24.7))
        if opposing is not None:
            other_rank = rules.valor_carta_truco(*opposing)
            for i in obs['available']:
                rank = rules.valor_carta_truco(*obs['hand'][i])
                outcomes[i] = (rank > other_rank) - (rank < other_rank)
        values.extend(outcomes)
        values.extend(rates.values())
        values.append(len(obs['tricks']) / 2)
        if pending and pending['kind'] == 'envido':
            if pending['call'] == 'falta_envido':
                win = rules.puntos_falta_envido(*obs['score'], obs['target'], ('humano', 'robot')[player])[0]
                loss = rules.puntos_falta_envido(*obs['score'], obs['target'], ('humano', 'robot')[1-player])[0]
            else:
                win = loss = pending['accepted']
            values.extend((win / 30, -loss / 30, -pending['declined'] / 30))
        else:
            values.extend((0, 0, 0))
    else:
        card = obs['hand'][obs['available'][0]] if obs['available'] else obs['hand'][0]
        summaries = belief.summaries(card, ties_win=obs['mano'] == player)
        values.extend((summaries['brava_probability'], summaries['expected_strength'] / 24.7, summaries['next_trick_win_probability']))
        values.extend(rates.values())
    values.extend([0] * (INPUTS - len(values)))
    return np.asarray(values, dtype=np.float64)

class NetworkPolicy:
    def __init__(self, genome, seed=0, bias=None, profile=None, style=None, style_strength=1.0, explain=False, encoder='v1'):
        if encoder not in ('v1', 'v2', 'v3'):
            raise ValueError('Unknown encoder')
        self.encoder = encoder
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
        x = encode_observation(observation, self.profile, self.inference, encoder=self.encoder)
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
        self.profile.observe(decisions, opponent, semantics='v3' if self.encoder == 'v3' else 'legacy')
        self.inference.reset()

    def save(self, path):
        np.save(path, self.genome)
        path = Path(path)
        if path.suffix != '.npy':
            path = Path(str(path) + '.npy')
        metadata = path.with_suffix('.encoder.json')
        if self.encoder in ('v2', 'v3'):
            metadata.write_text(json.dumps({'encoder': self.encoder}), encoding='utf-8')
        elif metadata.exists():
            metadata.unlink()

    @classmethod
    def load(cls, path, **kwargs):
        metadata = Path(path).with_suffix('.encoder.json')
        if metadata.exists():
            sidecar = json.loads(metadata.read_text(encoding='utf-8'))
            if not isinstance(sidecar, dict) or 'encoder' not in sidecar:
                raise ValueError('Sidecar is missing encoder')
            stored_encoder = sidecar['encoder']
            if 'encoder' in kwargs and kwargs['encoder'] != stored_encoder:
                raise ValueError('Explicit encoder disagrees with sidecar encoder')
            kwargs['encoder'] = stored_encoder
        else:
            kwargs.setdefault('encoder', 'v1')
        return cls(np.load(path, allow_pickle=False), **kwargs)


