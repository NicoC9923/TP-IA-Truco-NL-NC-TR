"""Phase 2 observation encoder, independent of PyTorch.

INDEX maps all zero-based input ranges: summary 0:64 (phase 1 v3 map,
with exact rather than weighted beliefs at 50:55); confidence 64:68
(bluff, fold, accept, envido counts); buenas 68:70 (self, opponent);
own_cards 70:190 (three 40-way slots, suit then rules.VALORES);
legal_actions 190:202 (phase1_network.ACTIONS order); events 202:1618
(24 rows of 59); exact_belief 1618:1623 and weighted_belief 1623:1628
(brava, strength / 24.7, three slot outcomes).
Each event row: 0 presence, 1 relative actor (+1 self, -1 opponent),
2:12 action category (card then ACTIONS[3:]), 12:52 public card identity,
52:55 trick one-hot, 55:57 declared tantos / 33 (self, opponent),
57:59 son buenas flags (self, opponent). Declarations appear on the
accepted envido event. The engine permits at most 17 events in a completed
hand: six cards, three truco raises and three responses, four envido calls
and one envido response. The capacity remains 24 and overflow is rejected.
"""
import numpy as np
import truco_reglas as rules
from phase1_network import ACTIONS, encode_observation as encode_v3
from opponent_inference import inference_from_observation

EVENT_SIZE = 59
MAX_EVENTS = 24
INDEX = {'summary': slice(0, 64), 'confidence': slice(64, 68),
         'buenas': slice(68, 70), 'own_cards': slice(70, 190),
         'legal_actions': slice(190, 202), 'events': slice(202, 1618),
         'exact_belief': slice(1618, 1623), 'weighted_belief': slice(1623, 1628)}
INPUT_SIZE = 1628
CATEGORIES = ('card',) + ACTIONS[3:]

def _card_id(card):
    value, suit = card
    return rules.PALOS.index(suit) * len(rules.VALORES) + rules.VALORES.index(value)

class _FixedInference:
    def __init__(self, belief):
        self.belief = belief
    def update(self, obs, bluff_rate):
        return self.belief

def encode_observation(obs, profile, legal_actions, *, weighted_beliefs=True,
                       exact_inference=None, weighted_inference=None):
    if len(obs['history']) > MAX_EVENTS:
        raise ValueError('Current hand event capacity exceeded')
    exact = (exact_inference.update(obs, 1.0) if exact_inference is not None
             else inference_from_observation(obs, 1.0))
    summary = encode_v3(obs, profile, _FixedInference(exact), encoder='v3')
    result = np.zeros(INPUT_SIZE, dtype=np.float32)
    result[INDEX['summary']] = summary
    result[INDEX['exact_belief']] = summary[50:55]
    if weighted_beliefs:
        weighted = encode_v3(obs, profile, weighted_inference, encoder='v3')
        result[INDEX['weighted_belief']] = weighted[50:55]
    result[INDEX['confidence']] = list(profile.confidence().values())
    player = obs['player']
    result[INDEX['buenas']] = [obs['target'] != 30 or obs['score'][player] >= 15,
                               obs['target'] != 30 or obs['score'][1-player] >= 15]
    for slot, card in enumerate(obs['hand']):
        result[70 + slot * 40 + _card_id(card)] = 1
    for action in legal_actions:
        result[190 + ACTIONS.index(action)] = 1
    rows = result[INDEX['events']].reshape(MAX_EVENTS, EVENT_SIZE)
    cards_played = 0
    pending_envido = False
    for row, (actor, action) in zip(rows, obs['history']):
        row[0] = 1
        row[1] = 1 if actor == player else -1
        category = 'card' if action.startswith('card_') else action
        row[2 + CATEGORIES.index(category)] = 1
        row[52 + min(cards_played // 2, 2)] = 1
        if category == 'card':
            table_actor, card = obs['table'][cards_played]
            if table_actor != actor:
                raise ValueError('History and public table disagree')
            row[12 + _card_id(card)] = 1
            cards_played += 1
        if action in ('envido', 'real_envido', 'falta_envido'):
            pending_envido = True
        elif pending_envido and action in ('quiero', 'no_quiero', 'fold'):
            if action == 'quiero' and obs.get('envido_reveal'):
                for seat, declaration in obs['envido_reveal']['declarations']:
                    relative = 0 if seat == player else 1
                    if declaration == 'son buenas':
                        row[57 + relative] = 1
                    else:
                        row[55 + relative] = declaration / 33
            pending_envido = False
    return result
