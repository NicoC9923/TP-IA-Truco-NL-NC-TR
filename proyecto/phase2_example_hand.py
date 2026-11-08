"""One fixed-seed example hand: the reigning phase 2 champion against a fixed opponent.

Torch is optional, so everything torch-related is imported lazily and a missing
torch (or a missing champion) yields an "unavailable" payload instead of an error.
"""
import json
import math
from pathlib import Path

SEED = 7
DEFAULT_TEMPERATURE = 0.1
CHAMPION_FILE = 'reigning_champion.pt'
OPPONENT_FILE = Path('champions') / 'phase1' / 'reigning_champion.npy'
OPPONENT_NAME = 'Campeona de la fase 1'


def _card(card):
    return {'valor': card[0], 'palo': card[1]}


def find_champion(root):
    root = Path(root)
    for directory in (root / 'training_output' / 'phase2', root / 'champions' / 'phase2'):
        if (directory / CHAMPION_FILE).is_file():
            return directory / CHAMPION_FILE
    return None


def recorded_temperature(root):
    """Temperature the run recorded in its progress file (live run first), else the default."""
    root = Path(root)
    for directory in (root / 'training_output' / 'phase2', root / 'champions' / 'phase2'):
        try:
            value = json.loads((directory / 'progress.json').read_text(encoding='utf-8')).get('temperature')
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            return float(value)
    return DEFAULT_TEMPERATURE


def unavailable(reason):
    return {'available': False, 'reason': reason}


def example_hand_payload(root, seed=SEED):
    champion_path = find_champion(root)
    if champion_path is None:
        return unavailable('no_champion')
    opponent_path = Path(root) / OPPONENT_FILE
    if not opponent_path.is_file():
        return unavailable('no_opponent')
    try:
        import numpy as np
        import torch
        from phase1_network import NetworkPolicy
        from phase2_torch import Phase2Network, Phase2Policy
        from symmetric_engine import play_match
    except ImportError:
        return unavailable('torch')
    temperature = recorded_temperature(root)
    model = Phase2Network()
    model.load_state_dict(torch.load(champion_path, map_location='cpu', weights_only=True))
    model.eval()
    policy = Phase2Policy(model, seed=seed, temperature=temperature)
    top = []

    def champion(observation, legal_actions):
        action = policy(observation, legal_actions)
        ranked = sorted(policy.last_probabilities.items(), key=lambda item: -item[1])[:3]
        top.append([{'action': name, 'pct': round(100 * p, 1)} for name, p in ranked])
        return action

    rival = NetworkPolicy(np.load(opponent_path), seed=seed)
    result = play_match(champion, rival, seed=seed)
    hand = [d for d in result.decisions if d['hand_index'] == 0]
    own = iter(top)
    steps = []
    for decision in hand:
        step = {'by': 'champion' if decision['player'] == 0 else 'rival', 'action': decision['action']}
        if decision['player'] == 0:
            step['top'] = next(own)
        steps.append(step)
    deal = result.deals[0]
    return {'available': True, 'seed': seed, 'temperature': temperature, 'opponent': OPPONENT_NAME,
            'champion_hand': [_card(c) for c in deal[0]], 'opponent_hand': [_card(c) for c in deal[1]],
            'steps': steps}
