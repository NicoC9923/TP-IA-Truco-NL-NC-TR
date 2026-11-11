"""One fixed-seed example hand: the reigning phase 2 champion against a fixed opponent.

Torch is optional, so everything torch-related is imported lazily and a missing
torch (or a missing champion) yields an "unavailable" payload instead of an error.
"""
import hashlib
import json
import threading
import math
from pathlib import Path

SEED = 7
DEFAULT_TEMPERATURE = 0.1
CHAMPION_FILE = 'reigning_champion.pt'
SIDECAR_FILE = 'reigning_champion.json'
_cache = {}
_cache_lock = threading.Lock()
OPPONENT_FILE = Path('champions') / 'phase1' / 'reigning_champion.npy'
OPPONENT_NAME = 'Campeona de la fase 1'


def _card(card):
    return {'valor': card[0], 'palo': card[1]}


def _valid_temperature(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def find_run(root, directory=None):
    """The champion file and its temperature, both taken from the same directory.

    The live run directory is one unit and the committed copies are another; a
    champion is never sampled at a temperature recorded by a different run."""
    root = Path(root)
    for directory in ((Path(directory),) if directory is not None else (root / 'training_output' / 'phase2', root / 'champions' / 'phase2')):
        champion = directory / CHAMPION_FILE
        if not champion.is_file():
            continue
        temperature = DEFAULT_TEMPERATURE
        for name in (SIDECAR_FILE, 'progress.json'):
            try:
                value = json.loads((directory / name).read_text(encoding='utf-8')).get('temperature')
            except (OSError, ValueError, AttributeError):
                continue
            if _valid_temperature(value):
                temperature = float(value)
                break
        return champion, temperature
    return None, DEFAULT_TEMPERATURE


def find_champion(root):
    return find_run(root)[0]


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unavailable(reason):
    return {'available': False, 'reason': reason}


def decision_trace(model, inputs, probabilities, action):
    """Describe one policy decision using its exact encoded input and model."""
    import torch
    from phase1_network import ACTIONS
    with torch.no_grad():
        tensor = torch.as_tensor(inputs, dtype=torch.float32)
        first = model.layers[:2](tensor)
        second = model.layers[2:4](first)
        values = model.layers[4](second)
    # Only three representative units per layer are exposed, with exact sums
    # over all their inputs. This does not change the model or policy.
    groups = [(0, 70), (70, 190), (190, 202), (202, 1618),
              (1618, 1623), (1623, 1628)]
    if tensor.numel() > 1628:
        groups.append((1628, tensor.numel()))  # run 3 hand facts (encoder v2/v3)
    calculations = []
    for layer, source in ((model.layers[0], tensor), (model.layers[2], first)):
        units = []
        for index in range(3):
            products = layer.weight[index].detach() * source
            ranges = groups if source.numel() == tensor.numel() else [(0, source.numel())]
            units.append({'bias': float(layer.bias[index].detach()),
                          'groups': [float(products[a:b].sum()) for a, b in ranges],
                          'sum': float(products.sum() + layer.bias[index].detach())})
        calculations.append(units)
    return {
        'calculations': calculations,
        'input': tensor.tolist(), 'hidden_256': first.tolist(), 'hidden_128': second.tolist(),
        'actions': [{'action': name, 'value': float(values[i]),
                     'legal': name in probabilities, 'probability': probabilities.get(name)}
                    for i, name in enumerate(ACTIONS)],
        'chosen': action,
    }


def example_hand_payload(root, seed=SEED, directory=None):
    champion_path, temperature = find_run(root, directory=directory)
    if champion_path is None:
        return unavailable('no_champion')
    try:
        metadata = json.loads((champion_path.parent / SIDECAR_FILE).read_text(encoding='utf-8'))
        has_sidecar = True
    except FileNotFoundError:
        metadata = {}
        has_sidecar = False
    except (OSError, ValueError):
        return unavailable('invalid_champion_metadata')
    if not isinstance(metadata, dict):
        return unavailable('invalid_champion_metadata')
    if 'temperature' in metadata and not _valid_temperature(metadata['temperature']):
        return unavailable('invalid_champion_metadata')
    try:
        progress = json.loads((champion_path.parent / 'progress.json').read_text(encoding='utf-8'))
        if not isinstance(progress, dict):
            progress = {}
    except (OSError, ValueError):
        progress = {}
    profile_prior = metadata.get('profile_prior', progress.get('profile_prior', 20))
    if not _valid_temperature(profile_prior):
        return unavailable('invalid_champion_metadata')
    if not has_sidecar and profile_prior != 20:
        return unavailable('invalid_champion_metadata')
    opponent_path = Path(root) / OPPONENT_FILE
    if not opponent_path.is_file():
        return unavailable('no_opponent')
    try:
        champion_hash, opponent_hash = _digest(champion_path), _digest(opponent_path)
    except OSError:
        return unavailable('unreadable_champion')
    selection = metadata.get('selection', 'sampled')
    key = (champion_hash, opponent_hash, temperature, profile_prior, metadata.get('generation'), seed, selection)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    try:
        import numpy as np
        from phase1_network import NetworkPolicy
        from phase2_loader import load_phase2_policy
        from symmetric_engine import play_match
    except ImportError:
        return unavailable('torch')
    try:
        policy = load_phase2_policy(champion_path, seed=seed, metadata={'temperature': temperature, 'profile_prior': profile_prior, 'selection': selection})
    except ImportError:
        return unavailable('torch')
    except Exception:
        return unavailable('unreadable_champion')
    recorded = []
    captured = {}
    def capture(inputs, action):
        captured['inputs'] = inputs.copy()
    policy.record_decision = capture
    first_hand = True
    last_history = -1

    def champion(observation, legal_actions):
        nonlocal first_hand, last_history
        history_size = len(observation['history'])
        if history_size <= last_history:
            first_hand = False
        last_history = history_size
        action = policy(observation, legal_actions)
        if first_hand:
            probabilities = dict(policy.last_probabilities)
            ranked = sorted(probabilities.items(), key=lambda item: -item[1])[:3]
            recorded.append({'top': [{'action': name, 'pct': round(100 * p, 1)} for name, p in ranked],
                             'trace': decision_trace(policy.model, captured['inputs'], probabilities, action)})
        return action

    rival = NetworkPolicy(np.load(opponent_path), seed=seed)
    result = play_match(champion, rival, seed=seed)
    hand = [d for d in result.decisions if d['hand_index'] == 0]
    own = iter(recorded)
    steps = []
    for decision in hand:
        step = {'by': 'champion' if decision['player'] == 0 else 'rival', 'action': decision['action']}
        if decision['player'] == 0:
            step.update(next(own))
        steps.append(step)
    deal = result.deals[0]
    payload = {'available': True, 'seed': seed, 'temperature': temperature, 'selection': selection, 'opponent': OPPONENT_NAME,
            'model_sha256': champion_hash, 'generation': metadata.get('generation'), 'profile_prior': profile_prior, 'sample': 'illustrative',
            'champion_hand': [_card(c) for c in deal[0]],
            'steps': steps}
    with _cache_lock:
        _cache.clear()
        _cache[key] = payload
    return payload
