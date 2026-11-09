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


def find_run(root):
    """The champion file and its temperature, both taken from the same directory.

    The live run directory is one unit and the committed copies are another; a
    champion is never sampled at a temperature recorded by a different run."""
    root = Path(root)
    for directory in (root / 'training_output' / 'phase2', root / 'champions' / 'phase2'):
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


def example_hand_payload(root, seed=SEED):
    champion_path, temperature = find_run(root)
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
    key = (_digest(champion_path), _digest(opponent_path), temperature, profile_prior, seed)
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
        policy = load_phase2_policy(champion_path, seed=seed, metadata={'temperature': temperature, 'profile_prior': profile_prior})
    except ImportError:
        return unavailable('torch')
    except Exception:
        return unavailable('unreadable_champion')
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
    payload = {'available': True, 'seed': seed, 'temperature': temperature, 'opponent': OPPONENT_NAME,
            'champion_hand': [_card(c) for c in deal[0]], 'opponent_hand': [_card(c) for c in deal[1]],
            'steps': steps}
    with _cache_lock:
        _cache.clear()
        _cache[key] = payload
    return payload
