"""Pure selection-panel decisions. Scores are fractions per mirror pair."""
import hashlib
import json
import math
import statistics
from pathlib import Path

RESERVED_SEEDS = frozenset([0, 1, 2, 3, 777, 4242, 9001, 9002, 9101, 9201, 9202, *range(9211, 9217), 9221, 9222, *range(9301, 9329)])

def check_seed(base, check):
    if base in RESERVED_SEEDS:
        raise ValueError('reserved schedule seed')
    if check < 0:
        raise ValueError('negative check index')
    seed = base + check
    if seed in RESERVED_SEEDS:
        raise ValueError('reserved schedule seed')
    return seed

def promotion_decision(challenger, champion, direct, min_gain=.01, min_direct=.475):
    if not challenger or challenger.keys() != champion.keys() or not direct:
        raise ValueError('missing panel or direct results')
    means, variances = [], []
    for name, scores in challenger.items():
        other = champion[name]
        if len(scores) != len(other) or len(scores) < 2:
            raise ValueError('paired results require at least two matching deals')
        differences = [a-b for a, b in zip(scores, other)]
        means.append(statistics.mean(differences))
        variances.append(statistics.variance(differences) / len(differences))
    gain = statistics.mean(means)
    margin = 1.96 * math.sqrt(sum(variances)) / len(means)
    lower = gain - margin
    rate = statistics.mean(direct)
    return {'gain': gain, 'lower_bound': lower, 'upper_bound': gain + margin,
            'head_to_head': rate, 'promoted': gain >= min_gain and lower > 0 and rate >= min_direct}

def saturated(history, champion_id, members, checks=3, min_pairs=300, threshold=.7):
    recent = []
    for record in reversed(history):
        if record['champion'] != champion_id:
            break
        recent.append(record)
        if len(recent) == checks:
            break
    if len(recent) < checks:
        return False
    for name in members:
        scores = [s for record in recent for s in record['scores'][name]]
        if len(scores) < min_pairs or statistics.mean(scores) < threshold:
            return False
    return True

def build_panel(panel, scores, snapshots, champion_sha256):
    """Rank strong by gain, distinct by L1 distance; break ties by hash."""
    originals = [m for m in panel['members'] if m['kind'] == 'phase1']
    if len(originals) < 2:
        raise ValueError('two original phase 1 members required')
    kept = sorted(originals, key=lambda m: (scores[m['name']], m['name']))[:2]
    eligible = {s['sha256']: s for s in snapshots if s['sha256'] != champion_sha256 and s['sha256'] not in {m['sha256'] for m in kept}}
    if len(eligible) < 2:
        return None
    strong = sorted(eligible.values(), key=lambda s: (-s['gain'], s['sha256']))[0]
    def distance(s):
        return sum(abs(s['scores'][name] - strong['scores'][name]) for name in sorted(strong['scores']))
    distinct = sorted((s for s in eligible.values() if s['sha256'] != strong['sha256']), key=lambda s: (-distance(s), s['sha256']))[0]
    if distance(distinct) == 0:
        return None
    return {'version': panel['version'] + 1, 'members': kept + [strong, distinct], 'reserved': [strong['sha256'], distinct['sha256']]}

def load_panel(path):
    path = Path(path)
    panel = json.loads(path.read_text(encoding='utf-8'))
    if panel.get('pending_member'):
        raise ValueError('panel has a pending member')
    if not panel.get('members'):
        raise ValueError('empty panel')
    for member in panel['members']:
        if member['kind'] not in ('phase1', 'phase2') or member['encoder'] not in ('v1', 'v2', 'v3'):
            raise ValueError('invalid panel member')
        if hashlib.sha256((path.parent / member['path']).read_bytes()).hexdigest() != member['sha256']:
            raise ValueError('panel hash mismatch: ' + member['name'])
    return panel
