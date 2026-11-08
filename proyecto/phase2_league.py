"""Pure, reproducible phase 2 league draws and frozen phase 1 manifest."""
import hashlib
import json
import math
import random
from pathlib import Path

STYLES = ('bluffer', 'cautious', 'aggressive', 'calculating')
DEFAULT_SHARES = {'self': .5, 'phase1': .1, 'snapshot': .2, **{s: .05 for s in STYLES}}


def load_phase1_manifest(path):
    path = Path(path)
    members = json.loads(path.read_text(encoding='utf-8'))['members']
    for member in members:
        file = (path.parent / member['path']).resolve()
        if hashlib.sha256(file.read_bytes()).hexdigest() != member['sha256']:
            raise ValueError(f"Hash mismatch for {member['name']}")
    return members


def validate_shares(shares):
    shares = dict(DEFAULT_SHARES if shares is None else shares)
    if set(shares) != set(DEFAULT_SHARES) or any(not math.isfinite(v) or v < 0 for v in shares.values()) or abs(sum(shares.values()) - 1) > 1e-9:
        raise ValueError('League shares must be nonnegative and sum to one')
    return shares


def draw_opponent(seed, block_id, shares=None, pool=(), phase1=(), exclusions=(), matches_so_far=0, target_offset=.1, ramp_matches=20000, neutral_rate=.05):
    shares = validate_shares(shares)
    if not 0 <= neutral_rate <= 1 or target_offset < 0 or ramp_matches < 0 or matches_so_far < 0:
        raise ValueError('Invalid league configuration')
    def identity(item):
        return item['sha256'] if isinstance(item, dict) else item
    pool = [x for x in pool if identity(x) not in exclusions]
    phase1 = [x for x in phase1 if identity(x) not in exclusions]
    # Unavailable frozen opponents are replaced with current self.
    for category, available in [('snapshot', pool), ('phase1', phase1)]:
        if not available:
            shares['self'] += shares[category]
            shares[category] = 0
    digest = hashlib.sha256(f'{seed}:{block_id}'.encode()).digest()
    rng = random.Random(int.from_bytes(digest, 'big'))
    pick = rng.random()
    category = 'self'
    for name, weight in shares.items():
        pick -= weight
        if pick < 0:
            category = name
            break
    style = category if category in STYLES else None
    candidates = pool if category == 'snapshot' or style else phase1 if category == 'phase1' else ()
    opponent = identity(rng.choice(candidates)) if candidates else 'current'
    offset = target_offset * min(1, matches_so_far / ramp_matches) if ramp_matches else target_offset
    return {'category': 'styled' if style else category, 'identity': opponent, 'style': style,
            'offset': offset if style else 0.0, 'neutral_profile': rng.random() < neutral_rate}



def retain_snapshots(snapshots, recent=8, diverse=4, exclusions=()):
    """Keep recent snapshots and older ones farthest from their nearest recent vector.

    Only identical, nonempty panel score dimensions are comparable. Missing or
    incompatible scores receive distance zero; hashes break ties ascending.
    """
    if recent < 0 or diverse < 0:
        raise ValueError('Pool limits must be nonnegative')
    eligible = sorted((s for s in snapshots if s['sha256'] not in exclusions),
                      key=lambda s: (s['generation'], s['sha256']))
    newest = eligible[-recent:] if recent else []
    older = eligible[:-recent] if recent else eligible
    def distance(candidate):
        scores = candidate.get('scores', {})
        distances = [sum(abs(scores[k] - other['scores'][k]) for k in scores)
                     for other in newest if scores and scores.keys() == other.get('scores', {}).keys()]
        return min(distances) if distances else 0
    return newest + sorted(older, key=lambda s: (-distance(s), s['sha256']))[:diverse]


def panel_exclusions(panel, history=()):
    """Collect all hashes ever reserved by a selection panel."""
    result = set()
    for item in [panel, *history]:
        if not item:
            continue
        result.update(member['sha256'] for member in item.get('members', []) if member.get('sha256'))
        result.update(item.get('reserved', []))
        pending = item.get('pending_member')
        if isinstance(pending, dict) and pending.get('sha256'):
            result.add(pending['sha256'])
    return result
