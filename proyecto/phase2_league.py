"""Pure, reproducible phase 2 league draws and frozen phase 1 manifest."""
import hashlib
import json
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


def draw_opponent(seed, block_id, shares=None, pool=(), phase1=(), exclusions=(), matches_so_far=0, target_offset=.1, ramp_matches=20000, neutral_rate=.05):
    shares = dict(DEFAULT_SHARES if shares is None else shares)
    if set(shares) != set(DEFAULT_SHARES) or any(v < 0 for v in shares.values()) or abs(sum(shares.values()) - 1) > 1e-9:
        raise ValueError('League shares must be nonnegative and sum to one')
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
    opponent = identity(rng.choice(candidates)) if candidates else None
    offset = target_offset * min(1, matches_so_far / ramp_matches) if ramp_matches else target_offset
    return {'category': 'styled' if style else category, 'identity': opponent, 'style': style,
            'offset': offset if style else 0.0, 'neutral_profile': rng.random() < neutral_rate}
