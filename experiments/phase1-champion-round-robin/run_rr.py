import hashlib, itertools, json, sys
from pathlib import Path
sys.path.insert(0, 'proyecto')
import numpy as np
from panel_evaluation import _measure
NETS = [('run1-g48', 'champions/phase1/reigning_champion.npy'), ('v1s1-g23', 'experiments/phase1-seed1/reigning_champion.npy'),
        ('v1s2-g2', 'experiments/phase1-seed2/reigning_champion.npy'), ('gate53-g26', 'experiments/phase1-gate53-seed1/reigning_champion.npy'),
        ('v2s1-g46', 'experiments/phase1-encoder-v2-seed1/reigning_champion.npy'), ('v2s2-g25', 'experiments/phase1-seed2-matched/v2/reigning_champion.npy'),
        ('v3s1-g55', 'experiments/phase1-encoder-v3-seed1/reigning_champion.npy'), ('v3s2-g50', 'experiments/phase1-seed2-matched/v3/reigning_champion.npy')]
def encoder(p):
    s = Path(p).with_suffix('.encoder.json')
    return json.loads(s.read_text())['encoder'] if s.exists() else 'v1'
if __name__ == '__main__':
    nets = [dict(name=n, path=p, sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(), encoder=encoder(p), genome=np.load(p)) for n, p in NETS]
    pairs = []
    for k, (a, b) in enumerate(itertools.combinations(nets, 2)):
        seed = 9301 + k
        r = _measure(a['genome'], b['genome'], 1000, seed, 12, a['encoder'], b['encoder'])
        pairs.append(dict(a=a['name'], b=b['name'], seed=seed, **r)); print(a['name'], b['name'], round(r['win_rate'] * 100, 2), round(r['margin_of_error'] * 100, 2), flush=True)
    names = [n['name'] for n in nets]
    rate = {}
    for p in pairs:
        rate[(p['a'], p['b'])] = (p['win_rate'], p['margin_of_error']); rate[(p['b'], p['a'])] = (1 - p['win_rate'], p['margin_of_error'])
    clear = lambda x, y: rate[(x, y)][0] - rate[(x, y)][1] > .5
    cycles = sorted({tuple(c) for c in itertools.permutations(names, 3) if clear(c[0], c[1]) and clear(c[1], c[2]) and clear(c[2], c[0]) and c[0] == min(c)})
    means = {n: float(np.mean([rate[(n, m)][0] for m in names if m != n])) for n in names}
    out = dict(networks=[{k: v for k, v in n.items() if k != 'genome'} for n in nets], pairs=pairs, clear_cycles=cycles, mean_win_rate=means)
    Path('training_output/round_robin/results.json').write_text(json.dumps(out, indent=2))
    print('cycles', cycles); print('means', {k: round(v * 100, 2) for k, v in sorted(means.items(), key=lambda kv: -kv[1])})
