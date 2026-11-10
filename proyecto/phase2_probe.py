"""One-shot, bounded post-training card probe. See experiments/phase2-probe/RULE.txt."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / 'champions/phase2/reigning_champion.pt'
OUTPUT = ROOT / 'champions/phase2'
RULE = ROOT / 'experiments/phase2-probe/RULE.txt'
MODELS = {
    'champion': ('champions/phase2/reigning_champion.pt', '05f189491a1d6ca7a4d7cebfa556a16e70d7512028e6bb224d61d87a35d4cdcd'),
    'league1': ('experiments/phase1-seed1/reigning_champion.npy', '413542520b893418371fa0ff29b4b693d0675ef60cd4be55f8e782dd544db31c'),
    'league2': ('experiments/phase1-seed2/reigning_champion.npy', '5f00bfbaa27d7ee9fcedd7610ca36b092c71b8e935218a8f337a7cc1e25a3e79'),
}
LIMITS = {'train': 10000, 'dev': 2000, 'test': 2000}
MIXTURE = ('champion', 'league1', 'league2', 'reactive', 'adaptive')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def streams(i):
    if not 0 <= i < 100000:
        raise ValueError('match index outside preregistered range')
    split = ('train', 'train', 'train', 'train', 'train', 'dev', 'test')[i % 7]
    # Each complete 140-match cycle balances split, seat, target, and opponents.
    seat = (i // 7) % 2
    target = (15, 30)[(i // 14) % 2]
    opponent = MIXTURE[(i // 28) % 5]
    return split, 2 * (80001 * 10**9 + i), (2**129 + 2*i, 2**129 + 2*i+1), seat, target, opponent


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    os.replace(tmp, path)


def verify_models():
    for name, (path, expected) in MODELS.items():
        if digest(ROOT / path) != expected:
            raise ValueError('model identity mismatch: ' + name)


def make_probe():
    import torch
    return torch.nn.Linear(128, 40)


def masked_scores(logits, mask):
    import torch
    from opponent_inference import _IDS
    ids = torch.as_tensor(_IDS, device=logits.device)
    if not mask.any(dim=1).all():
        raise ValueError('no compatible hands')
    return logits[:, ids].sum(dim=2).masked_fill(~mask, float('-inf'))


def combination_loss(logits, mask, labels):
    import torch
    if not mask[torch.arange(len(labels)), labels].all():
        raise ValueError('true hand excluded by public evidence')
    return torch.nn.functional.cross_entropy(masked_scores(logits, mask), labels)


def grouped_summary(values):
    groups = {}
    for match, value in values:
        groups.setdefault(match, []).append(value)
    if not groups:
        return {'mean': None, 'lower': None, 'upper': None, 'matches': 0, 'situations': 0}
    # Cluster-robust normal interval for the situation-weighted mean.
    n = sum(map(len, groups.values()))
    mean = sum(sum(v) for v in groups.values()) / n
    g = len(groups)
    variance = (g / (g - 1) * sum((sum(v) - len(v)*mean)**2 for v in groups.values()) / n**2) if g > 1 else None
    margin = 1.96 * math.sqrt(variance) if variance is not None else None
    return {'mean': mean, 'lower': mean-margin if margin is not None else None,
            'upper': mean+margin if margin is not None else None, 'matches': g, 'situations': n}


def load_probe(path, champion=CHAMPION):
    import torch
    from phase2_network import ENCODER_VERSION
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (set(payload) != {'schema', 'champion_sha256', 'encoder_version', 'architecture', 'weights'} or
        payload['schema'] != 1 or payload['champion_sha256'] != digest(champion) or
        payload['encoder_version'] != ENCODER_VERSION or payload['architecture'] != 'linear128x40'):
        raise ValueError('incompatible probe metadata')
    model = make_probe()
    model.load_state_dict(payload['weights'], strict=True)
    model.eval()
    return model


def run_child(deadline):
    import numpy as np
    import torch
    from engine_policies import ReactivePolicy, AdaptiveUtilityPolicy
    from phase1_network import NetworkPolicy
    from phase2_network import ENCODER_VERSION
    from phase2_torch import Phase2Network, Phase2Policy
    from opponent_inference import compact_evidence, compatible_mask, exact_inference_from_observation, inference_from_observation, _CARD_IDS, _HAND_IDS, _HANDS
    from phase2_card_metrics import prediction_metrics
    from symmetric_engine import play_match
    torch.set_num_threads(1)
    verify_models()
    before = digest(CHAMPION)
    champion = Phase2Network.load(CHAMPION).eval()
    control = Phase2Network(seed=39171).eval()
    for model in (champion, control):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    league = {name: np.load(ROOT / MODELS[name][0]) for name in ('league1', 'league2')}
    class Collector(Phase2Policy):
        def __call__(self, obs, legal):
            self.obs = copy.deepcopy(obs)
            self.rate = self.profile.rates()['bluff']
            return super().__call__(obs, legal)
        def record_decision(self, inputs, action):
            self.records.append((self.obs, inputs.copy(), self.rate))
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.records = []
    def opponent(name, seed):
        if name == 'champion':
            return Phase2Policy(champion, seed=seed, temperature=.05, profile_prior=5)
        if name in league:
            return NetworkPolicy(league[name], seed=seed, encoder='v1')
        return ReactivePolicy(seed) if name == 'reactive' else AdaptiveUtilityPolicy(seed)
    start = time.monotonic()
    rows = {key: [] for key in LIMITS}
    matches = []
    # Reserve half the total for fitting/evaluation; no reallocation or second run.
    collect_until = min(deadline, start + 290)
    for i in range(100000):
        if time.monotonic() >= collect_until or all(len(rows[k]) >= LIMITS[k] for k in rows):
            break
        split, deal, seeds, seat, target, name = streams(i)
        if len(rows[split]) >= LIMITS[split]:
            continue
        policy = Collector(champion, seed=seeds[seat], temperature=.05, profile_prior=5)
        other = opponent(name, seeds[1-seat])
        policies = (policy, other) if seat == 0 else (other, policy)
        result = play_match(*policies, seed=deal, target=target, deal_partition=0)
        decisions = [d for d in result.decisions if d['player'] == seat]
        if len(decisions) != len(policy.records):
            raise ValueError('decision mismatch')
        candidates = []
        for d, (obs, x, rate) in zip(decisions, policy.records):
            exposed = tuple(card for player, card in obs['table'] if player != seat)
            if len(exposed) < 3:
                hand = tuple(sorted(result.deals[d['hand_index']][1-seat], key=_CARD_IDS.__getitem__))
                candidates.append((obs, x, rate, hand, exposed))
        count = min(8, len(candidates), LIMITS[split] - len(rows[split]))
        for j in np.linspace(0, len(candidates)-1, count, dtype=int):
            obs, x, rate, hand, exposed = candidates[j]
            # The label is attached only after the observation and profile are frozen.
            with torch.no_grad():
                tensor = torch.from_numpy(x)
                a = champion.layers[:4](tensor).numpy().copy()
                b = control.layers[:4](tensor).numpy().copy()
            rows[split].append((i, obs, rate, hand, exposed, compact_evidence(obs), _HAND_IDS[hand], a, b))
        matches.append({'index': i, 'split': split, 'deal_seed': deal, 'action_seeds': list(map(str, seeds)), 'seat': seat, 'target': target, 'opponent': name, 'situations': count})
        if time.monotonic() >= deadline:
            break
    collected = time.monotonic() - start
    report = {'schema': 1, 'status': 'insufficient_data', 'champion_sha256': before,
              'encoder_version': ENCODER_VERSION, 'architecture': 'linear128x40', 'models': MODELS,
              'rule_sha256': digest(RULE), 'revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
              'counts': {k: len(v) for k,v in rows.items()}, 'matches': matches,
              'collection_seconds': collected, 'fit_seconds': 0, 'predictors': {}, 'budget_seconds': 600}
    atomic_json(OUTPUT / 'card-probe-report.json', report)
    if not all(rows.values()) or time.monotonic() >= deadline:
        return
    torch.manual_seed(39172)
    probes = [make_probe(), make_probe()]
    optimizers = [torch.optim.Adam(p.parameters(), lr=.001) for p in probes]
    best = [float('inf'), float('inf')]
    best_state = [None, None]
    stale = [0, 0]
    rng = np.random.default_rng(39173)
    def batch(data, indices, model_index):
        selected = [data[i] for i in indices]
        x = torch.tensor(np.asarray([r[7+model_index] for r in selected]))
        mask = torch.from_numpy(compatible_mask([r[5] for r in selected]))
        labels = torch.tensor([r[6] for r in selected])
        return x, mask, labels
    fit_start = time.monotonic()
    for epoch in range(10):
        order = rng.permutation(len(rows['train']))
        for start_index in range(0, len(order), 256):
            if time.monotonic() >= deadline:
                break
            inds = order[start_index:start_index+256]
            for k in range(2):
                if stale[k] >= 2:
                    continue
                x, mask, labels = batch(rows['train'], inds, k)
                loss = combination_loss(probes[k](x), mask, labels)
                optimizers[k].zero_grad(); loss.backward(); optimizers[k].step()
        if time.monotonic() >= deadline:
            break
        for k in range(2):
            if stale[k] >= 2:
                continue
            total = 0
            with torch.no_grad():
                for offset in range(0, len(rows['dev']), 256):
                    if time.monotonic() >= deadline:
                        break
                    inds = range(offset, min(offset+256, len(rows['dev'])))
                    x, mask, labels = batch(rows['dev'], inds, k)
                    total += float(combination_loss(probes[k](x), mask, labels)) * len(labels)
                else:
                    value = total / len(rows['dev'])
                    if value < best[k]:
                        best[k] = value; best_state[k] = copy.deepcopy(probes[k].state_dict()); stale[k] = 0
                    else:
                        stale[k] += 1
        if all(x >= 2 for x in stale):
            break
    report['fit_seconds'] = time.monotonic() - fit_start
    report['epochs_attempted'] = epoch + 1
    report['dev_loss'] = [v if math.isfinite(v) else None for v in best]
    if any(state is None for state in best_state) or time.monotonic() >= deadline:
        report['status'] = 'budget_exhausted'; atomic_json(OUTPUT / 'card-probe-report.json', report); return
    for probe, state in zip(probes, best_state):
        probe.load_state_dict(state); probe.eval()
    values = {name: {'log_loss': [], 'hidden_card_accuracy': []} for name in ('exact', 'weighted', 'champion', 'control')}
    for row in rows['test']:
        if time.monotonic() >= deadline:
            report['status'] = 'budget_exhausted'; atomic_json(OUTPUT / 'card-probe-report.json', report); return
        match, obs, rate, hand, exposed, evidence, label, a, b = row
        exact = exact_inference_from_observation(obs).distribution()
        weighted = inference_from_observation(obs, rate).distribution()
        hands = tuple(exact)
        ids = torch.tensor([[_CARD_IDS[c] for c in h] for h in hands])
        distributions = {'exact': list(exact.values()), 'weighted': [weighted[h] for h in hands]}
        with torch.no_grad():
            for name, probe, features in zip(('champion', 'control'), probes, (a,b)):
                logits = probe(torch.from_numpy(features))
                distributions[name] = torch.softmax(logits[ids].sum(dim=1), dim=0).tolist()
        for name, probabilities in distributions.items():
            metrics = prediction_metrics(hands, probabilities, hand, exposed)
            for key, value in metrics.items():
                values[name][key].append((match, value))
    report['predictors'] = {name: {metric: grouped_summary(v) for metric, v in metrics.items()} for name, metrics in values.items()}
    report['status'] = 'complete' if all(len(rows[k]) == LIMITS[k] for k in rows) else 'underfilled'
    report['elapsed_seconds'] = time.monotonic() - start
    if digest(CHAMPION) != before:
        raise ValueError('champion changed')
    for name, probe in zip(('champion','control'), probes):
        torch.save({'schema': 1, 'champion_sha256': before, 'encoder_version': ENCODER_VERSION,
                    'architecture': 'linear128x40', 'weights': probe.state_dict()}, OUTPUT / ('card-probe-' + name + '.pt'))
    atomic_json(OUTPUT / 'card-probe-report.json', report)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--child':
        run_child(float(sys.argv[2])); return
    if len(sys.argv) != 1:
        raise SystemExit('Usage: python proyecto/phase2_probe.py')
    start = time.monotonic()
    verify_models()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'proyecto',
                                     'experiments/phase2-probe/RULE.txt',
                                     *[item[0] for item in MODELS.values()]], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError('Probe sources, rule or models are not clean: ' + dirty)
    marker = OUTPUT / 'card-probe-started.lock'
    with marker.open('x', encoding='utf8') as f:
        f.write('One attempt only; no automatic retry.\n')
    try:
        remaining = 595 - (time.monotonic() - start)
        result = subprocess.run([sys.executable, __file__, '--child', str(start + 590)], timeout=max(0, remaining))
        if result.returncode:
            raise RuntimeError('probe child failed with exit code ' + str(result.returncode))
    except subprocess.TimeoutExpired:
        report_path = OUTPUT / 'card-probe-report.json'
        report = json.loads(report_path.read_text(encoding='utf8')) if report_path.exists() else {'schema': 1, 'predictors': {}}
        report['status'] = 'hard_timeout'; report['elapsed_seconds'] = time.monotonic() - start
        atomic_json(report_path, report)
        raise SystemExit('Probe exceeded watchdog budget; partial report retained')


if __name__ == '__main__':
    main()
