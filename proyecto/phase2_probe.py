"""One-shot, bounded post-training card probe. See experiments/phase2-probe/RULE.txt."""
import time
COMMAND_START = time.monotonic()

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / 'champions/phase2/reigning_champion.pt'
OUTPUT = ROOT / 'champions/phase2'
RULE = ROOT / 'experiments/phase2-probe/RULE.txt'
MODELS = {
    'champion': ('champions/phase2/reigning_champion.pt', '05f189491a1d6ca7a4d7cebfa556a16e70d7512028e6bb224d61d87a35d4cdcd'),
    'league1': ('experiments/phase1-gate53-seed1/reigning_champion.npy', 'b00e64118ce54e8063b7eb61c8085ec6093f732c00112d61a26a608ee666f5fd'),
    'league2': ('experiments/phase1-encoder-v3-seed1/reigning_champion.npy', '24773046ba6753f1f13614fda74d15a31b1756c2976a76cc6eb384504a311763'),
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


def initialize_models(champion_path=CHAMPION):
    import torch
    from phase2_torch import Phase2Network
    champion = Phase2Network.load(champion_path).eval()
    control = Phase2Network(seed=39171).eval()
    for model in (champion, control):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    torch.manual_seed(39172)
    first = make_probe()
    second = make_probe()
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    return (champion, control), (first, second)


def capture_public(observation, inputs, bluff_rate):
    """Freeze only information available before the policy chooses its action."""
    from opponent_inference import compact_evidence
    return {'observation': copy.deepcopy(observation), 'features': inputs.copy(),
            'bluff_rate': bluff_rate, 'evidence': compact_evidence(observation)}


def attach_label(capture, original_hand, match_id, split, situation_index, champion, control):
    """Attach simulation label after match, without feeding it to feature extraction."""
    import torch
    from opponent_inference import _CARD_IDS, _HAND_IDS
    hand = tuple(sorted(original_hand, key=_CARD_IDS.__getitem__))
    obs = capture['observation']
    exposed = tuple(card for player, card in obs['table'] if player != obs['player'])
    with torch.no_grad():
        tensor = torch.from_numpy(capture['features'])
        a = champion.layers[:4](tensor).numpy().copy()
        b = control.layers[:4](tensor).numpy().copy()
    return (match_id, obs, capture['bluff_rate'], hand, exposed, capture['evidence'],
            _HAND_IDS[hand], a, b, split, situation_index)


def score_situation(row, probes):
    """Score four predictors with stable log probabilities and common evidence."""
    import numpy as np
    import torch
    from opponent_inference import exact_inference_from_observation, inference_from_observation, _CARD_IDS
    match, obs, rate, hand, exposed, evidence, label, a, b, split, index = row
    exact = exact_inference_from_observation(obs).distribution()
    weighted = inference_from_observation(obs, rate).distribution()
    hands = tuple(exact)
    true_index = next(i for i, h in enumerate(hands) if frozenset(h) == frozenset(hand))
    ids = torch.tensor([[_CARD_IDS[c] for c in h] for h in hands])
    log_probs = {'exact': np.log(np.asarray(list(exact.values()), dtype=np.float64)),
                 'weighted': np.log(np.asarray([weighted[h] for h in hands], dtype=np.float64))}
    with torch.no_grad():
        for name, probe, features in zip(('champion', 'control'), probes, (a, b)):
            logits = probe(torch.from_numpy(features)).double()
            log_probs[name] = torch.log_softmax(logits[ids].sum(dim=1), dim=0).numpy()
    hidden = set(hand).difference(exposed)
    if not hidden:
        raise ValueError('no hidden cards')
    metrics, marginals = {}, {}
    hand_ids = ids.numpy()
    for name, logs in log_probs.items():
        marginal = np.zeros(40, dtype=np.float64)
        probs = np.exp(logs)
        for column in range(3):
            np.add.at(marginal, hand_ids[:, column], probs)
        for card in exposed:
            marginal[_CARD_IDS[card]] = 0
        top = set(sorted(range(40), key=lambda c: (-marginal[c], c))[:len(hidden)])
        metrics[name] = {'log_loss': float(-logs[true_index]),
                         'hidden_card_accuracy': sum(_CARD_IDS[c] in top for c in hidden) / len(hidden)}
        marginals[name] = marginal.tolist()
    return {'match': match, 'split': split, 'index': index,
            'public': {'observation': obs, 'bluff_rate': rate, 'compact_evidence': evidence},
            'label': {'original_hand': hand, 'hand_index': label}, 'metrics': metrics}, marginals


def summarize_evidence(records):
    return {name: {metric: grouped_summary([(r['match'], r['metrics'][name][metric]) for r in records])
                   for metric in ('log_loss', 'hidden_card_accuracy')}
            for name in ('exact', 'weighted', 'champion', 'control')}


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


def run_child(command_start, deadline):
    import numpy as np
    import torch
    from engine_policies import ReactivePolicy, AdaptiveUtilityPolicy
    from phase1_network import NetworkPolicy
    from phase2_network import ENCODER_VERSION
    from phase2_torch import Phase2Policy
    from opponent_inference import compatible_mask
    from symmetric_engine import play_match
    torch.set_num_threads(1)
    verify_models()
    before = digest(CHAMPION)
    (champion, control), probes = initialize_models()
    league = {name: np.load(ROOT / MODELS[name][0]) for name in ('league1', 'league2')}
    class Collector(Phase2Policy):
        def __call__(self, obs, legal):
            self.obs = copy.deepcopy(obs)
            self.rate = self.profile.rates()['bluff']
            return super().__call__(obs, legal)
        def record_decision(self, inputs, action):
            self.records.append(capture_public(self.obs, inputs, self.rate))
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.records = []
    def opponent(name, seed):
        if name == 'champion':
            return Phase2Policy(champion, seed=seed, temperature=.05, profile_prior=5)
        if name in league:
            return NetworkPolicy(league[name], seed=seed, encoder={'league1': 'v1', 'league2': 'v3'}[name])
        return ReactivePolicy(seed) if name == 'reactive' else AdaptiveUtilityPolicy(seed)
    start = time.monotonic()
    rows = {key: [] for key in LIMITS}
    matches = []
    # Reserve half the total for fitting/evaluation; no reallocation or second run.
    collect_until = min(deadline, command_start + 290)
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
        for d, capture in zip(decisions, policy.records):
            obs = capture['observation']
            exposed = tuple(card for player, card in obs['table'] if player != seat)
            if len(exposed) < 3:
                candidates.append((capture, result.deals[d['hand_index']][1-seat]))
        count = min(8, len(candidates), LIMITS[split] - len(rows[split]))
        for j in np.linspace(0, len(candidates)-1, count, dtype=int):
            capture, hand = candidates[j]
            rows[split].append(attach_label(capture, hand, i, split, len(rows[split]), champion, control))
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
    fit_deadline = min(deadline, command_start + 450)
    for epoch in range(10):
        order = rng.permutation(len(rows['train']))
        for start_index in range(0, len(order), 256):
            if time.monotonic() >= fit_deadline:
                break
            inds = order[start_index:start_index+256]
            for k in range(2):
                if stale[k] >= 2:
                    continue
                x, mask, labels = batch(rows['train'], inds, k)
                loss = combination_loss(probes[k](x), mask, labels)
                optimizers[k].zero_grad(); loss.backward(); optimizers[k].step()
        if time.monotonic() >= fit_deadline:
            break
        for k in range(2):
            if stale[k] >= 2:
                continue
            total = 0
            with torch.no_grad():
                for offset in range(0, len(rows['dev']), 256):
                    if time.monotonic() >= fit_deadline:
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
    # Bounded cache: separate labels/evidence from frozen trunk features.
    all_rows = [r for key in LIMITS for r in rows[key]]
    np.savez_compressed(OUTPUT / 'card-probe-cache.npz',
        match=np.asarray([r[0] for r in all_rows]), split=np.asarray([r[9] for r in all_rows]),
        index=np.asarray([r[10] for r in all_rows]), labels=np.asarray([r[6] for r in all_rows]),
        champion_features=np.asarray([r[7] for r in all_rows], dtype=np.float32),
        control_features=np.asarray([r[8] for r in all_rows], dtype=np.float32))
    records, examples = [], []
    evidence = OUTPUT / 'card-probe-test.jsonl'
    with evidence.open('w', encoding='utf8', buffering=1) as stream:
        for row in rows['test']:
            if time.monotonic() >= deadline:
                break
            record, marginals = score_situation(row, probes)
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')
            if len(examples) < 4:
                examples.append({'match': record['match'], 'index': record['index'],
                                 'public': record['public'], 'label': record['label'], 'marginals': marginals})
    report['predictors'] = summarize_evidence(records)
    report['examples'] = examples
    report['test_scored'] = len(records)
    report['evidence_file'] = evidence.name
    report['evidence_sha256'] = digest(evidence)
    report['cache_file'] = 'card-probe-cache.npz'
    report['cache_sha256'] = digest(OUTPUT / report['cache_file'])
    report['status'] = ('partial_test' if len(records) != len(rows['test']) else
                        'complete' if all(len(rows[k]) == LIMITS[k] for k in rows) else 'underfilled')
    for name, probe in zip(('champion','control'), probes):
        torch.save({'schema': 1, 'champion_sha256': before, 'encoder_version': ENCODER_VERSION,
                    'architecture': 'linear128x40', 'weights': probe.state_dict()}, OUTPUT / ('card-probe-' + name + '.pt'))
    atomic_json(OUTPUT / 'card-probe-report.json', report)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--child':
        run_child(float(sys.argv[2]), float(sys.argv[3])); return
    if len(sys.argv) != 1:
        raise SystemExit('Usage: python proyecto/phase2_probe.py')
    start = COMMAND_START
    verify_models()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'proyecto',
                                     'experiments/phase2-probe/RULE.txt',
                                     *[item[0] for item in MODELS.values()]], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError('Probe sources, rule or models are not clean: ' + dirty)
    marker = OUTPUT / 'card-probe-started.lock'
    with marker.open('x', encoding='utf8') as f:
        f.write('One attempt only; no automatic retry.\n')
    before = digest(CHAMPION)
    failure = None
    try:
        remaining = 595 - (time.monotonic() - start)
        result = subprocess.run([sys.executable, __file__, '--child', str(start), str(start + 585)], timeout=max(0, remaining))
        if result.returncode:
            raise RuntimeError('probe child failed with exit code ' + str(result.returncode))
    except subprocess.TimeoutExpired:
        failure = 'hard_timeout'
    except Exception as exc:
        failure = 'failed: ' + str(exc)
    finally:
        report_path = OUTPUT / 'card-probe-report.json'
        report = json.loads(report_path.read_text(encoding='utf8')) if report_path.exists() else {'schema': 1, 'predictors': {}}
        after = digest(CHAMPION)
        report['champion_sha256_before'] = before
        report['champion_sha256_after'] = after
        report['champion_unchanged'] = before == after
        report['elapsed_seconds'] = time.monotonic() - start
        if failure or before != after:
            report['status'] = failure or 'champion_changed'
        atomic_json(report_path, report)
    if failure or before != after:
        raise SystemExit(failure or 'champion changed')


if __name__ == '__main__':
    main()
