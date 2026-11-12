"""Offline, deterministic measuring-run analysis; never modifies trainer semantics."""
import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np
import torch
from phase1_network import ACTIONS, NetworkPolicy
from phase2_torch import Phase2Network, Phase2Policy
from opponent_profile import OpponentProfile
from symmetric_engine import play_match
from validation_schedule import validation_schedule

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'experiments/phase2-selection-panel/panel-v0-measuring.json'
LEAGUE = ROOT / 'experiments/phase2-selection-panel/league-phase1.json'

class Probe(Phase2Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decisions = []

    def __call__(self, observation, legal_actions):
        self.legal = list(legal_actions)
        return super().__call__(observation, legal_actions)

    def record_decision(self, inputs, action):
        with torch.no_grad():
            values = self.model(torch.from_numpy(inputs)).numpy()
        self.decisions.append(([float(values[ACTIONS.index(a)]) for a in self.legal], self.legal.index(action)))

class Events(OpponentProfile):
    def __init__(self):
        super().__init__()
        self.events = []

    def _record(self, name, success):
        self.events.append((name, bool(success)))
        super()._record(name, success)

def temperature_metrics(decisions):
    tops, mistakes = [], []
    for values, chosen in decisions:
        logits = np.asarray(values, dtype=np.float64) / .1
        weights = np.exp(logits - max(logits))
        tops.append(float(max(weights) / sum(weights)))
        mistakes.append(max(values) - values[chosen] > .3)
    if not tops:
        raise ValueError('No temperature decisions')
    median = statistics.median(tops)
    return {'median_top_probability': median, 'tactical_mistake_rate': statistics.mean(mistakes),
            'decisions': len(tops), 'temperature': .2 if median > .98 else .05 if median < .40 else .1}

def prior_metrics(histories):
    losses = {p: [] for p in (20, 10, 5)}
    for events in histories:
        counts = {}
        for name, outcome in events:
            success, total = counts.get(name, (0, 0))
            if total:  # first observation builds the profile; every later observation is held out
                for prior in losses:
                    probability = (prior * .5 + success) / (prior + total)
                    losses[prior].append(-math.log(probability if outcome else 1 - probability))
            counts[name] = (success + int(outcome), total + 1)
    if not losses[20]:
        raise ValueError('No held-out profile opportunities')
    means = {p: statistics.mean(v) for p, v in losses.items()}
    return {'mean_log_loss': means, 'scored_opportunities': len(losses[20]),
            'chosen_prior': min((20, 10, 5), key=lambda p: means[p])}

def generation_size(rate, overhead):
    if rate <= 0 or overhead < 0:
        raise ValueError('Invalid throughput or overhead')
    choices = [(abs((n / rate + overhead) - 450), n, n / rate + overhead)
               for n in range(240, int(rate * 600) + 241, 240)
               if 300 <= n / rate + overhead <= 600 and overhead / (n / rate + overhead) <= .15]
    if not choices:
        return {'matches': None, 'reason': 'No multiple of 240 satisfies both limits'}
    _, n, duration = min(choices)
    return {'matches': n, 'estimated_seconds': duration, 'overhead_fraction': overhead / duration}

def replay(snapshot, smoke=False):
    torch.set_num_threads(1)
    model = Phase2Network.load(snapshot)
    members = json.loads(LEAGUE.read_text())['members']
    opponents = [('self', None)] + [(m['name'], m) for m in members] + [(s, None) for s in ('bluffer', 'calculating', 'cautious', 'aggressive')]
    schedule = validation_schedule(26003 if smoke else 49000, None, len(opponents))
    decisions, histories = [], []
    for i, ((name, member), (deal, target)) in enumerate(zip(opponents, schedule)):
        profile = Events()
        learner = Probe(model, seed=(27000 if smoke else 50000) + i * 2, profile=profile)
        if member:
            path = (LEAGUE.parent / member['path']).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest() != member['sha256']:
                raise ValueError('League hash mismatch: ' + name)
            opponent = NetworkPolicy(np.load(path), seed=(27001 if smoke else 50001) + i * 2, encoder=member['encoder'])
        else:
            opponent = Phase2Policy(model, seed=(27001 if smoke else 50001) + i * 2, style=None if name == 'self' else name, style_strength=.1)
        play_match(learner, opponent, seed=deal, target=target)
        if i < 5:
            decisions.extend(learner.decisions)
        if i > 0:
            histories.append(profile.events)
    return temperature_metrics(decisions), prior_metrics(histories)

def segment(directory, benchmark, smoke=False):
    progress = json.loads((directory / 'progress.json').read_text())
    rows = [json.loads(x) for x in (directory / 'generations.jsonl').read_text().splitlines() if x]
    bench = json.loads(benchmark.read_text())
    complete = [r for r in rows if r['matches'] == 240 and r.get('check')]
    check = statistics.mean(bench['check_seconds'])
    if not smoke and len(complete) < 2:
        raise ValueError(f'{directory}: fewer than two complete checks')
    # Head-off generation elapsed is logged before its check.
    if complete:
        train_seconds = sum(r['elapsed_seconds'] for r in complete)
        rate = sum(r['matches'] for r in complete) / train_seconds if train_seconds > 0 else None
    else:
        rate = progress['matches'] / progress['elapsed'] if progress['elapsed'] > 0 else None
    return {'matches_per_s': progress['matches'] / progress['elapsed'], 'samples_per_s': progress['samples'] / progress['elapsed'],
            'recent_matches_per_s': progress['matches_per_s'], 'recent_samples_per_s': progress['samples_per_s'],
            'seconds_per_check': check,
            'learner_ms_per_update': bench['learner_seconds_per_update'] * 1000,
            'generation_seconds': [r['elapsed_seconds'] + (check if r.get('check') else 0) for r in rows], 'training_matches_per_s': rate,
            'matches_per_generation': generation_size(rate, check) if rate else None,
            'checks': sum(r.get('check') is not None for r in rows), 'log': rows}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=ROOT / 'training_output/phase2_measuring')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = args.directory.resolve()
    a = segment(root / 'A', root / 'bench_A.json', args.smoke)
    snapshot = sorted((root / 'A').glob('generation_*.pt'))[-1]
    temperature, prior = replay(snapshot, args.smoke)
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    panel = json.loads(PANEL.read_text())
    for m in panel['members']:
        m['path'] = os.path.relpath((PANEL.parent / m['path']).resolve(), root).replace('\\', '/')
    panel['members'].append(dict(name='measuring-final', path=os.path.relpath(snapshot, root).replace('\\', '/'), kind='phase2', encoder='phase2', sha256=digest))
    panel['version'] = 1
    panel.pop('measuring_only', None)
    (root / 'panel-v1.json').write_text(json.dumps(panel, indent=2) + '\n')
    a.pop('log')
    result = {'A': a, 'temperature': temperature, 'profile_prior': prior,
              'auxiliary_head': False, 'snapshot': str(snapshot), 'snapshot_sha256': digest,
              'selected_matches_per_generation': a['matches_per_generation']}
    (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    (root / 'summary.txt').write_text('Phase 2 measuring analysis\n' + json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

