"""Measure policy distributions on a frozen bank of legal decisions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from phase1_network import ACTIONS, INPUTS, HIDDEN, NetworkPolicy, encode_observation
from opponent_profile import OpponentProfile
from symmetric_engine import play_match
from offspring_diagnostic import variants
from panel_evaluation import PANEL, digest

BANK_SEED = 10**19

def compare(a, b):
    return float(np.mean(np.abs(a-b).sum(axis=1)/2)), float(np.mean(a.argmax(axis=1) == b.argmax(axis=1)))

def _bank_match(job):
    a, b, seed = job
    result = play_match(NetworkPolicy(a, seed=seed*2), NetworkPolicy(b, seed=seed*2+1), seed=seed, target=15, deal_partition=1)
    return [(d['observation'], d['legal_actions']) for d in result.decisions]

def probabilities(genome, features, mask):
    a = INPUTS*HIDDEN; b = a+HIDDEN; c = b+HIDDEN*len(ACTIONS)
    hidden = np.tanh(features @ genome[:a].reshape(INPUTS, HIDDEN) + genome[a:b])
    logits = hidden @ genome[b:c].reshape(HIDDEN, len(ACTIONS)) + genome[c:]
    logits = np.where(mask, logits, -np.inf)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)

def diversity(items):
    comparisons = [compare(a, b) for a,b in itertools.combinations(items, 2)]
    return {'mean_pairwise_tv': float(np.mean([x[0] for x in comparisons])), 'mean_argmax_agreement': float(np.mean([x[1] for x in comparisons]))}

def diagnose(checkpoint, observations=2000, k=32, workers=12):
    start = perf_counter()
    with np.load(checkpoint, allow_pickle=False) as data:
        population = data['population']
        state = json.loads(str(data['state']))
    if state.get('encoder', 'v1') != 'v1':
        raise ValueError('This diagnostic is v1-only')
    if len(population)<6 or min(observations, k, workers)<1: raise ValueError('invalid counts')
    rng = np.random.default_rng(770183)
    bank = []
    # Fixed batches keep construction reproducible across worker counts.
    while len(bank) < observations:
        jobs = []
        for _ in range(8):
            a,b = rng.choice(len(population), 2, replace=False)
            seed = 2*(BANK_SEED + len(bank) + len(jobs))+1
            jobs.append((population[a], population[b], seed))
        if workers == 1: matches = map(_bank_match, jobs)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool: matches = list(pool.map(_bank_match, jobs))
        bank.extend(item for match in matches for item in match)
    bank = bank[:observations]
    profile = OpponentProfile()
    features = np.array([encode_observation(obs, profile) for obs, _ in bank])
    mask = np.array([[action in legal for action in ACTIONS] for _, legal in bank])
    members = json.loads((PANEL/'panel.json').read_text(encoding='utf-8'))
    for member in members:
        if digest(PANEL/member['file']) != member['sha256']: raise ValueError('panel hash mismatch')
    panel = [np.load(PANEL/member['file'], allow_pickle=False) for member in members]
    pop_probs = [probabilities(g, features, mask) for g in population]
    panel_probs = [probabilities(g, features, mask) for g in panel]
    rng = np.random.default_rng(720394)
    reserved = rng.choice(len(population), 4, replace=False)
    candidates = np.array([i for i in range(len(population)) if i not in reserved])
    indices = rng.choice(candidates, (k, 2), replace=True)
    results = {}
    for a,b in indices:
        group = variants(population[a], population[b], rng)
        for name, genome in group.items():
            if name != 'parent': results.setdefault(name, []).append(compare(pop_probs[a], probabilities(genome, features, mask)))
    return {'checkpoint': str(checkpoint), 'observations': observations, 'bank_seed_namespace': 'odd 2*(10^19 + offset)+1', 'profile': 'fresh default profile for all observations', 'population': diversity(pop_probs), 'panel': diversity(panel_probs), 'variants': {name: {'mean_tv': float(np.mean([v[0] for v in vals])), 'mean_argmax_agreement': float(np.mean([v[1] for v in vals]))} for name, vals in results.items()}, 'seconds': perf_counter()-start}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint'); parser.add_argument('--output', required=True); parser.add_argument('--observations', type=int, default=2000); parser.add_argument('--k', type=int, default=32); parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if min(args.observations, args.k, args.workers)<1: parser.error('counts must be positive')
    Path(args.output).write_text(json.dumps(diagnose(args.checkpoint, args.observations, args.k, args.workers), indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__': main()
