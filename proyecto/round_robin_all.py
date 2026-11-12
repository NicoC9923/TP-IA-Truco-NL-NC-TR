"""Descriptive round robin of playable agents (experiments/all-vs-all and experiments/run-evolution).

A config lists agents and a pairing mode (all pairs, or group A x group B plus pairs within groups); every pairing
plays `count` mirror pairs. It reuses the final exam tool for seeds, fresh-state play,
durable partial log, resume, identity and provenance checks. Nothing here selects or ships an agent.

    py round_robin_all.py experiments/all-vs-all/config.json --check     # audits and hashes only, plays nothing
    py round_robin_all.py experiments/all-vs-all/config.json --workers 12
    py round_robin_all.py experiments/all-vs-all/config.json --workers 12 --resume
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import phase2_final_exam as fx

Z95 = fx.Z95
SINCE_RUN3_SCHEDULE_SEEDS = frozenset(range(93001, 93006))     # run 3 exam (Part E, five opponent blocks)
SINCE_RUN3_ACTION_RANGES = ((2 ** 133, 2 ** 133 + 19999),)     # run 3 exam action seeds (experiment 133)
SINCE_RUN3 = (fx.EARLIER_SCHEDULE_SEEDS | fx.SINCE_RUN2_SCHEDULE_SEEDS | SINCE_RUN3_SCHEDULE_SEEDS,
              fx.EARLIER_ACTION_RANGES + fx.SINCE_RUN2_ACTION_RANGES + SINCE_RUN3_ACTION_RANGES)
# The all-vs-all config owns deal bases 94001-94099 and action exponent 134 (up to 2^134 + 2^20).
SINCE_ALL_VS_ALL = (SINCE_RUN3[0] | frozenset(range(94001, 94100)),
                    SINCE_RUN3[1] + ((2 ** 134, 2 ** 134 + 2 ** 20),))
SEED_AUDITS = {'since_run3': SINCE_RUN3, 'since_all_vs_all': SINCE_ALL_VS_ALL}
BT_PRIOR = 0.5   # pseudo-wins given to each side of every pairing so ratings stay finite


# ---------------------------------------------------------------- pairings and seeds

def pairings(agent_count):
    """Every unordered pair (a, b), a < b, in lexicographic order; the position is the pairing index."""
    return list(itertools.combinations(range(agent_count), 2))


def build_pairings(ids, spec):
    """Pairings as (first, second) index pairs. Mode 'all': every unordered pair. Mode 'cross': every agent of group
    'a' against every agent of group 'b', then every unordered pair inside each 'within' group. Self pairs and
    unordered pairs already listed are skipped; the position in the list is the pairing index."""
    index = {agent: k for k, agent in enumerate(ids)}
    if spec['mode'] == 'all':
        return pairings(len(ids))
    if spec['mode'] != 'cross':
        raise ValueError('pairing mode must be all or cross')
    candidates = [(x, y) for x in spec['a'] for y in spec['b']]
    for group in spec.get('within', []):
        candidates += list(itertools.combinations(group['agents'], 2))
    table, seen = [], set()
    for x, y in candidates:
        if x not in index or y not in index:
            raise ValueError(f'unknown agent in pairing {x}, {y}')
        key = frozenset((x, y))
        if x != y and key not in seen:
            seen.add(key)
            table.append((index[x], index[y]))
    return table


def make_plan(table, deal_base, action_exp, count, targets=(15, 30)):
    """An exam plan whose 'blocks' are the pairings: block = pairing index (fx seed formulas apply unchanged)."""
    labels = [f'{a}-{b}' for a, b in table]
    return fx.make_plan('A', deal_base, action_exp, count, labels, targets)


def audit_seeds(plan, earlier=SINCE_RUN3):
    """Refuse if the plan's deal ids or action seeds touch any earlier namespace (incl. the run 3 exam)."""
    fx.audit_plan(plan, *earlier)


# ---------------------------------------------------------------- config

def load_config(path, root=fx.REAL_ROOT):
    """Agents, resolved model identities and plan from the JSON config (relative paths resolve against `root`).
    Refuses unfilled hashes, placeholders, unknown models and duplicate agent ids."""
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    root = Path(root)
    models = json.loads(json.dumps(raw['models']))
    for name, entry in models.items():
        if 'path' in entry:
            fx._check_hash(name, entry.get('sha256'))
            entry['path'] = fx._resolve(root, entry['path'])
        if 'equals' in entry:
            fx._check_hash(name + ' equals', entry['equals'].get('sha256'))
            entry['equals']['path'] = fx._resolve(root, entry['equals']['path'])
        if 'sidecar' in entry:
            fx._check_hash(name + ' sidecar', entry['sidecar'].get('sha256'), allow_none=True)
            entry['sidecar']['path'] = fx._resolve(root, entry['sidecar']['path'])
        if 'last_checked' in entry:
            entry['last_checked']['log'] = fx._resolve(root, entry['last_checked']['log'])
    fx._refuse_placeholders(models)
    agents = raw['agents']
    if len(agents) < 2 or len({a['id'] for a in agents}) != len(agents):
        raise ValueError('need at least two agents with distinct ids')
    for agent in agents:
        if set(agent) != {'id', 'name', 'model'} or agent['model'] not in models:
            raise ValueError(f'invalid agent {agent!r}')
    spec = raw['plan']
    if raw.get('seed_audit') not in SEED_AUDITS:
        raise ValueError('seed_audit must name a known namespace set')
    table = build_pairings([a['id'] for a in agents], raw['pairing'])
    return {'name': raw['name'], 'rule_path': fx._resolve(root, raw['rule']), 'output': fx._resolve(root, raw['output']),
            'agents': agents, 'models': models, 'pairing': raw['pairing'], 'pairings': table,
            'seed_audit': raw['seed_audit'],
            'plan': make_plan(table, spec['deal_base'], spec['action_exp'], spec['count'], tuple(spec['targets']))}


# ---------------------------------------------------------------- workers

_STATE = {}


def build_specs(agents, models):
    """One policy spec per agent in index order; networks are loaded once (the encoder follows the input width) and
    every phase 2 network plays greedy with adaptation on."""
    import numpy as np
    from phase2_torch import Phase2Network
    specs = []
    for agent in agents:
        entry = models[agent['model']]
        kind = entry['kind']
        if kind == 'phase2':
            network = Phase2Network.load(entry['path'])
            network.eval()
            specs.append({'kind': 'phase2', 'model': network, 'temperature': None,
                          'profile_prior': entry['profile_prior']})
        elif kind == 'phase1':
            specs.append({'kind': 'phase1', 'weights': np.load(entry['path'])})
        else:
            specs.append({'kind': kind})
    return specs


def _initialize_worker(config):
    import torch
    torch.set_num_threads(1)
    _STATE['plan'] = config['plan']
    _STATE['pairings'] = [tuple(p) for p in config['pairings']]
    _STATE['specs'] = build_specs(config['agents'], config['models'])


def play_pair(plan, pairing, block, i, specs, policy_factory=fx.make_policy):
    """One mirror pair of a pairing: game 0 seats the lower-index agent first, game 1 second; new policies each game."""
    a, b = pairing
    return fx.play_record(plan, block, i, {'games': specs[a]}, specs[b], policy_factory)


def _work(job):
    block, i = job
    return play_pair(_STATE['plan'], _STATE['pairings'][block], block, i, _STATE['specs'])


def _play_jobs(config, jobs, workers):
    import multiprocessing as mp
    done = 0
    with mp.get_context('spawn').Pool(min(workers, len(jobs)), initializer=_initialize_worker,
                                      initargs=(config,)) as pool:
        for record in pool.imap_unordered(_work, jobs, chunksize=1):
            done += 1
            if done % 100 == 0:
                print(f'pairing {record["block"]}: {done} pairs done in this pass', file=sys.stderr, flush=True)
            yield record


# ---------------------------------------------------------------- analysis

def bradley_terry(wins, prior=BT_PRIOR, iterations=10000, tolerance=1e-12):
    """Bradley-Terry strengths from wins[i][j] = games i won against j (MM / Zermelo iteration).
    Each side of every pair gets `prior` pseudo-wins so strengths stay finite. Returns strengths with geometric
    mean 1 (P(i beats j) = s_i / (s_i + s_j)); descriptive only."""
    n = len(wins)
    w = [[(wins[i][j] + prior) if i != j else 0.0 for j in range(n)] for i in range(n)]
    s = [1.0] * n
    for _ in range(iterations):
        new = []
        for i in range(n):
            denominator = sum((w[i][j] + w[j][i]) / (s[i] + s[j]) for j in range(n) if j != i)
            new.append(sum(w[i]) / denominator)
        center = math.exp(sum(math.log(x) for x in new) / n)
        new = [x / center for x in new]
        change = max(abs(a - b) for a, b in zip(new, s))
        s = new
        if change < tolerance:
            break
    return s


def elo(strength):
    """Elo-scale rating of a Bradley-Terry strength (centered: the mean over agents is 0)."""
    return 400 / math.log(10) * math.log(strength)


def _ordered_stats(scores, table, agent_ids):
    """Both orientations of every pairing: [{row, col, mean, se, lower, upper, n}] and per-agent stat lists."""
    ordered, per_agent = [], {}
    for block, (a, b) in enumerate(table):
        values = scores['games'][block]
        for row, col, series in ((a, b, values), (b, a, [1 - v for v in values])):
            stat = fx.paired_stat(series)
            ordered.append({'row': agent_ids[row], 'col': agent_ids[col], **stat})
            per_agent.setdefault(row, []).append(stat)
    return ordered, per_agent


NOTES = {'interval_note': 'mean +/- 1.96 * sample SD of the pair scores / sqrt(n); unadjusted for multiple '
                          'comparisons; the pair is the independent unit.'}


def _all_mode(ordered, per_agent, table, agent_ids, games):
    n = len(agent_ids)
    wins = [[0.0] * n for _ in range(n)]
    for (a, b), first_orientation in zip(table, ordered[0::2]):
        wins[a][b] = first_orientation['mean'] * games
        wins[b][a] = (1 - first_orientation['mean']) * games
    agents = []
    for k in range(n):
        stats = per_agent[k]
        mean = sum(s['mean'] for s in stats) / len(stats)
        se = math.sqrt(sum(s['se'] ** 2 for s in stats)) / len(stats)   # opponents use disjoint deal seeds
        agents.append({'id': agent_ids[k], 'equal_weight_mean': mean, 'se': se,
                       'lower': mean - Z95 * se, 'upper': mean + Z95 * se, 'opponents': len(stats)})
    strengths = bradley_terry(wins)
    for k, item in enumerate(agents):
        item['bradley_terry_strength'] = strengths[k]
        item['elo'] = elo(strengths[k])
    return {'agents': agents,
            'rating_note': 'Bradley-Terry fitted on all games (each pairing: 2*count games, seats swapped), '
                           f'{BT_PRIOR} pseudo-win per side per pairing, Elo scale with mean 0. Descriptive only: '
                           'ratings assume transitive strength, which intransitive matchups violate. '
                           'No selection or shipping decision follows from it.'}


def _cross_mode(ordered, spec):
    """Per within-group ('run'): a table member x group-b opponent and the within-group matrix (row's score)."""
    lookup = {(o['row'], o['col']): {k: o[k] for k in ('mean', 'se', 'lower', 'upper', 'n')} for o in ordered}
    runs = []
    for group in spec.get('within', []):
        members = group['agents']
        runs.append({'name': group['name'], 'agents': members, 'opponents': spec['b'],
                     'table': {m: {o: lookup.get((m, o)) for o in spec['b']} for m in members},
                     'within': {m: {o: lookup.get((m, o)) for o in members if o != m} for m in members}})
    return {'runs': runs}


def analyze(scores, agent_ids, table=None, spec=None):
    """scores['games'][block][i] = pair score of the first agent of pairing `block`. Descriptive statistics only."""
    table = pairings(len(agent_ids)) if table is None else table
    spec = spec or {'mode': 'all'}
    ordered, per_agent = _ordered_stats(scores, table, agent_ids)
    games = 2 * len(scores['games'][0])
    extra = (_all_mode(ordered, per_agent, table, agent_ids, games) if spec['mode'] == 'all'
             else _cross_mode(ordered, spec))
    return {'pairs': ordered, **extra, **NOTES}


# ---------------------------------------------------------------- run

def run(config, *, root=fx.REAL_ROOT, workers, resume=False, earlier=None, play_jobs=_play_jobs):
    """Play (or resume) the whole round robin. `config` comes from load_config."""
    agents, plan, table = config['agents'], config['plan'], config['pairings']
    ids = [a['id'] for a in agents]
    earlier = earlier or SEED_AUDITS[config['seed_audit']]
    if len(table) != len(plan['opponents']):
        raise ValueError('plan does not match the pairings')

    def check(_config):
        audit_seeds(plan, earlier)

    return fx._run_part(
        part='A', plan=plan, evaluated=[{'name': 'games'}], models=config['models'],
        needed=sorted({a['model'] for a in agents}), root=root, output=config['output'], workers=workers,
        resume=resume, rule_path=config['rule_path'],
        analyze=lambda scores: analyze(scores, ids, table, config['pairing']),
        extra={'agents': agents, 'pairing': config['pairing'], 'pairings': [list(p) for p in table],
               'seed_audit': config['seed_audit'],
               'record_layout': 'block = pairing index; winners.games = [winner seat of game 0, of game 1]; '
                                'game 0 seats the first agent of the pairing first, game 1 second',
               'settings': 'phase2: greedy, adapt=true, default weighted beliefs, profile_prior as recorded, '
                           'encoder by input width; phase1: encoder v1, default sampling and profile; '
                           'AU/Reactive: fixed initial state, fresh history; new policies every game and seat'},
        check=check, play_jobs=play_jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('--workers', type=int, default=fx.MAX_WORKERS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true', help='audit seeds and verify model hashes; play nothing')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.check:
        audit_seeds(config['plan'], SEED_AUDITS[config['seed_audit']])
        fx.verify_models(config['models'], sorted({a['model'] for a in config['agents']}))
        print('seed audit and model identities OK; nothing was played')
        return
    run(config, workers=args.workers, resume=args.resume)


if __name__ == '__main__':
    main()
