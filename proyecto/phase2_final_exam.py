"""One-time phase 2 final exam (ADR 0061): Part T (temperature) and Part E (exam).

The frozen protocol is experiments/phase2-final-exam/RULE.txt. Run each part once.
"""
import hashlib
import json
import math
import os
import statistics
import subprocess
from fractions import Fraction
from pathlib import Path

from symmetric_engine import ENGINE_VERSION, play_match

Z95 = 1.96
GUARDRAIL_THRESHOLD = -0.02
HOLM_ALPHA = 0.025


def pair_score(winners):
    """Evaluated network's wins over both games / 2 (seat 0 in game 0, seat 1 in game 1)."""
    return (int(winners[0] == 0) + int(winners[1] == 1)) / 2


def paired_stat(differences):
    """Mean, SE = sample SD / sqrt(n) and a 95% interval of paired differences."""
    n = len(differences)
    if n < 2:
        raise ValueError('at least two pairs required')
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(n)
    return {'n': n, 'mean': mean, 'se': se, 'lower': mean - Z95 * se, 'upper': mean + Z95 * se}


def one_sided_p(gain, se):
    """p = 1 - Phi(gain / se) with the normal CDF."""
    if se == 0:
        return 0.5 if gain == 0 else (0.0 if gain > 0 else 1.0)
    return 0.5 * math.erfc(gain / se / math.sqrt(2))


def holm(p_values, alpha=HOLM_ALPHA):
    """One-sided Holm over two tests: smaller p at alpha/2, larger p at alpha only if the smaller passed."""
    if len(p_values) != 2:
        raise ValueError('Holm is defined over exactly two candidates')
    (first, p_first), (second, p_second) = sorted(p_values.items(), key=lambda item: item[1])
    rejected_first = p_first <= alpha / 2
    return {first: rejected_first, second: rejected_first and p_second <= alpha}


def guardrail_pass(gain, se):
    """The 95% lower bound must be strictly above -0.02."""
    return gain - Z95 * se > GUARDRAIL_THRESHOLD


CANDIDATES = ('C1', 'C2')
BASELINE = 'B'


def analyze_exam(scores, *, candidates=CANDIDATES, baseline=BASELINE):
    """Part E analysis. scores[name][k] is the list of pair scores of `name` against opponent k (k=0 is AU)."""
    blocks = len(scores[baseline])
    gains, primary_p = {}, {}
    for name in candidates:
        gains[name] = []
        for k in range(blocks):
            stat = paired_stat([c - b for c, b in zip(scores[name][k], scores[baseline][k])])
            gains[name].append({'gain': stat['mean'], 'se': stat['se'], 'lower': stat['lower'],
                                'upper': stat['upper'], 'n': stat['n']})
        primary_p[name] = one_sided_p(gains[name][0]['gain'], gains[name][0]['se'])
    rejected = holm(primary_p)
    result = {'candidates': {}, 'win_rates': {}}
    for name in candidates:
        g = gains[name]
        guardrails = [{'k': k, **g[k], 'pass': guardrail_pass(g[k]['gain'], g[k]['se'])} for k in range(1, blocks)]
        balance = statistics.mean(item['gain'] for item in g)
        z = g[0]['gain'] / g[0]['se'] if g[0]['se'] else (math.inf if g[0]['gain'] > 0 else -math.inf if g[0]['gain'] < 0 else 0.0)
        entry = {'gains': g,
                 'primary': {'z': z, 'p': primary_p[name], 'holm_rejected': rejected[name]},
                 'guardrails': guardrails, 'guardrails_pass': all(x['pass'] for x in guardrails),
                 'balance': {'mean': balance, 'pass': balance > 0}}
        entry['passes'] = bool(entry['primary']['holm_rejected'] and entry['guardrails_pass'] and entry['balance']['pass'])
        result['candidates'][name] = entry
    for name in (*candidates, baseline):
        result['win_rates'][name] = []
        for k in range(blocks):
            stat = paired_stat(scores[name][k])
            result['win_rates'][name].append({**stat, 'above_half': stat['mean'] > 0.5,
                                              'lower_above_half': stat['lower'] > 0.5})
    result['decision'] = next((name for name in candidates if result['candidates'][name]['passes']), baseline)
    return result


TEMPERATURES = ('0.05', '0.02', 'greedy')  # tie-break preference order: toward 0.05, then 0.02
TEMPERATURE_MIN_GAIN = Fraction(2, 100)


def temperature_value(name):
    """Sampling temperature for a name in TEMPERATURES, or None for greedy."""
    return None if name == 'greedy' else float(name)


def _exact_mean(values):
    return sum((Fraction(v) for v in values), Fraction(0)) / len(values)


def analyze_temperature(scores):
    """Part T analysis. scores[name][t] is the list of pair scores of C1 at that temperature against block t
    (0 = AU, 1 = P1). m(T) is the equal-weight mean of the two block means, computed exactly."""
    m = {name: (_exact_mean(scores[name][0]) + _exact_mean(scores[name][1])) / 2 for name in TEMPERATURES}
    best = max(m.values())
    tstar = next(name for name in TEMPERATURES if m[name] == best)
    changed = tstar != '0.05' and m[tstar] - m['0.05'] >= TEMPERATURE_MIN_GAIN
    paired = {}
    for name in TEMPERATURES[1:]:
        per_opponent = [paired_stat([a - b for a, b in zip(scores[name][t], scores['0.05'][t])]) for t in (0, 1)]
        se = math.sqrt(sum(stat['se'] ** 2 for stat in per_opponent)) / 2
        mean = float(m[name] - m['0.05'])
        paired[name] = {'per_opponent': per_opponent,
                        'm_diff': {'mean': mean, 'se': se, 'lower': mean - Z95 * se, 'upper': mean + Z95 * se}}
    return {'m': {name: float(value) for name, value in m.items()},
            'tstar': tstar if changed else '0.05', 'changed': bool(changed),
            'paired_vs_0.05': paired,
            'win_rates': {name: [paired_stat(scores[name][t]) for t in (0, 1)] for name in TEMPERATURES}}


# ---------------------------------------------------------------- seeds and records

def make_plan(part, deal_base, action_exp, count, opponents, targets=(15, 30)):
    return {'part': part, 'deal_base': deal_base, 'action_exp': action_exp, 'count': count,
            'targets': list(targets), 'opponents': list(opponents)}


PART_T_PLAN = make_plan('T', 90001, 131, 300, ['AU', 'P1'])
PART_E_PLAN = make_plan('E', 91001, 130, 1000, ['AU', 'R', 'P1', 'H1', 'H2'])


def deal_seed(plan, block, i):
    return 2 * ((plan['deal_base'] + block) * 10 ** 9 + i)


def action_seed(plan, block, i, game, seat):
    return 2 ** plan['action_exp'] + (((block * plan['count'] + i) * 2 + game) * 2 + seat)


def pair_target(plan, i):
    return plan['targets'][i % 2]


def all_seeds(plan):
    blocks, count = len(plan['opponents']), plan['count']
    deals = {deal_seed(plan, b, i) for b in range(blocks) for i in range(count)}
    actions = {action_seed(plan, b, i, g, s) for b in range(blocks) for i in range(count)
               for g in range(2) for s in range(2)}
    return deals, actions


def seed_ranges(plan):
    deals, actions = all_seeds(plan)
    return (min(deals), max(deals)), (min(actions), max(actions))


# Earlier namespaces (RULE.txt SEED AUDIT). A schedule seed s owns deal ids 2*(s*10^9 + p) for p < 10^9.
# Training and check seeds get a +999 allowance for per-generation offsets.
EARLIER_SCHEDULE_SEEDS = frozenset(
    {0, 1, 2, 3, 777, 4242, 9001, 9002, 9101, 9201, 9202, *range(9211, 9217), 9221, 9222, *range(9301, 9329),
     *range(9001, 9329), *range(9211, 9223), *range(39171, 39174), 47000, 49000, *range(50000, 50018),
     *range(53000, 53005), *range(55000, 55046), *range(58000, 59000), *range(60001, 60100),
     61001, 62000, *range(70001, 70007), 80001,
     *(s + o for s in (42001, 52001, 54001, 57001) for o in range(1000)), *range(43000, 44000)})
EARLIER_ACTION_RANGES = ((0, 2 ** 32 - 1),                    # small seeds and SeedSequence-derived words
                         (2 ** 128, 2 ** 128 + 23999),        # round robin
                         (2 ** 129, 2 ** 129 + 2 ** 20))      # probe


# Namespaces used after the run 2 exam was frozen (run 3 audit): equity 67001/67002, pilot 63001, run 3 training 65001
# (+999 allowance), run 3 checks 66000-66999, run 3 pilot deal base 92001, the run 2 exam's own schedule seeds, and the
# action exponents 130-132 (experiments 130/131 = run 2 exam parts, 132 = run 3 pilot).
SINCE_RUN2_SCHEDULE_SEEDS = frozenset({63001, 67001, 67002, 92001, 90001, 90002, *range(91001, 91006),
                                       *range(65001, 66000), *range(66000, 67000)})
SINCE_RUN2_ACTION_RANGES = ((2 ** 130, 2 ** 130 + 19999), (2 ** 131, 2 ** 131 + 2399), (2 ** 132, 2 ** 132 + 1199))
SEED_AUDITS = {'run2': (EARLIER_SCHEDULE_SEEDS, EARLIER_ACTION_RANGES),
               'since_run2': (EARLIER_SCHEDULE_SEEDS | SINCE_RUN2_SCHEDULE_SEEDS,
                              EARLIER_ACTION_RANGES + SINCE_RUN2_ACTION_RANGES)}


def audit_plan(plan, schedule_seeds, action_ranges):
    """Refuse if a plan's deal ids or action seeds overlap the given earlier schedule seeds or action ranges."""
    (deal_lo, deal_hi), (action_lo, action_hi) = seed_ranges(plan)
    for seed in sorted(schedule_seeds):
        if not (2 * (seed * 10 ** 9 + 10 ** 9 - 1) < deal_lo or 2 * seed * 10 ** 9 > deal_hi):
            raise ValueError(f'plan {plan["part"]}: deal ids overlap earlier schedule seed {seed}')
    for lo, hi in action_ranges:
        if not (hi < action_lo or lo > action_hi):
            raise ValueError(f'plan {plan["part"]}: action seeds overlap earlier range {lo}-{hi}')


def audit_exam_seeds(exam):
    """Refuse if a plan's deal ids or action seeds overlap an earlier namespace or another plan of the same exam."""
    schedule_seeds, action_ranges = SEED_AUDITS[exam['seed_audit']]
    plans = [plan for plan in (exam['part_t'], exam['part_e']) if plan]
    for plan in plans:
        audit_plan(plan, schedule_seeds, action_ranges)
    for first, second in zip(plans, plans[1:]):
        (a_deals, a_actions), (b_deals, b_actions) = all_seeds(first), all_seeds(second)
        if a_deals & b_deals or a_actions & b_actions:
            raise ValueError('the exam parts share seeds')


def expected_record(plan, block, i):
    return {'block': block, 'i': i, 'deal_seed': deal_seed(plan, block, i), 'target': pair_target(plan, i),
            'action_seeds': [[str(action_seed(plan, block, i, g, s)) for s in range(2)] for g in range(2)]}


def validate_records(records, plan, names, *, complete=True):
    """Check every record against the plan; return the set of (block, i) seen."""
    blocks, count = len(plan['opponents']), plan['count']
    seen = set()
    for row in records:
        b, i = row.get('block'), row.get('i')
        if type(b) is not int or type(i) is not int or not 0 <= b < blocks or not 0 <= i < count:
            raise ValueError('out-of-range pair')
        if (b, i) in seen:
            raise ValueError('duplicate pair')
        seen.add((b, i))
        winners = row.get('winners')
        if (not isinstance(winners, dict) or sorted(winners) != sorted(names)
                or any(not isinstance(w, list) or len(w) != 2 or any(type(x) is not int or x not in (0, 1) for x in w)
                       for w in winners.values())
                or {k: v for k, v in row.items() if k != 'winners'} != expected_record(plan, b, i)):
            raise ValueError('invalid pair record')
    if complete and len(seen) != blocks * count:
        raise ValueError('incomplete pairs')
    return seen


# ---------------------------------------------------------------- playing

def make_policy(spec, seed):
    """Fresh policy for one seat of one game. Specs: phase2 (model, temperature or greedy), phase1, au, reactive."""
    kind = spec['kind']
    if kind == 'phase2':
        from phase2_torch import Phase2Policy
        if spec['temperature'] is None:
            return Phase2Policy(spec['model'], seed=seed, adapt=True, greedy=True, profile_prior=spec['profile_prior'])
        return Phase2Policy(spec['model'], seed=seed, adapt=True, temperature=spec['temperature'],
                            profile_prior=spec['profile_prior'])
    if kind == 'phase1':
        from phase1_network import NetworkPolicy
        return NetworkPolicy(spec['weights'], seed=seed, encoder='v1')
    from engine_policies import AdaptiveUtilityPolicy, ReactivePolicy
    return {'au': AdaptiveUtilityPolicy, 'reactive': ReactivePolicy}[kind](seed)


def play_record(plan, block, i, evaluated, opponent, policy_factory=make_policy):
    """One pair for every evaluated network: same deal and action seeds, new policies for every game and seat.
    Game 0 seats the evaluated network first, game 1 second. Returns the raw record."""
    record = expected_record(plan, block, i)
    seeds = [[int(x) for x in game] for game in record['action_seeds']]
    winners = {}
    for name, spec in evaluated.items():
        winners[name] = []
        for game in range(2):
            specs = (spec, opponent) if game == 0 else (opponent, spec)
            policies = [policy_factory(s, seeds[game][seat]) for seat, s in enumerate(specs)]
            result = play_match(*policies, seed=record['deal_seed'], target=record['target'],
                                starting_score=(0, 0), deal_partition=0)
            winners[name].append(int(result.winner))
    record['winners'] = winners
    return record


# ---------------------------------------------------------------- identities and provenance

MAX_WORKERS = 12
REAL_ROOT = Path(__file__).resolve().parent.parent
RULE_RELATIVE = 'experiments/phase2-final-exam/RULE.txt'


def real_models(root, training_dir):
    """Frozen identities from RULE.txt. Every file is hashed before play; any mismatch refuses the run."""
    root, training_dir = Path(root), Path(training_dir)

    def file(path, sha256):
        return {'path': str(path), 'sha256': sha256}
    c1_dir = root / 'champions/phase2-run2'
    b_dir = root / 'champions/phase2'
    return {
        'C1': {'kind': 'phase2', **file(c1_dir / 'reigning_champion.pt',
                                        '95eee708e8839233c27e4b3d710b6b5da74f3a149f7fe5ce8deef4cb720b60bf'),
               'profile_prior': 5,
               'equals': file(training_dir / 'generation_000038.pt',
                              '927f462178ab03048b3c9e76567d5a9226d3fe329f0bd673e842a8aa067790e7'),
               'sidecar': {'path': str(c1_dir / 'reigning_champion.json'), 'sha256': None,
                           'require': {'generation': 38, 'profile_prior': 5}}},
        'C2': {'kind': 'phase2', **file(training_dir / 'generation_000049.pt',
                                        'e789d7b1a171ab73e33766101f8bbf62204649d79db0cb9b20052acfd5264f44'),
               'profile_prior': 5},
        'B': {'kind': 'phase2', **file(b_dir / 'reigning_champion.pt',
                                       '05f189491a1d6ca7a4d7cebfa556a16e70d7512028e6bb224d61d87a35d4cdcd'),
              'profile_prior': 5, 'temperature': 0.05,
              'sidecar': {'path': str(b_dir / 'reigning_champion.json'), 'sha256': None,
                          'require': {'temperature': 0.05, 'profile_prior': 5}}},
        'AU': {'kind': 'au'},
        'R': {'kind': 'reactive'},
        'P1': {'kind': 'phase1', **file(root / 'champions/phase1/reigning_champion.npy',
                                        '0b3e04fad09334f60d2d416b9c6848c7b3aabcfa6bb0d6593ac825feb32e47f7')},
        'H1': {'kind': 'phase1', **file(root / 'experiments/phase1-seed1/reigning_champion.npy',
                                        '413542520b893418371fa0ff29b4b693d0675ef60cd4be55f8e782dd544db31c')},
        'H2': {'kind': 'phase1', **file(root / 'experiments/phase1-seed2/reigning_champion.npy',
                                        '5f00bfbaa27d7ee9fcedd7610ca36b092c71b8e935218a8f337a7cc1e25a3e79')},
    }


_HEX64 = frozenset('0123456789abcdef')


def _resolve(root, relative):
    return str(Path(root) / relative)


def _check_hash(where, value, allow_none=False):
    if value is None and allow_none:
        return
    if not (isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64):
        raise ValueError(f'{where}: sha256 is not a filled-in 64-digit hash ({value!r})')


def _refuse_placeholders(node):
    if isinstance(node, dict):
        for value in node.values():
            _refuse_placeholders(value)
    elif isinstance(node, str) and ('TODO' in node or 'NNNNNN' in node):
        raise ValueError(f'config still has a placeholder: {node!r}')


def load_exam_config(path, root=REAL_ROOT):
    """An exam config (JSON, paths relative to `root`) as plans, resolved model identities and evaluated networks.
    Refuses placeholder hashes and evaluated networks without an explicit temperature (null = greedy)."""
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    root = Path(root)
    plans = {}
    for key, part in (('part_t', 'T'), ('part_e', 'E')):
        spec = raw.get(key)
        plans[key] = (make_plan(part, spec['deal_base'], spec['action_exp'], spec['count'], spec['opponents'],
                                tuple(spec['targets'])) if spec else None)
    if plans['part_e'] is None:
        raise ValueError('config has no part_e')
    models = json.loads(json.dumps(raw['models']))
    for name, entry in models.items():
        if 'path' in entry:
            _check_hash(name, entry.get('sha256'))
            entry['path'] = _resolve(root, entry['path'])
        if 'equals' in entry:
            _check_hash(name + ' equals', entry['equals'].get('sha256'))
            entry['equals']['path'] = _resolve(root, entry['equals']['path'])
        if 'sidecar' in entry:
            _check_hash(name + ' sidecar', entry['sidecar'].get('sha256'), allow_none=True)
            entry['sidecar']['path'] = _resolve(root, entry['sidecar']['path'])
        if 'last_checked' in entry:
            entry['last_checked']['log'] = _resolve(root, entry['last_checked']['log'])
    _refuse_placeholders(models)
    evaluated = raw['evaluated']
    for item in evaluated:
        if set(item) != {'name', 'model', 'temperature'}:
            raise ValueError('evaluated networks need name, model and an explicit temperature (null = greedy)')
        if item['model'] not in models:
            raise ValueError(f'unknown model {item["model"]}')
        t = item['temperature']
        if t == 'tstar' and plans['part_t'] is None:
            raise ValueError('temperature "tstar" needs a Part T')
        if not (t is None or t == 'tstar' or (type(t) in (int, float) and t > 0)):
            raise ValueError('temperature must be a positive number, null (greedy) or "tstar"')
    if raw.get('seed_audit') not in SEED_AUDITS:
        raise ValueError('seed_audit must name a known namespace set')
    return {'name': raw['name'], 'rule_path': _resolve(root, raw['rule']), 'seed_audit': raw['seed_audit'],
            'part_t': plans['part_t'], 'part_e': plans['part_e'], 'models': models, 'evaluated': evaluated,
            'output_t': _resolve(root, raw['output_t']) if raw.get('output_t') else None,
            'output_e': _resolve(root, raw['output_e'])}


def resolve_evaluated(evaluated, tstar=None):
    """Evaluated networks with "tstar" replaced by Part T's temperature (None = greedy)."""
    t = None if tstar is None else temperature_value(tstar)
    return [{**item, 'temperature': t if item['temperature'] == 'tstar' else item['temperature']}
            for item in evaluated]


def _input_width(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)['layers.0.weight'].shape[1]


def last_checked_generation(log_path):
    """Generation number of the last row of a trainer generations.jsonl that carries a check."""
    try:
        rows = [json.loads(line) for line in Path(log_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise ValueError('generations log unreadable') from exc
    checked = [row['generation'] for row in rows if row.get('check')]
    if not checked:
        raise ValueError('generations log has no check')
    return checked[-1]


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_tensors(first, second):
    import torch
    a = torch.load(first, map_location='cpu', weights_only=True)
    b = torch.load(second, map_location='cpu', weights_only=True)
    return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
            and all(torch.is_tensor(a[key]) and torch.is_tensor(b[key])
                    and a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]) for key in a))


def verify_models(models, names):
    """Check hashes, snapshot tensor equality and sidecar contents of the named models; return their records.
    Sidecars whose expected hash is None are only checked for content and are hashed into the record."""
    verified = {}
    for name in names:
        entry = json.loads(json.dumps(models[name]))
        if 'path' in entry:
            if not Path(entry['path']).is_file() or _hash(entry['path']) != entry['sha256']:
                raise ValueError(f'{name}: model hash mismatch')
        if 'input_size' in entry and _input_width(entry['path']) != entry['input_size']:
            raise ValueError(f'{name}: network input width must be {entry["input_size"]}')
        if 'last_checked' in entry:
            expected = entry['last_checked']['generation']
            if (last_checked_generation(entry['last_checked']['log']) != expected
                    or Path(entry['path']).name != f'generation_{expected:06d}.pt'):
                raise ValueError(f'{name}: not the last generation with a check ({expected})')
        if 'equals' in entry:
            other = entry['equals']
            if not Path(other['path']).is_file() or _hash(other['path']) != other['sha256']:
                raise ValueError(f'{name}: snapshot hash mismatch')
            if not _same_tensors(entry['path'], other['path']):
                raise ValueError(f'{name}: tensors differ from the snapshot')
        if 'sidecar' in entry:
            sidecar = entry['sidecar']
            path = Path(sidecar['path'])
            if not path.is_file():
                raise ValueError(f'{name}: sidecar missing')
            actual = _hash(path)
            if sidecar.get('sha256') not in (None, actual):
                raise ValueError(f'{name}: sidecar hash mismatch')
            sidecar['sha256'] = actual
            try:
                content = json.loads(path.read_text(encoding='utf-8'))
            except ValueError as exc:
                raise ValueError(f'{name}: sidecar unreadable') from exc
            for key, value in sidecar['require'].items():
                if not isinstance(content, dict) or content.get(key) != value:
                    raise ValueError(f'{name}: sidecar {key} must be {value}')
        verified[name] = entry
    return verified


def _git(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError('git verification failed: ' + result.stderr.strip())
    return result.stdout


def provenance(root, rule_path):
    """Refuse on uncommitted changes; return git revision and hashes of every project Python file and the rule."""
    root, rule_path = Path(root).resolve(), Path(rule_path).resolve()
    rule_relative = rule_path.relative_to(root).as_posix()
    if _git(root, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('working tree has uncommitted changes to tracked files')
    if _git(root, 'status', '--porcelain', '--untracked-files=all', '--', 'proyecto', rule_relative):
        raise ValueError('project sources or the rule have uncommitted or untracked files')
    sources = sorted(p for p in (root / 'proyecto').rglob('*.py') if '__pycache__' not in p.parts)
    return {'git_revision': _git(root, 'rev-parse', 'HEAD').strip(),
            'sha256': {p.relative_to(root).as_posix(): _hash(p) for p in sources},
            'rule_sha256': _hash(rule_path)}


# ---------------------------------------------------------------- workers

_STATE = {}


def _build_specs(config):
    import numpy as np
    from phase2_torch import Phase2Network
    models, loaded = config['models'], {}

    def spec(name, temperature=None):
        entry = models[name]
        kind = entry['kind']
        if kind == 'phase2':
            if name not in loaded:
                loaded[name] = Phase2Network.load(entry['path'])
                loaded[name].eval()
            return {'kind': 'phase2', 'model': loaded[name], 'temperature': temperature,
                    'profile_prior': entry['profile_prior']}
        if kind == 'phase1':
            return {'kind': 'phase1', 'weights': np.load(entry['path'])}
        return {'kind': kind}
    evaluated = {item['name']: spec(item['model'], item['temperature']) for item in config['evaluated']}
    opponents = [spec(name) for name in config['plan']['opponents']]
    return evaluated, opponents


def _initialize_worker(config):
    import torch
    torch.set_num_threads(1)
    _STATE['plan'] = config['plan']
    _STATE['evaluated'], _STATE['opponents'] = _build_specs(config)


def _work(job):
    block, i = job
    return play_record(_STATE['plan'], block, i, _STATE['evaluated'], _STATE['opponents'][block])


def _play_jobs(config, jobs, workers):
    """Yield raw records for the (block, i) jobs as they finish."""
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(min(workers, len(jobs)), initializer=_initialize_worker,
                                      initargs=(config,)) as pool:
        yield from pool.imap_unordered(_work, jobs, chunksize=1)


# ---------------------------------------------------------------- durable part runner

def _fsync_write(path, text, mode='x'):
    with Path(path).open(mode, encoding='utf-8', newline='\n') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    _fsync_write(temp, json.dumps(value, indent=2, sort_keys=True) + '\n', 'w')
    os.replace(temp, path)


def canonical_raw(records):
    """Canonical text of the raw records: sorted by (block, i), one sorted-key JSON line each."""
    return ''.join(json.dumps(r, sort_keys=True) + '\n' for r in sorted(records, key=lambda r: (r['block'], r['i'])))


def raw_digest(records):
    return hashlib.sha256(canonical_raw(records).encode('utf-8')).hexdigest()


def read_partial(path, plan, names):
    """Valid records of an interrupted log. A torn last line is dropped (and cut from the file)."""
    path = Path(path)
    if not path.exists():
        return {}
    lines = path.read_bytes().splitlines(keepends=True)
    complete = [line for line in lines if line.endswith(b'\n')]
    if complete != lines[:len(complete)] or len(lines) - len(complete) > 1:
        raise ValueError('corrupt partial log')
    try:
        records = [json.loads(line) for line in complete]
    except ValueError as exc:
        raise ValueError('corrupt partial log') from exc
    validate_records(records, plan, names, complete=False)
    if len(complete) != len(lines):
        path.write_bytes(b''.join(complete))
    return {(r['block'], r['i']): r for r in records}


def _scores(records, plan, names):
    blocks = len(plan['opponents'])
    table = {name: [[None] * plan['count'] for _ in range(blocks)] for name in names}
    for r in records:
        for name in names:
            table[name][r['block']][r['i']] = pair_score(r['winners'][name])
    return table


def _run_part(*, part, plan, evaluated, models, needed, root, output, workers, resume, rule_path, analyze, extra=None,
              check=None, play_jobs=None):
    play_jobs = play_jobs or _play_jobs
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f'workers must be between 1 and {MAX_WORKERS}')
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(str(output))
    if resume and not (output / 'started.lock').is_file():
        raise ValueError('nothing to resume: no started part at ' + str(output))
    if resume and (output / 'complete.txt').exists():
        raise ValueError('part already complete')
    source = provenance(root, rule_path)
    verified = verify_models(models, needed)
    names = [item['name'] for item in evaluated]
    config = {'part': part, 'plan': plan, 'evaluated': evaluated, 'models': verified, **source,
              'max_workers': MAX_WORKERS, 'engine_version': ENGINE_VERSION, 'deal_partition': 0,
              'starting_score': [0, 0], 'record_layout': 'winners[name] = [winner of game 0, winner of game 1]; '
              'game 0 seats the evaluated network first, game 1 second',
              'settings': 'phase2: adapt=true, default weighted beliefs, profile_prior as recorded; '
                          'phase1: encoder v1, default sampling and profile; AU/R: fixed initial state, fresh history',
              **(extra or {})}
    if check:
        check(config)
    config = json.loads(json.dumps(config))
    if resume:
        saved = json.loads((output / 'config.json').read_text(encoding='utf-8'))
        if saved != config:
            raise ValueError('resume configuration mismatch')
    else:
        output.mkdir(parents=True, exist_ok=False)
        _fsync_write(output / 'started.lock', 'One run started; do not retry. An interrupted run may only be '
                                              'resumed to finish its missing pairs.\n')
        _atomic_json(output / 'config.json', config)
    partial = output / 'partial.jsonl'
    done = read_partial(partial, plan, names)
    blocks, count = len(plan['opponents']), plan['count']
    written = 0
    with partial.open('a', encoding='utf-8', newline='\n') as stream:
        for block in range(blocks):
            jobs = [(block, i) for i in range(count) if (block, i) not in done]
            if jobs:
                for record in play_jobs(config, jobs, workers):
                    stream.write(json.dumps(record, sort_keys=True) + '\n')
                    stream.flush()
                    written += 1
                    if written % 32 == 0:
                        os.fsync(stream.fileno())
                    done[(record['block'], record['i'])] = record
            stream.flush()
            os.fsync(stream.fileno())
    records = list(done.values())
    validate_records(records, plan, names)
    analysis = analyze(_scores(records, plan, names))
    result = {'config': config, 'raw_file': 'partial.jsonl', 'raw_sha256': raw_digest(records),
              'pair_count': len(records), 'analysis': analysis}
    _atomic_json(output / 'results.json', result)
    _fsync_write(output / 'complete.txt', f'Completed and validated all {blocks * count} unique pairs.\n')
    return result


def part_t_evaluated():
    return [{'name': name, 'model': 'C1', 'temperature': temperature_value(name)} for name in TEMPERATURES]


def part_e_evaluated(tstar, models):
    t = temperature_value(tstar)
    return [{'name': 'C1', 'model': 'C1', 'temperature': t}, {'name': 'C2', 'model': 'C2', 'temperature': t},
            {'name': 'B', 'model': 'B', 'temperature': models['B']['temperature']}]


def run_part_t(*, root=REAL_ROOT, output, workers, resume=False, models=None, plan=PART_T_PLAN, rule_path=None,
               training_dir=None):
    root = Path(root)
    models = models or real_models(root, training_dir or root / 'training_output/phase2-run2')
    return _run_part(part='T', plan=plan, evaluated=part_t_evaluated(), models=models,
                     needed=['C1', *plan['opponents']], root=root, output=output, workers=workers, resume=resume,
                     rule_path=rule_path or root / RULE_RELATIVE, analyze=analyze_temperature)


def resolve_tstar(part_t_output, plan_t, temperature=None):
    """T* recomputed from a complete Part T's raw records; refuses if it disagrees with its results or `temperature`."""
    folder = Path(part_t_output)
    if not (folder / 'complete.txt').is_file() or not (folder / 'results.json').is_file():
        raise ValueError('Part T is not complete')
    results = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
    saved = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if saved['part'] != 'T' or saved['plan'] != plan_t or results['config'] != saved:
        raise ValueError('Part T plan or configuration mismatch')
    try:
        records = [json.loads(line) for line in (folder / 'partial.jsonl').read_text(encoding='utf-8').splitlines()]
    except ValueError as exc:
        raise ValueError('Part T raw log unreadable') from exc
    validate_records(records, plan_t, list(TEMPERATURES))
    if raw_digest(records) != results['raw_sha256']:
        raise ValueError('Part T raw records do not match its results')
    analysis = analyze_temperature(_scores(records, plan_t, list(TEMPERATURES)))
    if analysis['tstar'] != results['analysis']['tstar']:
        raise ValueError('Part T results disagree with its raw records')
    if temperature is not None and str(temperature) != analysis['tstar']:
        raise ValueError(f'explicit temperature {temperature} is not Part T\'s T* {analysis["tstar"]}')
    return analysis['tstar'], results['raw_sha256'], saved


def run_part_e(*, root=REAL_ROOT, output, part_t_output=None, workers, resume=False, models=None, plan=PART_E_PLAN,
               plan_t=PART_T_PLAN, rule_path=None, temperature=None, training_dir=None, evaluated=None):
    """Part E. With a Part T folder, T* is recomputed from its raw records; without one (ADR 0062: every network
    greedy) `evaluated` must give each network's temperature explicitly (null = greedy)."""
    root = Path(root)
    models = models or real_models(root, training_dir or root / 'training_output/phase2-run2')
    if part_t_output is None:
        if evaluated is None:
            raise ValueError('without a Part T the evaluated networks must be given explicitly')
        if any(item['temperature'] == 'tstar' for item in evaluated):
            raise ValueError('temperature "tstar" needs a Part T')
        extra, check, run_evaluated = None, None, evaluated
    else:
        tstar, t_digest, t_config = resolve_tstar(part_t_output, plan_t, temperature)
        run_evaluated = (resolve_evaluated(evaluated, tstar) if evaluated is not None
                         else part_e_evaluated(tstar, models))
        extra = {'tstar': tstar, 'part_t_raw_sha256': t_digest, 'part_t_revision': t_config['git_revision']}

        def check(config):
            if (t_config['sha256'], t_config['rule_sha256']) != (config['sha256'], config['rule_sha256']):
                raise ValueError('project sources or the rule changed since Part T')
            if any(t_config['models'][n] != config['models'][n] for n in ('C1', 'AU', 'P1') if n in config['models']):
                raise ValueError('model identities changed since Part T')
    needed = list(dict.fromkeys([item['model'] for item in run_evaluated] + list(plan['opponents'])))
    return _run_part(part='E', plan=plan, evaluated=run_evaluated, models=models, needed=needed, root=root,
                     output=output, workers=workers, resume=resume, rule_path=rule_path or root / RULE_RELATIVE,
                     analyze=analyze_exam, extra=extra, check=check)


def run_configured_part(exam, part, *, root=REAL_ROOT, workers, resume=False, output=None):
    """Run Part T or Part E of a loaded exam config after its seed audit."""
    audit_exam_seeds(exam)
    common = dict(root=root, workers=workers, resume=resume, models=exam['models'], rule_path=exam['rule_path'])
    if part == 'part-t':
        if exam['part_t'] is None:
            raise ValueError('this exam has no Part T')
        return run_part_t(output=output or exam['output_t'], plan=exam['part_t'], **common)
    return run_part_e(output=output or exam['output_e'], plan=exam['part_e'], plan_t=exam['part_t'],
                      part_t_output=exam['output_t'] if exam['part_t'] else None, evaluated=exam['evaluated'],
                      **common)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='part', required=True)
    for name in ('part-t', 'part-e'):
        p = sub.add_parser(name)
        p.add_argument('--workers', type=int, required=True)
        p.add_argument('--resume', action='store_true', help='finish only the missing pairs of an interrupted part')
        p.add_argument('--training-dir', default=None)
        p.add_argument('--output', default=None)
        p.add_argument('--config', default=None, help='exam config JSON (default: the hardcoded run 2 exam)')
    exam = Path(REAL_ROOT) / 'experiments/phase2-final-exam'
    sub.choices['part-e'].add_argument('--temperature', default=None,
                                       help='optional check: must equal the T* recomputed from Part T')
    args = parser.parse_args(argv)
    if args.config:
        if args.training_dir or getattr(args, 'temperature', None):
            parser.error('--training-dir and --temperature do not apply with --config')
        result = run_configured_part(load_exam_config(args.config), args.part, workers=args.workers,
                                     resume=args.resume, output=args.output)
        print(json.dumps(result['analysis'], indent=2, sort_keys=True))
        return
    common = dict(workers=args.workers, resume=args.resume, training_dir=args.training_dir)
    if args.part == 'part-t':
        result = run_part_t(output=args.output or exam / 'part-t', **common)
    else:
        result = run_part_e(output=args.output or exam / 'part-e', part_t_output=exam / 'part-t',
                            temperature=args.temperature, **common)
    print(json.dumps(result['analysis'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
