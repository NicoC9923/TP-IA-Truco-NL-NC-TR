"""Synchronous self-play and Monte Carlo learning for phase 2."""
import os
import pickle
import tempfile
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from datetime import datetime, timezone
import copy
import hashlib
import statistics
import json
import math
import numbers
import random
import sys
import time
from pathlib import Path
from contextlib import nullcontext
from collections import OrderedDict
import multiprocessing as mp
from dataclasses import dataclass
import numpy as np
import torch
from phase1_network import ACTIONS, NetworkPolicy
from phase2_torch import Phase2Network, Phase2Policy, combination_scores
from opponent_inference import compact_evidence, compatible_mask, _IDS, _CARD_IDS, _HAND_IDS, _DECK
from symmetric_engine import ENGINE_VERSION, play_match, play_mirror_pair
from phase2_network import ENCODER_VERSION
from validation_schedule import validation_schedule
from phase2_league import retain_snapshots, panel_exclusions, draw_opponent, load_phase1_manifest, validate_shares, DEFAULT_SHARES
from phase2_progress import ProgressReporter, PROGRESS_FILE
from phase2_promotion import load_panel, check_seed, promotion_decision, saturated, build_panel, RESERVED_SEEDS
from phase2_league import REFERENCE_IDENTITIES

def _champion_metadata(generation, temperature, profile_prior):
    return {'generation': generation, 'temperature': temperature, 'profile_prior': profile_prior}


def _atomic_champion_file(path, write):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', dir=path.parent, prefix='.champion-', delete=False) as stream:
            temporary = Path(stream.name)
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _export_champion(directory, weights, generation, temperature, profile_prior):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = _champion_metadata(generation, temperature, profile_prior)
    _atomic_champion_file(directory / 'reigning_champion.pt', lambda stream: torch.save(weights, stream))
    _atomic_champion_file(directory / 'reigning_champion.json',
                          lambda stream: stream.write((json.dumps(metadata) + '\n').encode('utf-8')))


def _reconcile_champion(directory, weights, generation, temperature, profile_prior):
    directory = Path(directory)
    try:
        metadata = json.loads((directory / 'reigning_champion.json').read_text(encoding='utf-8'))
        exported = torch.load(directory / 'reigning_champion.pt', map_location='cpu', weights_only=True)
        correct = (metadata == _champion_metadata(generation, temperature, profile_prior)
                   and isinstance(exported, dict) and exported.keys() == weights.keys()
                   and all(torch.is_tensor(exported[k]) and exported[k].dtype == value.dtype
                           and torch.equal(exported[k], value)
                           for k, value in weights.items()))
    except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError, TypeError):
        correct = False
    if not correct:
        _export_champion(directory, weights, generation, temperature, profile_prior)


def _configure_threads():
    torch.set_num_threads(1)
    # PyTorch allows setting inter-op threads only once, before parallel work.
    # Repeated calls are safe when this process is already configured.
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError('Phase 2 requires one inter-op thread before parallel work') from exc

class RecordingPolicy(Phase2Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = []

    def record_decision(self, inputs, action):
        self.samples.append((inputs, ACTIONS.index(action)))

def training_seed(seed, match_id):
    """Assign a distinct deterministic seed to each training match."""
    # Cantor pairing is injective, unlike a truncated hash of the identifiers.
    diagonal = seed + match_id
    return 2 * (diagonal * (diagonal + 1) // 2 + match_id) + 1

def match_settings(seed, match_id, target='mixed', random_start_share=.3):
    rng = np.random.default_rng(np.random.SeedSequence([seed, match_id, 2]))
    match_target = int(rng.choice((15, 30))) if target == 'mixed' else target
    score = tuple(int(x) for x in rng.integers(match_target, size=2)) if rng.random() < random_start_share else (0, 0)
    return match_target, score

def _matches(args):
    weights, seed, ids, target, random_start_share = args
    model = Phase2Network(auxiliary='card_head.0.weight' in weights)
    model.load_state_dict(weights)
    results = []
    for match_id in ids:
        seeds = np.random.SeedSequence([seed, match_id, 1]).generate_state(2)
        policies = [RecordingPolicy(model, seed=int(seeds[i])) for i in range(2)]
        match_target, score = match_settings(seed, match_id, target, random_start_share)
        result = play_match(*policies, seed=training_seed(seed, match_id), target=match_target, starting_score=score, deal_partition=1)
        samples = [(x, action, 1.0 if seat == result.winner else -1.0, seat)
                   for seat, policy in enumerate(policies) for x, action in policy.samples]
        if hasattr(model, 'card_head'):
            decisions = [[d for d in result.decisions if d['player'] == seat] for seat in range(2)]
            offsets = [0, 0]
            enriched = []
            for sample in samples:
                seat = sample[3]
                decision = decisions[seat][offsets[seat]]
                offsets[seat] += 1
                hand = result.deals[decision['hand_index']][1 - seat]
                ordered = tuple(sorted(hand, key=_CARD_IDS.__getitem__))
                enriched.append((*sample, compact_evidence(decision['observation']), _HAND_IDS[ordered]))
            samples = enriched
        results.append((match_id, samples))
    return results

# Each spawned actor owns this cache; eviction bounds memory during long runs.
_OPPONENT_CACHE = OrderedDict()


def _load_league_opponent(identity, member, category):
    key = (identity, 'phase1' if category == 'phase1' else 'phase2')
    if key not in _OPPONENT_CACHE:
        path = Path(member['resolved_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity:
            raise ValueError('league opponent hash mismatch')
        _OPPONENT_CACHE[key] = np.load(path) if category == 'phase1' else Phase2Network.load(path)
        if len(_OPPONENT_CACHE) > 32:
            _OPPONENT_CACHE.popitem(last=False)
    _OPPONENT_CACHE.move_to_end(key)
    return _OPPONENT_CACHE[key]


def _league_block(args):
    if len(args) == 8:
        args = (*args, 20)
    weights, seed, ids, target, random_start_share, spec, opponents, temperature, profile_prior = args
    model = Phase2Network(auxiliary='card_head.0.weight' in weights)
    model.load_state_dict(weights)
    identity = spec['identity']
    opponent_model = model
    is_reference = spec['category'] in REFERENCE_IDENTITIES
    if is_reference and identity != REFERENCE_IDENTITIES[spec['category']]:
        raise ValueError('reference opponent identity mismatch')
    if identity != 'current' and not is_reference:
        member = opponents[identity]
        opponent_model = _load_league_opponent(identity, member, spec['category'])
    seeds = np.random.SeedSequence([seed, ids[0], 81]).generate_state(2)
    learner = RecordingPolicy(model, seed=int(seeds[0]), temperature=temperature, profile_prior=profile_prior, adapt=not spec['neutral_profile'])
    if is_reference:
        from engine_policies import reference_policy
        opponent = reference_policy(spec['category'], int(seeds[1]))
    elif spec['category'] == 'phase1':
        opponent = NetworkPolicy(opponent_model, seed=int(seeds[1]), encoder=member['encoder'])
    else:
        cls = RecordingPolicy if spec['category'] == 'self' else Phase2Policy
        opponent = cls(opponent_model, seed=int(seeds[1]), temperature=temperature, profile_prior=profile_prior,
                       style=spec['style'], style_strength=spec['offset'],
                       adapt=not spec['neutral_profile'] if spec['category'] == 'self' else True)
    results = []
    for match_id in ids:
        policies = (learner, opponent) if match_id % 2 == 0 else (opponent, learner)
        for policy in policies:
            if isinstance(policy, RecordingPolicy):
                policy.samples = []
            policy_seed = np.random.SeedSequence([seed, match_id, 1, policies.index(policy)])
            if is_reference and policy is opponent:
                policy.agent.rng = random.Random(int(policy_seed.generate_state(1)[0]))
            else:
                policy.rng = np.random.default_rng(policy_seed)
        match_target, score = match_settings(seed, match_id, target, random_start_share)
        result = play_match(*policies, seed=training_seed(seed, match_id), target=match_target, starting_score=score, deal_partition=1)
        samples = [(x, action, 1.0 if seat == result.winner else -1.0, seat)
                   for seat, policy in enumerate(policies) if isinstance(policy, RecordingPolicy)
                   for x, action in policy.samples]
        if hasattr(model, 'card_head'):
            decisions = [[d for d in result.decisions if d['player'] == seat] for seat in range(2)]
            offsets = [0, 0]
            enriched = []
            for sample in samples:
                seat = sample[3]
                decision = decisions[seat][offsets[seat]]
                offsets[seat] += 1
                hand = result.deals[decision['hand_index']][1 - seat]
                ordered = tuple(sorted(hand, key=_CARD_IDS.__getitem__))
                enriched.append((*sample, compact_evidence(decision['observation']), _HAND_IDS[ordered]))
            samples = enriched
        results.append((match_id, samples))
    return results


def _league_chunk(job):
    weights, seed, blocks, target, random_start_share, opponents, temperature, profile_prior = job
    return [result for ids, spec in blocks for result in
            _league_block((weights, seed, ids, target, random_start_share, spec, opponents, temperature, profile_prior))]


def _check_pair(job):
    challenger, opponent, deal_seed, target, action_seed, temperature = job
    def policy(spec, seed):
        if spec['kind'] == 'phase2':
            model = Phase2Network(auxiliary='card_head.0.weight' in spec['weights'])
            model.load_state_dict(spec['weights'])
            return Phase2Policy(model, seed=seed, temperature=temperature, profile_prior=spec.get('profile_prior', 20))
        return NetworkPolicy(spec['weights'], seed=seed, encoder=spec['encoder'])
    a, b = policy(challenger, action_seed), policy(opponent, action_seed + 1)
    first, second = play_mirror_pair(a, b, seed=deal_seed, target=target, deal_partition=0)
    return ((first.winner == 0) + (second.winner == 1)) / 2

def _check_chunk(job):
    a, b, deals, temperature = job
    return [_check_pair((a, b, deal, target, action_seed, temperature))
            for deal, target, action_seed in deals]

def _panel_jobs(panel, candidates, schedule_seed, pairs, target, temperature, workers):
    jobs, keys = [], []
    for member in panel['members']:
        opponent = {'kind': member['kind'], 'encoder': member['encoder']}
        if member['kind'] == 'phase2':
            opponent['profile_prior'] = member.get('profile_prior', 20)
        opponent['weights'] = (torch.load(member['resolved_path'], map_location='cpu', weights_only=True)
                               if member['kind'] == 'phase2' else np.load(member['resolved_path']))
        for label, candidate in candidates:
            deals = [(deal, t, schedule_seed * 100000 + i * 2)
                     for i, (deal, t) in enumerate(validation_schedule(schedule_seed, target, pairs))]
            for chunk in np.array_split(np.arange(pairs), min(workers, pairs)):
                jobs.append((candidate, opponent, [deals[int(i)] for i in chunk], temperature))
                keys.append((member['name'], label))
    return jobs, keys

def _run_check_jobs(jobs, keys, pool):
    batches = pool.map(_check_chunk, jobs, chunksize=1) if pool else list(map(_check_chunk, jobs))
    results = {}
    for key, batch in zip(keys, batches):
        results.setdefault(key, []).extend(batch)
    return results

def _champion_overlap(champion, panel, schedule_seed, pairs, target, temperature, pool, profile_prior=20):
    incumbent = {'kind': 'phase2', 'weights': champion, 'profile_prior': profile_prior}
    jobs, keys = _panel_jobs(panel, [('champion', incumbent)], schedule_seed, pairs, target, temperature, pool._processes if pool else 1)
    scores = _run_check_jobs(jobs, keys, pool)
    return {member['name']: scores[(member['name'], 'champion')] for member in panel['members']}

def _selection_check(model, champion, panel, schedule_seed, pairs, direct_pairs, target, temperature, pool, min_gain, min_direct, profile_prior=20):
    current = {'kind': 'phase2', 'weights': model.state_dict(), 'profile_prior': profile_prior}
    incumbent = {'kind': 'phase2', 'weights': champion, 'profile_prior': profile_prior}
    workers = pool._processes if pool else 1
    jobs, keys = _panel_jobs(panel, [('challenger', current), ('champion', incumbent)], schedule_seed, pairs, target, temperature, workers)
    deals = [(deal, t, schedule_seed * 100000 + i * 2 + 1) for i, (deal, t) in enumerate(validation_schedule(schedule_seed, target, direct_pairs))]
    for chunk in np.array_split(np.arange(direct_pairs), min(workers, direct_pairs)):
        jobs.append((current, incumbent, [deals[int(i)] for i in chunk], temperature))
        keys.append((None, 'direct'))
    scores = _run_check_jobs(jobs, keys, pool)
    members = {member['name']: {label: scores[(member['name'], label)] for label in ('challenger', 'champion')} for member in panel['members']}
    challenger_scores = {name: row['challenger'] for name, row in members.items()}
    champion_scores = {name: row['champion'] for name, row in members.items()}
    direct = scores[(None, 'direct')]
    result = promotion_decision(challenger_scores, champion_scores, direct, min_gain, min_direct)
    return dict(result, members=members, direct=direct, panel_version=panel['version'], measuring_only=bool(panel.get('measuring_only')), schedule_seed=schedule_seed)

def _portable(value):
    value = copy.deepcopy(value)
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items() if key != 'resolved_path'}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    return value

def _resolve_panel(panel, base):
    for member in panel['members']:
        path = (Path(base) / member['path']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != member['sha256']:
            raise ValueError('panel hash mismatch: ' + member['name'])
        member['resolved_path'] = str(path)
    return panel

@dataclass
class RoundResult:
    model: Phase2Network
    optimizer: object
    samples: int
    updates: int
    last_samples: list

def run_rounds(*, seed=0, rounds=1, matches_per_round=4, workers=1, target='mixed',
               lr=3e-4, minibatch=512, generation_rounds=None, log_path=None,
               random_start_share=.3, checkpoint_dir=None, config=None, resume_state=None,
               stop_after_round=None, panel=None, champions_dir=None, panel_pairs=100, head_to_head_pairs=200,
               check_seed_base=10000, temperature=.1, profile_prior=20, min_gain=.01, min_direct=.475,
               patience=40, time_cap_seconds=28800, clock=time.monotonic, saturation_checks=3, saturation_pairs=300, saturation_threshold=.7, pool_recent=8, pool_diverse=4, block_size=4, league_shares=None, phase1_manifest=None, style_target=.1, style_ramp_matches=20000, neutral_rate=.05, progress_path=None, console=None, auxiliary=False, auxiliary_weight=.1, card_prediction_situations=400, extension=None):
    if not math.isfinite(profile_prior) or profile_prior <= 0:
        raise ValueError('profile_prior must be positive and finite')
    if not isinstance(card_prediction_situations, int) or card_prediction_situations < 1:
        raise ValueError('Invalid card prediction situation count')
    if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError('Invalid auxiliary loss weight')
    if (min(rounds, matches_per_round, workers, minibatch) < 1 or min(pool_recent, pool_diverse) < 0 or seed < 0
            or (target != 'mixed' and (not isinstance(target, int) or target < 1))
            or not 0 <= random_start_share <= 1 or not math.isfinite(lr) or lr <= 0):
        raise ValueError('Invalid training configuration')
    if ((generation_rounds is None) != (log_path is None)
            or (generation_rounds is not None and generation_rounds < 1)
            or (checkpoint_dir is not None and generation_rounds is None)):
        raise ValueError('generation_rounds and log_path must be supplied together for checkpoints')
    if block_size < 1 or matches_per_round % block_size:
        raise ValueError('matches_per_round must be divisible by block_size')
    if phase1_manifest is None:
        phase1_manifest = Path(__file__).resolve().parent.parent / 'experiments/phase2-selection-panel/league-phase1.json'
    phase1 = load_phase1_manifest(phase1_manifest)
    for member in phase1:
        member['resolved_path'] = str((Path(phase1_manifest).resolve().parent / member['path']).resolve())
    shares = validate_shares(league_shares)
    _configure_threads()
    torch.use_deterministic_algorithms(True)
    model = Phase2Network(seed=seed, auxiliary=auxiliary)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    total = updates = 0
    last = []
    generation_samples = generation_updates = 0
    league_mix = {}
    phase1_counts = {}
    neutral_blocks = 0
    losses = []
    started = clock()
    run_tick = started
    elapsed = 0.0
    stale = 0
    history = []
    snapshots = []
    panel_history = []
    league_pool = []
    first_round = 0
    champion = None
    champion_generation = None
    if resume_state is not None:
        elapsed = resume_state.get('elapsed', 0.0)
        stale = resume_state.get('stale', 0)
        history = resume_state.get('saturation_history', [])
        snapshots = resume_state.get('panel_snapshots', [])
        panel_history = resume_state.get('panel_history', [])
        league_pool = resume_state.get('league_pool', [])
        panel = resume_state.get('panel', panel)
        if panel is not None:
            _resolve_panel(panel, checkpoint_dir)
        for snapshot in snapshots:
            snapshot['resolved_path'] = str((Path(checkpoint_dir) / snapshot['path']).resolve())
        champion = resume_state.get('champion_weights')
        champion_generation = resume_state.get('champion_generation')
        model.load_state_dict(resume_state['model'])
        optimizer.load_state_dict(resume_state['optimizer'])
        first_round = resume_state['round']
        total, updates = resume_state['samples'], resume_state['updates']
        generation_samples, generation_updates = resume_state['generation_samples'], resume_state['generation_updates']
        league_mix = resume_state.get('league_mix', {})
        phase1_counts = resume_state.get('phase1_counts', {})
        neutral_blocks = resume_state.get('neutral_blocks', 0)
        losses = resume_state['losses']
        random.setstate(resume_state['python_rng'])
        np.random.set_state(resume_state['numpy_rng'])
        torch.set_rng_state(resume_state['torch_rng'])
        rng.bit_generator.state = resume_state['shuffle_rng']
    if resume_state is not None and resume_state.get('phase_ended'):
        return RoundResult(model, optimizer, total, updates, last)
    promotions = 0
    if log_path is not None and Path(log_path).exists():
        promotions = sum(bool(json.loads(x).get('promoted')) for x in Path(log_path).read_text(encoding='utf-8').splitlines() if x)
    reporter = ProgressReporter(progress_path, console, time_cap_seconds, temperature, promotions, profile_prior=profile_prior, extension=extension)
    last_round = min(rounds, stop_after_round or rounds)
    context = mp.get_context('spawn')
    pool_context = context.Pool(workers, initializer=_configure_threads) if workers > 1 else nullcontext(None)
    with pool_context as pool:
        if resume_state is not None and resume_state.get('_extension_reopen') and first_round % generation_rounds == 0:
            generation = first_round // generation_rounds
            lines = Path(log_path).read_text(encoding='utf-8').splitlines()
            entry = json.loads(lines[-1])
            if entry['generation'] != generation:
                raise ValueError('Missing stopped generation log for extension')
            entry['phase_ended'] = False
            entry['stop_reason'] = None
            pending_check = not any(item.get('name') == f'generation-{generation}' for item in snapshots)
            if panel is not None and champion_generation != generation and pending_check:
                previous = Path(checkpoint_dir) / f'checkpoint_{first_round - generation_rounds:06d}.pt'
                if not previous.exists():
                    raise ValueError('Previous generation checkpoint required for pending boundary work')
                previous_state = torch.load(previous, map_location='cpu', weights_only=False)
                league_pool = previous_state.get('league_pool', [])
                snapshot = Path(checkpoint_dir) / f'generation_{generation:06d}.pt'
                entry['check'] = None
                entry['promoted'] = False
                entry.pop('panel_boundary', None)
                boundary = None
                names = [m['name'] for m in panel['members']]
                if not panel.get('measuring_only') and saturated(history, champion_generation, names, checks=saturation_checks, min_pairs=saturation_pairs, threshold=saturation_threshold):
                    pooled = {name: statistics.mean([x for row in history[-saturation_checks:] for x in row['scores'][name]]) for name in names}
                    champion_hash = hashlib.sha256((Path(checkpoint_dir) / f'generation_{champion_generation:06d}.pt').read_bytes()).hexdigest()
                    compatible = [x for x in snapshots if set(x['scores']) == set(names)]
                    new_panel = build_panel(panel, pooled, compatible, champion_hash)
                    if new_panel is not None:
                        old_panel = panel
                        panel = new_panel
                        for member in panel['members']:
                            member['path'] = os.path.relpath(member['resolved_path'], checkpoint_dir)
                        panel_history.append(old_panel)
                        path = Path(checkpoint_dir) / f"panel-v{panel['version']}.json"
                        path.write_text(json.dumps(_portable(panel), indent=2) + '\n', encoding='utf-8')
                        seed_for_overlap = check_seed(check_seed_base, generation - 2)
                        old_overlap = _champion_overlap(champion, old_panel, seed_for_overlap, panel_pairs, None if target == 'mixed' else target, temperature, pool, profile_prior)
                        boundary = {'from': old_panel['version'], 'to': panel['version'], 'old_champion': old_overlap}
                        history = []
                result = _selection_check(model, champion, panel, check_seed(check_seed_base, generation - 2),
                                          panel_pairs, head_to_head_pairs, None if target == 'mixed' else target,
                                          temperature, pool, min_gain, min_direct, profile_prior)
                if boundary is not None:
                    boundary['new_champion'] = {k: v['champion'] for k, v in result['members'].items()}
                    entry['panel_boundary'] = boundary
                entry['check'] = result
                history.append({'champion': champion_generation, 'scores': {k: v['champion'] for k, v in result['members'].items()}})
                snapshot_record = {'name': f'generation-{generation}', 'path': snapshot.name, 'resolved_path': str(snapshot), 'kind': 'phase2', 'encoder': 'v1', 'sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(), 'gain': result['gain'], 'scores': {k: statistics.mean(v['challenger']) for k, v in result['members'].items()}}
                if profile_prior != 20:
                    snapshot_record['profile_prior'] = profile_prior
                snapshots.append(snapshot_record)
                stale = 0 if result['promoted'] else stale + 1
                entry['promoted'] = result['promoted']
                changed = result['promoted']
                if changed:
                    champion = {k: v.clone() for k, v in model.state_dict().items()}
                    champion_generation = generation
                    history = []
                league_pool.append({'generation': generation, 'path': snapshot.name,
                                    'sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                                    'scores': {k: statistics.mean(v['challenger']) for k, v in result['members'].items()}})
                league_pool = retain_snapshots(league_pool, pool_recent, pool_diverse, panel_exclusions(panel, panel_history))
                entry['champion'] = champion_generation
                if changed and champions_dir is not None:
                    _export_champion(champions_dir, champion, champion_generation, temperature, profile_prior)
                now = clock()
                elapsed += now - run_tick
                run_tick = now
                if elapsed >= time_cap_seconds or stale >= patience:
                    entry['phase_ended'] = True
                    entry['stop_reason'] = 'time_cap' if elapsed >= time_cap_seconds else 'patience'
            lines[-1] = json.dumps(entry)
            reporter.promotions = sum(bool(json.loads(line).get('promoted')) for line in lines)
            # Commit the log first; if checkpoint replacement is interrupted, replay
            # from the old state replaces this same record and recounts promotions.
            log_file = Path(log_path)
            log_temporary = log_file.with_name(log_file.name + '.boundary.tmp')
            with log_temporary.open('w', encoding='utf-8') as stream:
                stream.write(''.join(line + '\n' for line in lines))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(log_temporary, log_file)
            if checkpoint_dir is not None:
                saved = dict(resume_state)
                saved.pop('_extension_reopen', None)
                saved.update(phase_ended=entry['phase_ended'], stop_reason=entry['stop_reason'], elapsed=elapsed,
                             promotions=reporter.promotions,
                             stale=stale, saturation_history=history, panel_snapshots=_portable(snapshots),
                             league_pool=_portable(league_pool), panel_history=_portable(panel_history), panel=_portable(panel),
                             champion_weights=champion, champion_generation=champion_generation,
                             python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                             torch_rng=torch.get_rng_state(), shuffle_rng=rng.bit_generator.state)
                path = Path(checkpoint_dir) / f'checkpoint_{first_round:06d}.pt'
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    torch.save(saved, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                reporter.round_done(elapsed=elapsed, generation=generation, round=first_round,
                                    matches=resume_state['matches'], samples=total, updates=updates,
                                    recent_loss=None, league_mix={}, champion_generation=champion_generation,
                                    panel_version=panel['version'] if panel else None,
                                    ended=entry['phase_ended'], stop_reason=entry['stop_reason'])
                if entry['phase_ended']:
                    return RoundResult(model, optimizer, total, updates, last)
        for round_id in range(first_round, min(rounds, stop_after_round or rounds)):
            weights = model.state_dict()
            exclusions = panel_exclusions(panel, panel_history)
            available = [x for x in league_pool if x['sha256'] not in exclusions]
            opponents = {x['sha256']: dict(x, resolved_path=str((Path(checkpoint_dir) / x['path']).resolve())) for x in available}
            opponents.update({x['sha256']: x for x in phase1 if x['sha256'] not in exclusions})
            blocks = []
            first_id = round_id * matches_per_round
            for match_id in range(first_id, first_id + matches_per_round, block_size):
                spec = draw_opponent(seed, match_id // block_size, shares=shares, pool=available,
                                     phase1=phase1, exclusions=exclusions, matches_so_far=match_id,
                                     target_offset=style_target, ramp_matches=style_ramp_matches,
                                     neutral_rate=neutral_rate)
                ids = list(range(match_id, match_id + block_size))
                blocks.append((ids, spec))
                category = spec['style'] or spec['category']
                league_mix[category] = league_mix.get(category, 0) + len(ids)
                if spec['category'] == 'phase1':
                    name = opponents[spec['identity']]['name']
                    phase1_counts[name] = phase1_counts.get(name, 0) + len(ids)
                neutral_blocks += int(spec['neutral_profile'])
            chunks = [list(chunk) for chunk in np.array_split(np.arange(len(blocks)), min(workers, len(blocks)))]
            jobs = [(weights, seed, [blocks[int(i)] for i in chunk], target, random_start_share, opponents, temperature, profile_prior) for chunk in chunks]
            batches = pool.map(_league_chunk, jobs, chunksize=1) if pool else list(map(_league_chunk, jobs))
            results = [result for batch in batches for result in batch]
            last = [sample for _, samples in sorted(results) for sample in samples]
            total += len(last)
            generation_samples += len(last)
            for indices in (np.array_split(rng.permutation(len(last)), range(minibatch, len(last), minibatch)) if last else ()):
                x = torch.from_numpy(np.stack([last[i][0] for i in indices]))
                actions = torch.tensor([last[i][1] for i in indices])
                targets = torch.tensor([last[i][2] for i in indices])
                loss = torch.nn.functional.mse_loss(model(x).gather(1, actions[:, None]).squeeze(1), targets)
                if auxiliary:
                    masks = torch.from_numpy(compatible_mask([last[i][4] for i in indices]))
                    hand_ids = torch.from_numpy(_IDS)
                    scores = combination_scores(model.card_logits(x), hand_ids)
                    labels = torch.tensor([last[i][5] for i in indices])
                    if not masks[torch.arange(len(indices)), labels].all():
                        raise ValueError('True hand excluded by exact evidence')
                    scores = scores.masked_fill(~masks, float('-inf'))
                    loss = loss + auxiliary_weight * torch.nn.functional.cross_entropy(scores, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                updates += 1
                generation_updates += 1
            recent_loss = statistics.mean(losses[-20:]) if losses else None
            mix_seen = dict(league_mix)
            round_stop = None
            now = clock()
            elapsed += now - run_tick
            run_tick = now
            if generation_rounds and (round_id + 1) % generation_rounds == 0:
                entry = dict(phase=2, generation=(round_id + 1) // generation_rounds,
                             matches=generation_rounds * matches_per_round,
                             learner_samples=generation_samples, updates=generation_updates,
                             mean_loss=sum(losses) / len(losses) if losses else None, elapsed_seconds=clock() - started,
                             league_mix=dict(league_mix), phase1_opponents=dict(phase1_counts), neutral_blocks=neutral_blocks, style_offset_round_end=style_target * min(1, ((round_id + 1) * matches_per_round) / style_ramp_matches) if style_ramp_matches else style_target,
                             check=None, promoted=False,
                             phase_ended=False, stop_reason=None, measuring_only=bool(panel and panel.get('measuring_only')), fitness=None, behavior=None,
                             diversity=None, champion=None, example_hand=None)
                if checkpoint_dir is not None:
                    generation = entry['generation']
                    snapshot = Path(checkpoint_dir) / f'generation_{generation:06d}.pt'
                    model.save(snapshot)
                    if champion is None:
                        champion = {k: v.clone() for k, v in model.state_dict().items()}
                        champion_generation = generation
                        changed = True
                    elif panel is not None and elapsed < time_cap_seconds:
                        boundary = None
                        names = [m['name'] for m in panel['members']]
                        if not panel.get('measuring_only') and saturated(history, champion_generation, names, checks=saturation_checks, min_pairs=saturation_pairs, threshold=saturation_threshold):
                            pooled = {name: statistics.mean([x for row in history[-saturation_checks:] for x in row['scores'][name]]) for name in names}
                            champion_hash = hashlib.sha256((Path(checkpoint_dir) / f'generation_{champion_generation:06d}.pt').read_bytes()).hexdigest()
                            compatible = [x for x in snapshots if set(x['scores']) == set(names)]
                            new_panel = build_panel(panel, pooled, compatible, champion_hash)
                            if new_panel is not None:
                                old_panel = panel
                                panel = new_panel
                                for member in panel['members']:
                                    member['path'] = os.path.relpath(member['resolved_path'], checkpoint_dir)
                                panel_history.append(old_panel)
                                path = Path(checkpoint_dir) / f"panel-v{panel['version']}.json"
                                path.write_text(json.dumps(_portable(panel), indent=2) + '\n', encoding='utf-8')
                                seed_for_overlap = check_seed(check_seed_base, generation - 2)
                                old_overlap = _champion_overlap(champion, old_panel, seed_for_overlap, panel_pairs, None if target == 'mixed' else target, temperature, pool, profile_prior)
                                boundary = {'from': old_panel['version'], 'to': panel['version'], 'old_champion': old_overlap}
                                history = []
                        result = _selection_check(model, champion, panel, check_seed(check_seed_base, generation - 2),
                                                  panel_pairs, head_to_head_pairs, None if target == 'mixed' else target,
                                                  temperature, pool, min_gain, min_direct, profile_prior)
                        if boundary is not None:
                            boundary['new_champion'] = {k: v['champion'] for k, v in result['members'].items()}
                            entry['panel_boundary'] = boundary
                        entry['check'] = result
                        history.append({'champion': champion_generation, 'scores': {k: v['champion'] for k, v in result['members'].items()}})
                        snapshot_record = {'name': f'generation-{generation}', 'path': snapshot.name, 'resolved_path': str(snapshot), 'kind': 'phase2', 'encoder': 'v1', 'sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(), 'gain': result['gain'], 'scores': {k: statistics.mean(v['challenger']) for k, v in result['members'].items()}}
                        if profile_prior != 20:
                            snapshot_record['profile_prior'] = profile_prior
                        snapshots.append(snapshot_record)
                        stale = 0 if result['promoted'] else stale + 1
                        entry['promoted'] = result['promoted']
                        changed = result['promoted']
                        if changed:
                            champion = {k: v.clone() for k, v in model.state_dict().items()}
                            champion_generation = generation
                            history = []
                    else:
                        changed = False
                    league_pool.append({'generation': generation, 'path': snapshot.name,
                                        'sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                                        'scores': {k: statistics.mean(v['challenger']) for k, v in result['members'].items()} if entry['check'] else {}})
                    league_pool = retain_snapshots(league_pool, pool_recent, pool_diverse, panel_exclusions(panel, panel_history))
                    entry['champion'] = champion_generation
                    now = clock()
                    elapsed += now - run_tick
                    run_tick = now
                    entry['stop_reason'] = 'time_cap' if elapsed >= time_cap_seconds else ('patience' if stale >= patience else None)
                    entry['phase_ended'] = entry['stop_reason'] is not None
                    round_stop = entry['stop_reason']
                    if changed and panel is not None and champions_dir is not None:
                        _export_champion(champions_dir, champion, champion_generation, temperature, profile_prior)
                if auxiliary:
                    from phase2_card_metrics import validation_predictions
                    entry['card_prediction'] = validation_predictions(model, schedule_seed=check_seed_base, target=target, temperature=temperature, situation_count=card_prediction_situations, profile_prior=profile_prior)
                    now = clock()
                    elapsed += now - run_tick
                    run_tick = now
                    entry['elapsed_seconds'] = now - started
                    if elapsed >= time_cap_seconds:
                        entry['stop_reason'] = round_stop = 'time_cap'
                        entry['phase_ended'] = True
                with Path(log_path).open('a', encoding='utf-8') as log:
                    log.write(json.dumps(entry) + '\n')
                reporter.generation_done(entry, elapsed)
                generation_samples = generation_updates = 0
                league_mix = {}
                phase1_counts = {}
                neutral_blocks = 0
                losses = []
                started = clock()
            if generation_rounds and (round_id + 1) % generation_rounds and elapsed >= time_cap_seconds:
                entry = dict(phase=2, generation=(round_id + 1) // generation_rounds + 1,
                             matches=(round_id + 1) % generation_rounds * matches_per_round,
                             learner_samples=generation_samples, updates=generation_updates,
                             mean_loss=sum(losses) / len(losses) if losses else None, elapsed_seconds=clock() - started,
                             league_mix=dict(league_mix), phase1_opponents=dict(phase1_counts), neutral_blocks=neutral_blocks, style_offset_round_end=style_target * min(1, ((round_id + 1) * matches_per_round) / style_ramp_matches) if style_ramp_matches else style_target,
                             check=None, promoted=False, phase_ended=True,
                             stop_reason='time_cap', measuring_only=bool(panel and panel.get('measuring_only')), fitness=None, behavior=None, diversity=None,
                             champion=champion_generation, example_hand=None)
                if auxiliary:
                    from phase2_card_metrics import validation_predictions
                    entry['card_prediction'] = validation_predictions(model, schedule_seed=check_seed_base, target=target, temperature=temperature, situation_count=card_prediction_situations, profile_prior=profile_prior)
                    now = clock()
                    elapsed += now - run_tick
                    run_tick = now
                    entry['elapsed_seconds'] = now - started
                    if elapsed >= time_cap_seconds:
                        entry['stop_reason'] = round_stop = 'time_cap'
                        entry['phase_ended'] = True
                with Path(log_path).open('a', encoding='utf-8') as log:
                    log.write(json.dumps(entry) + '\n')
                round_stop = 'time_cap'
                reporter.generation_done(entry, elapsed)
            if checkpoint_dir is not None:
                state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             round=round_id + 1, generation=(round_id + 1) // generation_rounds,
                             matches=(round_id + 1) * matches_per_round, samples=total, updates=updates,
                             generation_samples=generation_samples, generation_updates=generation_updates,
                             losses=losses, league_mix=league_mix, phase1_counts=phase1_counts, neutral_blocks=neutral_blocks, config=config, phase_ended=entry['phase_ended'] if generation_rounds and ((round_id + 1) % generation_rounds == 0 or elapsed >= time_cap_seconds) else False, stop_reason=entry.get('stop_reason') if generation_rounds and ((round_id + 1) % generation_rounds == 0 or elapsed >= time_cap_seconds) else None, elapsed=elapsed, stale=stale, saturation_history=history, panel_snapshots=_portable(snapshots), league_pool=_portable(league_pool), panel_history=_portable(panel_history), panel=_portable(panel), champion_weights=champion, champion_generation=champion_generation, python_rng=random.getstate(),
                             numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                             shuffle_rng=rng.bit_generator.state)
                if extension is not None:
                    state['extension'] = extension
                    state['promotions'] = reporter.promotions
                path = Path(checkpoint_dir) / f'checkpoint_{round_id + 1:06d}.pt'
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    torch.save(state, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                for old in Path(checkpoint_dir).glob('checkpoint_*.pt'):
                    suffix = old.stem.removeprefix('checkpoint_')
                    if not suffix.isdigit():
                        continue
                    old_round = int(suffix)
                    if old_round != round_id + 1 and (old_round > round_id + 1 or old_round % generation_rounds):
                        old.unlink()
            finished = round_stop is None and round_id + 1 == last_round and stop_after_round is None
            reporter.round_done(elapsed=elapsed, generation=(round_id + 1) // generation_rounds if generation_rounds else 0,
                                round=round_id + 1, matches=(round_id + 1) * matches_per_round, samples=total, updates=updates,
                                recent_loss=recent_loss, league_mix=mix_seen, champion_generation=champion_generation,
                                panel_version=panel['version'] if panel else None,
                                ended=round_stop is not None or finished,
                                stop_reason=round_stop or ('completed' if finished else None))
            if generation_rounds and ((round_id + 1) % generation_rounds == 0 or elapsed >= time_cap_seconds) and entry['phase_ended']:
                break
    return RoundResult(model, optimizer, total, updates, last)

def train(directory, *, generations=1, matches_per_generation=120, matches_per_round=12,
          workers=1, seed=0, target='mixed', lr=3e-4, minibatch=512,
          random_start_share=.3, resume=None, stop_after_round=None, panel_path=Path(__file__).resolve().parent.parent / 'experiments/phase2-selection-panel/panel-v1.json',
          champions_dir="champions/phase2", panel_pairs=100, head_to_head_pairs=200,
          check_seed_base=10000, temperature=.1, profile_prior=20, min_gain=.01, min_direct=.475,
          patience=40, time_cap_seconds=28800, clock=time.monotonic, saturation_checks=3, saturation_pairs=300, saturation_threshold=.7, pool_recent=8, pool_diverse=4, block_size=4, league_shares=None, phase1_manifest=None, style_target=.1, style_ramp_matches=20000, neutral_rate=.05, progress=True, console=None, auxiliary=False, auxiliary_weight=.1, card_prediction_situations=400, extend_time_cap=None):
    """Round match counts up to complete blocks and generations to complete rounds."""
    if extend_time_cap is not None and not resume:
        raise ValueError("--extend-time-cap requires --resume")
    if extend_time_cap is not None and not math.isfinite(extend_time_cap):
        raise ValueError("Extension cap must be finite")
    positive_counts = (generations, matches_per_generation, matches_per_round, workers,
                       minibatch, patience, saturation_checks, saturation_pairs, block_size,
                       card_prediction_situations)
    nonnegative_counts = (seed, check_seed_base, pool_recent, pool_diverse, style_ramp_matches)
    if (any(not isinstance(value, numbers.Integral) or value < 1 for value in positive_counts)
            or any(not isinstance(value, numbers.Integral) or value < 0 for value in nonnegative_counts)
            or any(not isinstance(value, numbers.Integral) or value < 2 for value in (panel_pairs, head_to_head_pairs))
            or (stop_after_round is not None and (not isinstance(stop_after_round, numbers.Integral) or stop_after_round < 1))
            or (target != 'mixed' and (not isinstance(target, numbers.Integral) or target < 1))):
        raise ValueError('Invalid training counts')
    finite_settings = (time_cap_seconds, min_gain, min_direct, saturation_threshold,
                       style_target, neutral_rate, lr, random_start_share, temperature,
                       profile_prior, auxiliary_weight)
    if (not all(math.isfinite(value) for value in finite_settings)
            or time_cap_seconds <= 0 or lr <= 0 or temperature <= 0 or profile_prior <= 0
            or not 0 <= min_gain <= 1 or not 0 <= min_direct <= 1
            or not 0 <= saturation_threshold <= 1 or style_target < 0
            or not 0 <= neutral_rate <= 1 or not 0 <= random_start_share <= 1
            or auxiliary_weight < 0):
        raise ValueError('Invalid finite training configuration')
    if not math.isfinite(profile_prior) or profile_prior <= 0:
        raise ValueError('profile_prior must be positive and finite')
    if min(generations, matches_per_generation, matches_per_round, block_size) < 1:
        raise ValueError('Generation and match counts must be positive')
    if min(panel_pairs, head_to_head_pairs) < 2 or not math.isfinite(temperature) or temperature <= 0 or min(patience, time_cap_seconds, saturation_checks, saturation_pairs) <= 0:
        raise ValueError('Invalid selection configuration')
    if panel_path and (check_seed_base < 0 or any(check_seed_base <= seed <= check_seed_base + max(0, generations - 2) for seed in RESERVED_SEEDS)):
        raise ValueError('reserved schedule seed in requested checks')
    panel = load_panel(panel_path) if panel_path else None
    if panel is not None:
        for member in panel['members']:
            member['resolved_path'] = str((Path(panel_path).resolve().parent / member['path']).resolve())
            member['path'] = os.path.relpath(member['resolved_path'], Path(directory).resolve())
    matches_per_round = ((matches_per_round + block_size - 1) // block_size) * block_size
    phase1_manifest = str(Path(phase1_manifest or (Path(__file__).resolve().parent.parent / 'experiments/phase2-selection-panel/league-phase1.json')).resolve())
    panel_config = None
    if panel:
        members = []
        for member in panel['members']:
            entry = {k: member[k] for k in ('name', 'kind', 'encoder', 'sha256')}
            if member['kind'] == 'phase2':
                prior = member.get('profile_prior', 20)
                if not math.isfinite(prior) or prior <= 0:
                    raise ValueError('panel profile_prior must be positive and finite')
                if prior != 20:
                    entry['profile_prior'] = prior
            members.append(entry)
        panel_config = {'version': panel['version'], 'measuring_only': bool(panel.get('measuring_only')), 'members': members}
    config = dict(engine_version=ENGINE_VERSION, encoder_version=ENCODER_VERSION, block_size=block_size, league_shares=validate_shares(league_shares), phase1_manifest=[{k: m[k] for k in ('name', 'encoder', 'sha256')} for m in load_phase1_manifest(phase1_manifest)], style_target=style_target, style_ramp_matches=style_ramp_matches, neutral_rate=neutral_rate, pool_recent=pool_recent, pool_diverse=pool_diverse, patience=patience, time_cap_seconds=time_cap_seconds, saturation_checks=saturation_checks, saturation_pairs=saturation_pairs, saturation_threshold=saturation_threshold, panel_path=panel_config,
                  panel_pairs=panel_pairs, head_to_head_pairs=head_to_head_pairs,
                  check_seed_base=check_seed_base, temperature=temperature, min_gain=min_gain, min_direct=min_direct,
                  matches_per_generation=matches_per_generation,
                  matches_per_round=matches_per_round, seed=seed, target=target,
                  lr=lr, minibatch=minibatch, random_start_share=random_start_share)
    config.update(auxiliary=auxiliary, auxiliary_weight=auxiliary_weight, card_prediction_situations=card_prediction_situations)
    if profile_prior != 20:
        config['profile_prior'] = profile_prior
    generation_rounds = (matches_per_generation + matches_per_round - 1) // matches_per_round
    output = Path(directory).resolve()
    if resume and Path(resume).resolve().parent != output:
        raise ValueError('Resume checkpoint must be inside the output directory')
    state = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    old_config = dict(state['config']) if state is not None else None
    if old_config is not None:
        old_config.pop('workers', None)
        old_config.pop('generations', None)
        old_config.setdefault('auxiliary', False)
        old_config.setdefault('auxiliary_weight', .1)
        old_config.setdefault('card_prediction_situations', 400)
        if old_config.get('profile_prior', 20) == 20:
            old_config.pop('profile_prior', None)
    if state is not None and old_config != config:
        raise ValueError('Checkpoint configuration does not match')
    extension = state.get('extension') if state is not None else None
    if extend_time_cap is not None:
        if extension is not None:
            if extend_time_cap != extension['new_cap']:
                raise ValueError('Only the recorded time-cap extension is allowed')
        else:
            if extend_time_cap <= time_cap_seconds:
                raise ValueError('Extension cap must be strictly greater than checkpoint cap')
            if not state.get('phase_ended') or state.get('stop_reason') != 'time_cap' or state.get('stale', 0) >= patience or state['round'] >= generations * generation_rounds:
                raise ValueError('Only a run ended solely by the time cap can be extended')
            record_path = output / 'time_cap_extension.json'
            if record_path.exists():
                extension = json.loads(record_path.read_text(encoding='utf-8'))
                if extension['checkpoint_sha256'] != hashlib.sha256(Path(resume).read_bytes()).hexdigest() or extension['new_cap'] != extend_time_cap:
                    raise ValueError('A different time-cap extension is already recorded')
            else:
                import subprocess
                revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True).strip()
                extension = dict(version=1, old_cap=time_cap_seconds, new_cap=extend_time_cap,
                                 checkpoint_sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest(),
                                 code_revision=revision, timestamp=datetime.now(timezone.utc).isoformat())
                from phase2_progress import write_json_atomic
                if not write_json_atomic(record_path, extension):
                    raise OSError('Could not durably record time-cap extension')
        state['extension'] = extension
        if state.get('phase_ended') and state.get('stop_reason') == 'time_cap' and state['round'] * matches_per_round < generations * generation_rounds * matches_per_round and state.get('elapsed', 0) < extend_time_cap:
            state['_extension_reopen'] = True
            state['phase_ended'] = False
        time_cap_seconds = extend_time_cap
    elif extension is not None:
        time_cap_seconds = extension['new_cap']

    if state is not None and state.get('panel') is not None and champions_dir is not None and state.get('champion_weights') is not None:
        _reconcile_champion(champions_dir, state['champion_weights'], state['champion_generation'], temperature, profile_prior)
    if state is not None and state.get('phase_ended'):
        return run_rounds(seed=seed, rounds=generations * generation_rounds, matches_per_round=matches_per_round, workers=workers, target=target, lr=lr, minibatch=minibatch, generation_rounds=generation_rounds, log_path=output / 'generations.jsonl', checkpoint_dir=output, resume_state=state, block_size=block_size, league_shares=league_shares, phase1_manifest=phase1_manifest, style_target=style_target, style_ramp_matches=style_ramp_matches, neutral_rate=neutral_rate, progress_path=output / PROGRESS_FILE if progress else None, console=console, auxiliary=auxiliary, auxiliary_weight=auxiliary_weight, card_prediction_situations=card_prediction_situations, profile_prior=profile_prior)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / 'generations.jsonl'
    if state is None:
        log_path.write_text('', encoding='utf-8')
    else:
        lines = log_path.read_text(encoding='utf-8').splitlines() if log_path.exists() else []
        kept = [line for line in lines if json.loads(line)['generation'] <= state['generation']]
        log_path.write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
    return run_rounds(seed=seed, rounds=generations * generation_rounds,
                      matches_per_round=matches_per_round, workers=workers, target=target,
                      lr=lr, minibatch=minibatch, generation_rounds=generation_rounds, log_path=log_path,
                      random_start_share=random_start_share, checkpoint_dir=output, config=config,
                      resume_state=state, stop_after_round=stop_after_round, panel=panel, champions_dir=champions_dir,
                      panel_pairs=panel_pairs, head_to_head_pairs=head_to_head_pairs, check_seed_base=check_seed_base,
                      temperature=temperature, profile_prior=profile_prior, min_gain=min_gain, min_direct=min_direct, patience=patience, time_cap_seconds=time_cap_seconds, clock=clock, saturation_checks=saturation_checks, saturation_pairs=saturation_pairs, saturation_threshold=saturation_threshold, pool_recent=pool_recent, pool_diverse=pool_diverse, block_size=block_size, league_shares=league_shares, phase1_manifest=phase1_manifest, style_target=style_target, style_ramp_matches=style_ramp_matches, neutral_rate=neutral_rate, progress_path=output / PROGRESS_FILE if progress else None, console=console, auxiliary=auxiliary, auxiliary_weight=auxiliary_weight, card_prediction_situations=card_prediction_situations, extension=extension)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='training_output/phase2')
    parser.add_argument('--generations', type=int, default=1)
    parser.add_argument('--matches-per-generation', type=int, default=120)
    parser.add_argument('--matches-per-round', type=int, default=12)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', type=lambda value: 'mixed' if value == 'mixed' else int(value), default='mixed')
    parser.add_argument('--random-start-share', type=float, default=.3)
    parser.add_argument('--panel-path', default=str(Path(__file__).resolve().parent.parent / 'experiments/phase2-selection-panel/panel-v1.json'))
    parser.add_argument('--no-selection', action='store_true')
    parser.add_argument('--champions-dir', default='champions/phase2')
    parser.add_argument('--panel-pairs', type=int, default=100)
    parser.add_argument('--head-to-head-pairs', type=int, default=200)
    parser.add_argument('--check-seed-base', type=int, default=10000)
    parser.add_argument('--temperature', type=float, default=.1)
    parser.add_argument('--profile-prior', type=float, default=20)
    parser.add_argument('--min-gain', type=float, default=.01)
    parser.add_argument('--min-direct', type=float, default=.475)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--time-cap-seconds', type=float, default=28800)
    parser.add_argument('--saturation-checks', type=int, default=3)
    parser.add_argument('--saturation-pairs', type=int, default=300)
    parser.add_argument('--saturation-threshold', type=float, default=.7)
    parser.add_argument('--league-shares', type=json.loads, help='JSON mapping of league shares')
    parser.add_argument('--block-size', type=int, default=4)
    parser.add_argument('--pool-recent', type=int, default=8)
    parser.add_argument('--pool-diverse', type=int, default=4)
    parser.add_argument('--style-target', type=float, default=.1)
    parser.add_argument('--style-ramp-matches', type=int, default=20000)
    parser.add_argument('--neutral-rate', type=float, default=.05)
    parser.add_argument('--phase1-manifest')
    parser.add_argument('--no-progress', action='store_true', help='skip progress.json and the terminal lines')
    parser.add_argument('--auxiliary', action='store_true')
    parser.add_argument('--auxiliary-weight', type=float, default=.1)
    parser.add_argument('--card-prediction-situations', type=int, default=400)
    parser.add_argument('--resume')
    parser.add_argument('--extend-time-cap', type=float)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--minibatch', type=int, default=512)
    args = vars(parser.parse_args())
    if args.pop('no_selection'):
        args['panel_path'] = None
    args['directory'] = args.pop('output')
    args['progress'] = not args.pop('no_progress')
    args['console'] = sys.stdout if args['progress'] else None
    train(**args)

if __name__ == '__main__':
    main()
