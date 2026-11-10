"""Phase 2 speed benchmark with a trained network and run 2 settings.

Plays real training rounds (240 matches, run 2 league shares, style offsets fully
ramped) with a trained learner, the same network as a league snapshot, the phase 1
league and both references, then times the learner update and a reduced check.

    py proyecto/bench_phase2_speed.py round --weights <champion.pt> --output out.json
    py proyecto/bench_phase2_speed.py profile --weights <champion.pt> --output out

The weights file is only read.
"""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pickle
import pstats
import shutil
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
RUN2 = ROOT / 'experiments/phase2-run2'
SCRATCH = ROOT / 'training_output/phase2-speed'


def setup(weights_path, directory, encoder_version=1):
    import torch
    import phase2_trainer as trainer
    from phase2_league import load_phase1_manifest, panel_exclusions
    weights = torch.load(weights_path, map_location='cpu', weights_only=True)
    if encoder_version > 1:
        from phase2_torch import widen_state_dict
        weights = widen_state_dict(weights, encoder_version)
    snapshot = Path(directory) / 'generation_000001.pt'
    shutil.copyfile(weights_path, snapshot)
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    pool = [{'generation': 1, 'path': snapshot.name, 'sha256': digest, 'scores': {}}]
    manifest = ROOT / 'experiments/phase2-selection-panel/league-phase1.json'
    phase1 = load_phase1_manifest(manifest)
    for member in phase1:
        member['resolved_path'] = str((manifest.parent / member['path']).resolve())
    shares = json.loads((RUN2 / 'league-shares.json').read_text(encoding='utf-8'))
    panel = trainer.load_panel(RUN2 / 'panel.json')
    exclusions = panel_exclusions(panel)
    opponents = {x['sha256']: dict(x, resolved_path=str(snapshot.resolve())) for x in pool}
    opponents.update({x['sha256']: x for x in phase1 if x['sha256'] not in exclusions})
    return weights, pool, phase1, shares, exclusions, opponents


def round_blocks(seed, round_id, matches, pool, phase1, shares, exclusions):
    from phase2_league import draw_opponent
    blocks = []
    first = round_id * matches
    for match_id in range(first, first + matches, 4):
        # Late-run position: style offsets fully ramped (ADR 0052).
        spec = draw_opponent(seed, match_id // 4, shares=shares, pool=pool, phase1=phase1, exclusions=exclusions,
                             matches_so_far=10 ** 6, target_offset=.1, ramp_matches=20000, neutral_rate=.05)
        blocks.append((list(range(match_id, match_id + 4)), spec))
    return blocks


def play(trainer, pool, weights, seed, blocks, opponents):
    args = dict(weights=weights, seed=seed, blocks=blocks, target='mixed', random_start_share=.3,
                opponents=opponents, temperature=.05, profile_prior=5)
    if hasattr(trainer, '_play_round'):
        return [sample for _, samples in trainer._play_round(pool, **args) for sample in samples]
    import numpy as np
    workers = pool._processes if pool else 1
    chunks = [list(chunk) for chunk in np.array_split(np.arange(len(blocks)), min(workers, len(blocks)))]
    jobs = [(weights, seed, [blocks[int(i)] for i in chunk], 'mixed', .3, opponents, .05, 5) for chunk in chunks]
    batches = pool.map(trainer._league_chunk, jobs, chunksize=1) if pool else list(map(trainer._league_chunk, jobs))
    results = [result for batch in batches for result in batch]
    return [sample for _, samples in sorted(results) for sample in samples]


def learner_seconds(trainer, model, samples, minibatch=512, updates=20):
    import numpy as np
    import torch
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(samples))
    start = time.perf_counter()
    count = 0
    for indices in np.array_split(order, range(minibatch, len(samples), minibatch))[:updates]:
        if hasattr(trainer, '_minibatch_loss'):
            loss = trainer._minibatch_loss(model, samples, indices)
        else:
            x = torch.from_numpy(np.stack([samples[i][0] for i in indices]))
            actions = torch.tensor([samples[i][1] for i in indices])
            targets = torch.tensor([samples[i][2] for i in indices])
            loss = torch.nn.functional.mse_loss(model(x).gather(1, actions[:, None]).squeeze(1), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
    return (time.perf_counter() - start) / count


def sample_digest(samples):
    h = hashlib.sha256()
    for s in samples:
        h.update(s[0].tobytes())
        h.update(repr(tuple(s[1:])).encode())
    return h.hexdigest()


def round_mode(args):
    import torch
    import phase2_trainer as trainer
    trainer._configure_threads()
    torch.use_deterministic_algorithms(True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = {}
    with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
        weights, pool_list, phase1, shares, exclusions, opponents = setup(args.weights, directory, args.encoder_version)
        context = trainer.mp.get_context('spawn')
        with context.Pool(args.workers, initializer=trainer._configure_threads) as pool:
            # Warm-up round: process start, imports, opponent caches.
            start = time.perf_counter()
            play(trainer, pool, weights, 57001, round_blocks(57001, 0, args.matches, pool_list, phase1, shares, exclusions), opponents)
            out['warmup_seconds'] = time.perf_counter() - start
            rounds, digests, counts = [], [], []
            for round_id in range(1, args.rounds + 1):
                blocks = round_blocks(57001, round_id, args.matches, pool_list, phase1, shares, exclusions)
                start = time.perf_counter()
                samples = play(trainer, pool, weights, 57001, blocks, opponents)
                rounds.append(time.perf_counter() - start)
                if round_id == 1:
                    # The timed updates and the check's challenger use round 1, so
                    # the check digest does not depend on --rounds.
                    first_samples = samples
                digests.append(sample_digest(samples))
                counts.append(len(samples))
        model = trainer.Phase2Network.from_state(weights)
        update = learner_seconds(trainer, model, first_samples)
        payload = len(pickle.dumps(first_samples, protocol=pickle.HIGHEST_PROTOCOL))
        check_start = time.perf_counter()
        if args.check_pairs:
            panel = trainer.load_panel(RUN2 / 'panel.json')
            for member in panel['members']:
                if member['kind'] == 'reference':
                    member['identity'] = trainer.REFERENCE_IDENTITIES[member['agent']]
            panel = trainer._resolve_panel(panel, RUN2)
            with context.Pool(args.workers, initializer=trainer._configure_threads) as pool:
                check_start = time.perf_counter()
                result = trainer._selection_check(model, weights, panel, 58000, args.check_pairs, args.check_pairs,
                                                  None, .05, pool, .01, .475, 5)
            out['check_digest'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        out['check_seconds'] = time.perf_counter() - check_start if args.check_pairs else None
    matches = args.matches * args.rounds
    out.update(workers=args.workers, matches_per_round=args.matches, rounds=args.rounds,
               round_seconds=rounds, matches_per_s=matches / sum(rounds), samples_per_s=sum(counts) / sum(rounds),
               samples_per_round=counts, sample_digests=digests, first_round_pickle_bytes=payload,
               update_seconds=update, check_pairs=args.check_pairs)
    print(json.dumps({k: v for k, v in out.items() if k != 'sample_digests'}, indent=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')


def profile_mode(args):
    import torch
    import phase2_trainer as trainer
    trainer._configure_threads()
    torch.use_deterministic_algorithms(True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
        weights, pool_list, phase1, shares, exclusions, opponents = setup(args.weights, directory, args.encoder_version)
        blocks = round_blocks(57001, 1, args.matches, pool_list, phase1, shares, exclusions)
        job = (weights, 57001, blocks, 'mixed', .3, opponents, .05, 5)
        trainer._league_chunk((weights, 57001, blocks[:2], 'mixed', .3, opponents, .05, 5))  # warm caches
        profile = cProfile.Profile()
        start = time.perf_counter()
        profile.enable()
        results = trainer._league_chunk(job)
        profile.disable()
        wall = time.perf_counter() - start
    samples = sum(len(s) for _, s in results)
    profile.dump_stats(str(args.output) + '.pstats')
    with open(str(args.output) + '.txt', 'w', encoding='utf-8') as stream:
        stream.write(f'matches={len(results)} samples={samples} wall={wall:.3f}s\n')
        stats = pstats.Stats(profile, stream=stream).strip_dirs()
        stats.sort_stats('cumulative').print_stats(60)
        stats.sort_stats('tottime').print_stats(40)
    print(f'matches={len(results)} samples={samples} wall={wall:.3f}s')


def main():
    sys.path.insert(0, str(ROOT / 'proyecto'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('round', 'profile'))
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--matches', type=int, default=240)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--encoder-version', type=int, default=1, choices=(1, 2, 3), help='widen the weights with zero columns')
    parser.add_argument('--check-pairs', type=int, default=0)
    args = parser.parse_args()
    (round_mode if args.mode == 'round' else profile_mode)(args)


if __name__ == '__main__':
    main()


def timed(payload):
    function, job = payload
    start = time.perf_counter()
    result = function(job)
    return result, time.perf_counter() - start, time.perf_counter()


_PROBE_WEIGHTS = None


def timed_cached(payload):
    global _PROBE_WEIGHTS
    path, job = payload
    import torch
    import phase2_trainer as trainer
    if _PROBE_WEIGHTS is None:
        _PROBE_WEIGHTS = torch.load(path, map_location='cpu', weights_only=True)
    start = time.perf_counter()
    result = trainer._league_chunk((_PROBE_WEIGHTS, *job[1:]))
    return result, time.perf_counter() - start, time.perf_counter()
