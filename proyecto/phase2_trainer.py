"""Synchronous self-play and Monte Carlo learning for phase 2."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import json
import math
import random
import time
from pathlib import Path
from contextlib import nullcontext
import multiprocessing as mp
from dataclasses import dataclass
import numpy as np
import torch
from phase1_network import ACTIONS
from phase2_torch import Phase2Network, Phase2Policy
from symmetric_engine import play_match

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
    """Odd training deal seeds cannot overlap the even validation schedule."""
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
    model = Phase2Network()
    model.load_state_dict(weights)
    results = []
    for match_id in ids:
        seeds = np.random.SeedSequence([seed, match_id, 1]).generate_state(2)
        policies = [RecordingPolicy(model, seed=int(seeds[i])) for i in range(2)]
        match_target, score = match_settings(seed, match_id, target, random_start_share)
        result = play_match(*policies, seed=training_seed(seed, match_id), target=match_target, starting_score=score)
        samples = [(x, action, 1.0 if seat == result.winner else -1.0, seat)
                   for seat, policy in enumerate(policies) for x, action in policy.samples]
        results.append((match_id, samples))
    return results

@dataclass
class RoundResult:
    model: Phase2Network
    optimizer: object
    samples: int
    updates: int
    last_samples: list

def run_rounds(*, seed=0, rounds=1, matches_per_round=2, workers=1, target='mixed',
               lr=3e-4, minibatch=512, generation_rounds=None, log_path=None,
               random_start_share=.3, checkpoint_dir=None, config=None, resume_state=None,
               stop_after_round=None):
    if (min(rounds, matches_per_round, workers, minibatch) < 1 or seed < 0
            or (target != 'mixed' and (not isinstance(target, int) or target < 1))
            or not 0 <= random_start_share <= 1 or not math.isfinite(lr) or lr <= 0):
        raise ValueError('Invalid training configuration')
    if ((generation_rounds is None) != (log_path is None)
            or (generation_rounds is not None and generation_rounds < 1)
            or (checkpoint_dir is not None and generation_rounds is None)):
        raise ValueError('generation_rounds and log_path must be supplied together for checkpoints')
    _configure_threads()
    torch.use_deterministic_algorithms(True)
    model = Phase2Network(seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    total = updates = 0
    last = []
    generation_samples = generation_updates = 0
    losses = []
    started = time.monotonic()
    first_round = 0
    if resume_state is not None:
        model.load_state_dict(resume_state['model'])
        optimizer.load_state_dict(resume_state['optimizer'])
        first_round = resume_state['round']
        total, updates = resume_state['samples'], resume_state['updates']
        generation_samples, generation_updates = resume_state['generation_samples'], resume_state['generation_updates']
        losses = resume_state['losses']
        random.setstate(resume_state['python_rng'])
        np.random.set_state(resume_state['numpy_rng'])
        torch.set_rng_state(resume_state['torch_rng'])
        rng.bit_generator.state = resume_state['shuffle_rng']
    context = mp.get_context('spawn')
    pool_context = context.Pool(workers, initializer=_configure_threads) if workers > 1 else nullcontext(None)
    with pool_context as pool:
        for round_id in range(first_round, min(rounds, stop_after_round or rounds)):
            weights = model.state_dict()
            ids = range(round_id * matches_per_round, (round_id + 1) * matches_per_round)
            chunks = [list(chunk) for chunk in np.array_split(list(ids), min(workers, matches_per_round))]
            jobs = [(weights, seed, [int(i) for i in chunk], target, random_start_share) for chunk in chunks]
            batches = pool.map(_matches, jobs, chunksize=1) if pool else list(map(_matches, jobs))
            results = [result for batch in batches for result in batch]
            last = [sample for _, samples in sorted(results) for sample in samples]
            total += len(last)
            generation_samples += len(last)
            for indices in np.array_split(rng.permutation(len(last)), range(minibatch, len(last), minibatch)):
                x = torch.from_numpy(np.stack([last[i][0] for i in indices]))
                actions = torch.tensor([last[i][1] for i in indices])
                targets = torch.tensor([last[i][2] for i in indices])
                loss = torch.nn.functional.mse_loss(model(x).gather(1, actions[:, None]).squeeze(1), targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                updates += 1
                generation_updates += 1
            if generation_rounds and (round_id + 1) % generation_rounds == 0:
                entry = dict(phase=2, generation=(round_id + 1) // generation_rounds,
                             matches=generation_rounds * matches_per_round,
                             learner_samples=generation_samples, updates=generation_updates,
                             mean_loss=sum(losses) / len(losses), elapsed_seconds=time.monotonic() - started,
                             check=None, promotion_check=None, promoted=False,
                             phase_ended=False, fitness=None, behavior=None,
                             diversity=None, champion=None, example_hand=None)
                with Path(log_path).open('a', encoding='utf-8') as log:
                    log.write(json.dumps(entry) + '\n')
                generation_samples = generation_updates = 0
                losses = []
                started = time.monotonic()
            if checkpoint_dir is not None:
                state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             round=round_id + 1, generation=(round_id + 1) // generation_rounds,
                             matches=(round_id + 1) * matches_per_round, samples=total, updates=updates,
                             generation_samples=generation_samples, generation_updates=generation_updates,
                             losses=losses, config=config, python_rng=random.getstate(),
                             numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                             shuffle_rng=rng.bit_generator.state)
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
    return RoundResult(model, optimizer, total, updates, last)

def train(directory, *, generations=1, matches_per_generation=120, matches_per_round=12,
          workers=1, seed=0, target='mixed', lr=3e-4, minibatch=512,
          random_start_share=.3, resume=None, stop_after_round=None):
    """Round each generation up to a whole number of rounds."""
    if min(generations, matches_per_generation, matches_per_round) < 1:
        raise ValueError('Generation and match counts must be positive')
    config = dict(matches_per_generation=matches_per_generation,
                  matches_per_round=matches_per_round, seed=seed, target=target,
                  lr=lr, minibatch=minibatch, random_start_share=random_start_share)
    generation_rounds = (matches_per_generation + matches_per_round - 1) // matches_per_round
    output = Path(directory).resolve()
    if resume and Path(resume).resolve().parent != output:
        raise ValueError('Resume checkpoint must be inside the output directory')
    state = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    old_config = dict(state['config']) if state is not None else None
    if old_config is not None:
        old_config.pop('workers', None)
        old_config.pop('generations', None)
    if state is not None and old_config != config:
        raise ValueError('Checkpoint configuration does not match')
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
                      resume_state=state, stop_after_round=stop_after_round)

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
    parser.add_argument('--resume')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--minibatch', type=int, default=512)
    args = vars(parser.parse_args())
    args['directory'] = args.pop('output')
    train(**args)

if __name__ == '__main__':
    main()
