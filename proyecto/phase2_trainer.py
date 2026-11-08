"""Synchronous self-play and Monte Carlo learning for phase 2."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import json
import math
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
    value = int(np.random.SeedSequence([seed, match_id]).generate_state(1)[0])
    return 2 * value + 1

def _matches(args):
    weights, seed, ids, target = args
    model = Phase2Network()
    model.load_state_dict(weights)
    results = []
    for match_id in ids:
        seeds = np.random.SeedSequence([seed, match_id, 1]).generate_state(2)
        policies = [RecordingPolicy(model, seed=int(seeds[i])) for i in range(2)]
        result = play_match(*policies, seed=training_seed(seed, match_id), target=target)
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

def run_rounds(*, seed=0, rounds=1, matches_per_round=2, workers=1, target=15,
               lr=3e-4, minibatch=512, generation_rounds=None, log_path=None):
    if min(rounds, matches_per_round, workers, target, minibatch) < 1 or seed < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Invalid training configuration')
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
    context = mp.get_context('spawn')
    pool_context = context.Pool(workers, initializer=_configure_threads) if workers > 1 else nullcontext(None)
    with pool_context as pool:
        for round_id in range(rounds):
            weights = model.state_dict()
            ids = range(round_id * matches_per_round, (round_id + 1) * matches_per_round)
            chunks = [list(chunk) for chunk in np.array_split(list(ids), min(workers, matches_per_round))]
            jobs = [(weights, seed, [int(i) for i in chunk], target) for chunk in chunks]
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
    return RoundResult(model, optimizer, total, updates, last)

def train(directory, *, generations=1, matches_per_generation=120, matches_per_round=12,
          workers=1, seed=0, target=15, lr=3e-4, minibatch=512):
    """Round each generation up to a whole number of rounds."""
    if min(generations, matches_per_generation, matches_per_round) < 1:
        raise ValueError('Generation and match counts must be positive')
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / 'generations.jsonl'
    log_path.write_text('', encoding='utf-8')
    generation_rounds = (matches_per_generation + matches_per_round - 1) // matches_per_round
    return run_rounds(seed=seed, rounds=generations * generation_rounds,
                      matches_per_round=matches_per_round, workers=workers, target=target,
                      lr=lr, minibatch=minibatch, generation_rounds=generation_rounds, log_path=log_path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='training_output/phase2')
    parser.add_argument('--generations', type=int, default=1)
    parser.add_argument('--matches-per-generation', type=int, default=120)
    parser.add_argument('--matches-per-round', type=int, default=12)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--minibatch', type=int, default=512)
    args = vars(parser.parse_args())
    args['directory'] = args.pop('output')
    train(**args)

if __name__ == '__main__':
    main()
