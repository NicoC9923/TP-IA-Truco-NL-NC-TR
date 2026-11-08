"""Synchronous self-play and Monte Carlo learning for phase 2."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import multiprocessing as mp
from dataclasses import dataclass
import numpy as np
import torch
from phase1_network import ACTIONS
from phase2_torch import Phase2Network, Phase2Policy
from symmetric_engine import play_match

class RecordingPolicy(Phase2Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = []

    def __call__(self, observation, legal_actions):
        # Encode only once: inference maintains incremental state.
        inputs = self.encode(observation, legal_actions)
        with torch.no_grad():
            values = self.model(torch.from_numpy(inputs)).numpy()
        logits = np.array([values[ACTIONS.index(a)] for a in legal_actions], dtype=np.float64) / self.temperature
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        action = legal_actions[self.rng.choice(len(legal_actions), p=probabilities)]
        self.samples.append((inputs, ACTIONS.index(action)))
        return action

def _match(args):
    weights, seed, match_id, target = args
    torch.set_num_threads(1)
    model = Phase2Network()
    model.load_state_dict(weights)
    seeds = np.random.SeedSequence([seed, match_id]).generate_state(3)
    policies = [RecordingPolicy(model, seed=int(seeds[i+1])) for i in range(2)]
    result = play_match(*policies, seed=int(seeds[0]), target=target)
    samples = [(x, action, 1.0 if seat == result.winner else -1.0, seat)
               for seat, policy in enumerate(policies) for x, action in policy.samples]
    return match_id, samples

@dataclass
class RoundResult:
    model: Phase2Network
    optimizer: object
    samples: int
    updates: int
    last_samples: list

def run_rounds(*, seed=0, rounds=1, matches_per_round=2, workers=1, target=15, lr=3e-4, minibatch=512):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = Phase2Network(seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    total = updates = 0
    last = []
    context = mp.get_context('spawn')
    for round_id in range(rounds):
        weights = model.state_dict()
        jobs = [(weights, seed, round_id * matches_per_round + i, target) for i in range(matches_per_round)]
        if workers == 1:
            results = list(map(_match, jobs))
        else:
            with context.Pool(workers) as pool:
                results = pool.map(_match, jobs)
        last = [sample for _, samples in sorted(results) for sample in samples]
        total += len(last)
        for indices in np.array_split(rng.permutation(len(last)), range(minibatch, len(last), minibatch)):
            x = torch.from_numpy(np.stack([last[i][0] for i in indices]))
            actions = torch.tensor([last[i][1] for i in indices])
            targets = torch.tensor([last[i][2] for i in indices])
            loss = torch.nn.functional.mse_loss(model(x).gather(1, actions[:, None]).squeeze(1), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
    return RoundResult(model, optimizer, total, updates, last)
