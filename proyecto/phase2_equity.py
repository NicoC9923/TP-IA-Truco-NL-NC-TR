"""Match equity table for the run 3 training target (ADR 0062).

``EquityTable.equity(own, opp, target, own_is_mano)`` is P(the player with ``own`` points wins the
match) when the next hand is about to be dealt, for a 15 or 30 point match. ``own_is_mano`` says who is
mano of that next hand. A score at or above the target is terminal and exact (1 / 0).

Method (direct empirical cells, smoothed; not a Markov chain DP): every hand start of greedy
self-play matches is one observation per player view (own, opp, own_is_mano, won). A cell is
(wins + m * prior) / (visits + m) with m = ``prior_strength`` and a symmetric logistic prior on the
remaining-points difference. Sparse cells therefore fall back to the prior. The result is then forced
to rise with the own score and fall with the opponent's (average of the upper and lower monotone
envelopes) and symmetrised so that equity(own, opp, m) + equity(opp, own, not m) = 1. Both steps keep
the table monotone. Chosen over a DP on estimated per-hand point distributions because falta envido
makes the points of a hand depend on the score, which a score-independent hand model cannot express.

Build / check from the command line (never inside training):

    py proyecto/phase2_equity.py build --weights W.pt --matches N --seed S --workers 12 --output table.json
    py proyecto/phase2_equity.py check --table table.json --weights W.pt --matches N --seed S2 --workers 12
"""
import argparse
import hashlib
import json
import math
from collections import namedtuple
from pathlib import Path
import numpy as np

TARGETS = (15, 30)
VERSION = 1
MAX_WORKERS = 12
Observation = namedtuple('Observation', 'target score0 score1 mano winner')
"""A hand start seen from seat 0: scores, which seat is mano of the hand, who finally won the match."""


def _prior(target, scale):
    own = target - np.arange(target)[:, None]
    opp = target - np.arange(target)[None, :]
    return 1 / (1 + np.exp(-(opp - own) / (scale * np.sqrt(own + opp))))


def _monotone(grid):
    """Average of the upper and lower envelopes: nondecreasing in axis 0, nonincreasing in axis 1."""
    upper = np.maximum.accumulate(grid, axis=0)
    upper = np.maximum.accumulate(upper[:, ::-1], axis=1)[:, ::-1]
    lower = np.minimum.accumulate(grid[::-1], axis=0)[::-1]
    lower = np.minimum.accumulate(lower, axis=1)
    return (upper + lower) / 2


class EquityTable:
    def __init__(self, tables, metadata=None):
        self.tables = {int(t): np.asarray(v, dtype=np.float64) for t, v in tables.items()}
        self.metadata = dict(metadata or {})

    def equity(self, own, opp, target, own_is_mano):
        if target not in self.tables:
            raise ValueError('target must be 15 or 30')
        if own < 0 or opp < 0:
            raise ValueError('scores cannot be negative')
        if own >= target:
            return 1.0
        if opp >= target:
            return 0.0
        return float(self.tables[target][int(bool(own_is_mano))][own][opp])

    def is_monotone(self):
        return all((np.diff(grid, axis=1) >= -1e-12).all() and (np.diff(grid, axis=2) <= 1e-12).all()
                   for grid in self.tables.values())

    def to_json(self):
        return json.dumps({'version': VERSION, 'metadata': self.metadata,
                           'tables': {str(t): v.tolist() for t, v in sorted(self.tables.items())}}, sort_keys=True)

    def save(self, path):
        Path(path).write_text(self.to_json() + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version') != VERSION:
            raise ValueError('unsupported equity table version')
        return cls(data['tables'], data['metadata'])


def fit_equity_table(observations, *, prior_strength=5.0, prior_scale=1.5):
    if not math.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError('prior_strength must be positive')
    wins = {t: np.zeros((2, t, t)) for t in TARGETS}
    visits = {t: np.zeros((2, t, t)) for t in TARGETS}
    for row in observations:
        if row.target not in wins:
            raise ValueError('target must be 15 or 30')
        for seat, (own, opp) in enumerate(((row.score0, row.score1), (row.score1, row.score0))):
            m = int(row.mano == seat)
            visits[row.target][m][own][opp] += 1
            wins[row.target][m][own][opp] += row.winner == seat
    tables = {}
    for t in TARGETS:
        prior = _prior(t, prior_scale)
        smooth = np.stack([(wins[t][m] + prior_strength * prior) / (visits[t][m] + prior_strength) for m in (0, 1)])
        smooth = np.stack([_monotone(smooth[0]), _monotone(smooth[1])])
        # Mirror: the opponent's view of the same state, with mano swapped, must add up to one.
        mirror = 1 - np.stack([smooth[1].T, smooth[0].T])
        tables[t] = (smooth + mirror) / 2
    return EquityTable(tables, {'method': 'smoothed empirical cells, monotone envelope, symmetrised',
                                'prior_strength': prior_strength, 'prior_scale': prior_scale,
                                'observations': int(sum(v.sum() for v in visits.values()) / 2)})


def brier_comparison(table, observations):
    """Brier score of the equity and of the raw score difference over both player views of each hand start.

    The baseline is p = 1/2 + (own - opp) / (2 * target), clipped to [0, 1]. Lower is better."""
    equity_error = baseline_error = 0.0
    n = 0
    for row in observations:
        for seat, (own, opp) in enumerate(((row.score0, row.score1), (row.score1, row.score0))):
            won = float(row.winner == seat)
            p = table.equity(own, opp, row.target, row.mano == seat)
            q = min(1.0, max(0.0, 0.5 + (own - opp) / (2 * row.target)))
            equity_error += (p - won) ** 2
            baseline_error += (q - won) ** 2
            n += 1
    equity_score, baseline_score = equity_error / n, baseline_error / n
    return {'n': n, 'equity': equity_score, 'score_difference': baseline_score,
            'equity_better': equity_score < baseline_score}


def hand_start_observations(result, target):
    """One Observation per hand of a MatchResult (the first decision of each hand shows its start state)."""
    firsts = {}
    for decision in result.decisions:
        firsts.setdefault(decision['hand_index'], decision['observation'])
    return [Observation(target, *obs['score'], obs['mano'], result.winner)
            for _, obs in sorted(firsts.items())]


# ----------------------------------------------------------------- self-play collection

def _play_chunk(job):
    weights, seed, ids, share, profile_prior = job
    import numpy as np
    from phase2_torch import Phase2Network, Phase2Policy
    from phase2_trainer import match_settings, training_seed
    from symmetric_engine import play_match
    model = Phase2Network.from_state(weights)
    model.eval()
    rows = []
    for match_id in ids:
        seeds = np.random.SeedSequence([seed, match_id, 91]).generate_state(2)
        policies = [Phase2Policy(model, seed=int(s), greedy=True, profile_prior=profile_prior) for s in seeds]
        target, score = match_settings(seed, match_id, 'mixed', share)
        result = play_match(*policies, seed=training_seed(seed, match_id), target=target,
                            starting_score=score, deal_partition=1)
        rows.append((match_id, hand_start_observations(result, target)))
    return rows


def collect_observations(weights_path, *, matches, seed, workers=1, random_start_share=.5, profile_prior=5):
    """Greedy self-play of one network; results do not depend on the worker count."""
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f'workers must be between 1 and {MAX_WORKERS}')
    import torch
    from phase2_trainer import _configure_threads
    _configure_threads()
    weights = torch.load(weights_path, map_location='cpu', weights_only=True)
    ids = list(range(matches))
    chunks = [ids[i::workers] for i in range(workers)] if workers > 1 else [ids]
    jobs = [(weights, seed, chunk, random_start_share, profile_prior) for chunk in chunks if chunk]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(len(jobs), initializer=_configure_threads) as pool:
            batches = pool.map(_play_chunk, jobs, chunksize=1)
    else:
        batches = list(map(_play_chunk, jobs))
    return [row for _, rows in sorted(r for batch in batches for r in batch) for row in rows]


def build_equity_table(weights_path, *, matches, seed, workers=1, random_start_share=.5, profile_prior=5,
                       prior_strength=5.0):
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f'workers must be between 1 and {MAX_WORKERS}')
    rows = collect_observations(weights_path, matches=matches, seed=seed, workers=workers,
                                random_start_share=random_start_share, profile_prior=profile_prior)
    table = fit_equity_table(rows, prior_strength=prior_strength)
    table.metadata.update(seed=seed, matches=matches, selection='greedy', random_start_share=random_start_share,
                          profile_prior=profile_prior,
                          weights_sha256=hashlib.sha256(Path(weights_path).read_bytes()).hexdigest())
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build', 'check'))
    parser.add_argument('--weights', required=True)
    parser.add_argument('--matches', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--output')
    parser.add_argument('--table')
    parser.add_argument('--random-start-share', type=float, default=.5)
    parser.add_argument('--profile-prior', type=float, default=5)
    args = parser.parse_args(argv)
    if args.command == 'build':
        if not args.output or Path(args.output).exists():
            parser.error('--output is required and must not exist')
        table = build_equity_table(args.weights, matches=args.matches, seed=args.seed, workers=args.workers,
                                   random_start_share=args.random_start_share, profile_prior=args.profile_prior)
        table.save(args.output)
        print(f'wrote {args.output}: {table.metadata["observations"]} hand starts, monotone={table.is_monotone()}')
        return
    table = EquityTable.load(args.table)
    if args.seed == table.metadata.get('seed'):
        parser.error('held-out seed must differ from the build seed')
    held_out = collect_observations(args.weights, matches=args.matches, seed=args.seed, workers=args.workers,
                                    random_start_share=args.random_start_share, profile_prior=args.profile_prior)
    print(json.dumps({'monotone': table.is_monotone(), 'brier': brier_comparison(table, held_out)}, indent=1))


if __name__ == '__main__':
    main()
