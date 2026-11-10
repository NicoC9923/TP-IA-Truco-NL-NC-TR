"""Run 3 pilot (ADR 0062): about 1,000 updates from the warm start with every run 3 setting, then greedy
mirror pairs against fixed Adaptive Utility on seeds no earlier experiment used.

    py proyecto/phase2_run3_pilot.py --warm-start champions/phase2/reigning_champion.pt \\
        --equity-table experiments/phase2-run3/equity-table.json --output training_output/phase2-run3-pilot

Reports win rate against the 68% threshold, the measured end-to-end matches per second (actors and
learner, warm-started network) and the matches per generation that give about 7.5-minute generations.
The pilot output is only a go / no-go signal. It never selects or ships anything.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import phase2_final_exam as exam

MAX_WORKERS = 12
THRESHOLD = 0.68
ROOT = Path(__file__).resolve().parent.parent
RUN2 = ROOT / 'experiments/phase2-run2'

# Proposed seeds, audited by audit_seeds() against the RULE.txt namespaces and the final exam.
SEEDS = {'equity_build': 67001, 'equity_held_out': 67002, 'pilot_training': 63001,
         'run3_training': 65001, 'run3_check_base': 66000}  # checks use 66000..66999
PILOT_PLAN = exam.make_plan('P', 92001, 132, 300, ['AU'])
_DEFAULT = object()


def matches_per_generation(rate, *, seconds=450, per_round=240, check_share=.05):
    """Matches that fill ``seconds`` of a generation, minus the check's share, as whole rounds (at least one)."""
    if not rate > 0 or not seconds > 0 or not 0 <= check_share < 1 or per_round < 1:
        raise ValueError('rate and seconds must be positive and check_share in [0, 1)')
    rounds = int(rate * seconds * (1 - check_share) / per_round + 0.5)
    return max(1, rounds) * per_round


def audit_seeds(training_seed=None):
    """Check the proposed seeds against earlier namespaces (RULE.txt via the final exam's tables)."""
    from phase2_promotion import RESERVED_SEEDS
    used = set(exam.EARLIER_SCHEDULE_SEEDS) | set(RESERVED_SEEDS)
    for plan in (exam.PART_T_PLAN, exam.PART_E_PLAN):
        used |= {plan['deal_base'] + b for b in range(len(plan['opponents']))}
    seeds = dict(SEEDS)
    if training_seed is not None:
        seeds['pilot_training'] = training_seed
    conflicts = []
    for name, seed in seeds.items():
        span = range(seed, seed + (1000 if name == 'run3_check_base' else 1))
        if any(s in used for s in span):
            conflicts.append(f'{name} {seed}')
    deals, actions = exam.all_seeds(PILOT_PLAN)
    for plan in (exam.PART_T_PLAN, exam.PART_E_PLAN):
        other_deals, other_actions = exam.all_seeds(plan)
        if deals & other_deals or actions & other_actions:
            conflicts.append('pilot plan overlaps the final exam')
    if any(low <= a <= high for a in actions for low, high in exam.EARLIER_ACTION_RANGES):
        conflicts.append('pilot action seeds overlap an earlier range')
    if {PILOT_PLAN['deal_base']} & used:
        conflicts.append(f"pilot deal base {PILOT_PLAN['deal_base']}")
    return {'seeds': seeds, 'pilot_plan': PILOT_PLAN, 'conflicts': conflicts}


# ----------------------------------------------------------------- evaluation workers

_NETS = {}


def _initialize(paths):
    import torch
    from phase2_torch import Phase2Network
    torch.set_num_threads(1)
    for name, path in paths.items():
        _NETS[name] = Phase2Network.load(path)
        _NETS[name].eval()


def _pair(job):
    name, i, profile_prior, pairs = job
    spec = {'kind': 'phase2', 'model': _NETS[name], 'temperature': None, 'profile_prior': profile_prior}
    record = exam.play_record(dict(PILOT_PLAN, count=pairs), 0, i, {name: spec}, {'kind': 'au'})
    return name, i, exam.pair_score(record['winners'][name])


def evaluate(paths, *, pairs, workers, profile_prior):
    """Greedy mirror pairs against fixed Adaptive Utility (fresh policies and RNGs in every game).

    The action seeds depend on the plan's pair count, so a run with the planned 300 pairs uses exactly PILOT_PLAN."""
    jobs = [(name, i, profile_prior, pairs) for name in paths for i in range(pairs)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(min(workers, len(jobs)), initializer=_initialize,
                                          initargs=(paths,)) as pool:
            rows = pool.map(_pair, jobs, chunksize=1)
    else:
        _initialize(paths)
        rows = list(map(_pair, jobs))
    result = {'pairs': pairs}
    for name in paths:
        scores = [score for n, _, score in sorted(rows, key=lambda r: (r[0], r[1])) if n == name]
        result[name] = exam.paired_stat(scores)
    return result


def run_pilot(*, warm_start, equity_table, output, updates=1000, samples_per_match=35, matches_per_round=240,
              block_size=4, workers=MAX_WORKERS, pairs=300, seed=None, lr=3e-4, minibatch=512, encoder_version=3,
              temperature=.05, profile_prior=5, target_blend=.5, generation_seconds=450, check_share=.05,
              league_shares=_DEFAULT, held_out_manifest=_DEFAULT):
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f'workers must be between 1 and {MAX_WORKERS}')
    output = Path(output)
    if output.exists():
        raise FileExistsError(str(output))
    seed = SEEDS['pilot_training'] if seed is None else seed
    if league_shares is _DEFAULT:
        league_shares = json.loads((RUN2 / 'league-shares.json').read_text(encoding='utf-8'))
    if held_out_manifest is _DEFAULT:
        held_out_manifest = RUN2 / 'held-out.json'
    import phase2_trainer as trainer
    rounds = max(1, math.ceil(updates * minibatch / (matches_per_round * samples_per_match)))
    output.mkdir(parents=True)
    training = output / 'training'
    trained = trainer.train(training, generations=1, matches_per_generation=rounds * matches_per_round,
                            matches_per_round=matches_per_round, block_size=block_size, workers=workers, seed=seed,
                            target='mixed', lr=lr, minibatch=minibatch, panel_path=None, panel_rotation=False,
                            held_out_manifest=held_out_manifest, timing_log=True, temperature=temperature,
                            profile_prior=profile_prior, league_shares=league_shares, champions_dir=output / 'champions',
                            time_cap_seconds=10 ** 9, progress=False, encoder_version=encoder_version,
                            warm_start=warm_start, target_blend=target_blend, equity_table=equity_table,
                            check_selection='greedy')
    final = output / 'pilot_final.pt'
    trained.model.save(final)
    entry = json.loads((training / 'generations.jsonl').read_text(encoding='utf-8').splitlines()[-1])
    rate = entry['matches'] / entry['training_seconds']
    evaluation = evaluate({'warm_start': str(warm_start), 'pilot': str(final)}, pairs=pairs, workers=workers,
                          profile_prior=profile_prior)
    report = {
        'passes': bool(evaluation['pilot']['mean'] >= THRESHOLD),
        'threshold': THRESHOLD,
        'training': {key: entry.get(key) for key in ('matches', 'learner_samples', 'updates', 'mean_loss',
                                                     'greedy_disagreement', 'decisions', 'training_seconds')},
        'throughput': {'matches_per_second': rate, 'basis': 'matches / training seconds, actors and learner, warm start'},
        'generation_sizing': {'matches_per_second': rate, 'target_generation_seconds': generation_seconds,
                              'check_share': check_share, 'matches_per_round': matches_per_round,
                              'matches_per_generation': matches_per_generation(
                                  rate, seconds=generation_seconds, per_round=matches_per_round, check_share=check_share)},
        'evaluation': evaluation,
        'seeds': {'training': seed, 'evaluation': PILOT_PLAN},
        'settings': {'lr': lr, 'minibatch': minibatch, 'encoder_version': encoder_version, 'target_blend': target_blend,
                     'temperature': temperature, 'profile_prior': profile_prior, 'updates_requested': updates,
                     'rounds': rounds, 'workers': workers, 'check_selection': 'greedy'},
    }
    (output / 'pilot.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--warm-start', required=True)
    parser.add_argument('--equity-table', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--samples-per-match', type=float, default=35)
    parser.add_argument('--matches-per-round', type=int, default=240)
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--pairs', type=int, default=300)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--encoder-version', type=int, default=3)
    parser.add_argument('--generation-seconds', type=float, default=450)
    parser.add_argument('--temperature', type=float, default=.05, help='training rollout temperature')
    args = parser.parse_args(argv)
    audit = audit_seeds(args.seed)
    if audit['conflicts']:
        parser.error('seed conflicts: ' + ', '.join(audit['conflicts']))
    report = run_pilot(**{k: v for k, v in vars(args).items()})
    print(json.dumps({k: report[k] for k in ('passes', 'threshold', 'throughput', 'generation_sizing')}, indent=1))
    print('pilot win rate vs Adaptive Utility:', round(report['evaluation']['pilot']['mean'], 4),
          '(warm start:', round(report['evaluation']['warm_start']['mean'], 4), ')')
    sys.exit(0 if report['passes'] else 3)


if __name__ == '__main__':
    main()
