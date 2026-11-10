"""Run 2 settings golden for phase 2 speed work (bit-exact, workers 1 and 12).

Records or compares a short phase 2 run with the run 2 league shares, the run 2
six-member panel (two checks), held-out manifest, temperature 0.05, profile prior
5, no panel rotation and the auxiliary head off. Every comparison is exact:
tensors/arrays by raw bytes, floats by ``float.hex``. Only timing fields and
resolved filesystem paths are dropped.

    py proyecto/golden_phase2_speed.py record --workers 1 --output docs/perf/phase2-speed/golden.json
    py proyecto/golden_phase2_speed.py compare --workers 12 --reference docs/perf/phase2-speed/golden.json

The reference is recorded once, with workers 1, on the pre-optimization code. The
same reference must match for any worker count (ADR 0042).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUN2 = ROOT / 'experiments/phase2-run2'
SCRATCH = ROOT / 'training_output/phase2-speed'
TIMING_KEYS = frozenset(('elapsed', 'elapsed_seconds', 'resolved_path', 'training_seconds',
                         'check_seconds', 'generation_training_seconds'))


def semantic(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
        return ['tensor', str(array.dtype), list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()]
    if isinstance(value, np.ndarray):
        return ['array', str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]
    if isinstance(value, dict):
        return ['dict', [[str(key), semantic(item)] for key, item in sorted(value.items(), key=lambda x: str(x[0]))
                         if key not in TIMING_KEYS]]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [semantic(item) for item in value]]
    if isinstance(value, float):
        return ['float', value.hex()]
    if isinstance(value, np.generic):
        return [type(value).__name__, repr(value.item())]
    if isinstance(value, str) and ('/' in value or '\\' in value):
        return ['str', value.replace('\\', '/').split('/')[-1]]
    return [type(value).__name__, repr(value)]


def digest(value):
    return hashlib.sha256(json.dumps(semantic(value), sort_keys=True).encode()).hexdigest()


def sample_digest(sample):
    """Canonical learner sample: input bytes plus action, target and seat."""
    import numpy as np
    x = sample[0]
    if not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.shape != (1628,):
        raise AssertionError('learner input is not a float32 vector of 1628 values')
    return digest([x, int(sample[1]), float(sample[2]), int(sample[3]), list(sample[4:])])


class Recorder:
    def __init__(self):
        self.rounds = []

    def record(self, batches):
        matches = sorted(result for batch in batches for result in batch)
        self.rounds.append([[match_id, len(samples), digest([sample_digest(s) for s in samples])]
                            for match_id, samples in matches])


class PoolWrapper:
    def __init__(self, pool, recorder):
        self.pool, self.recorder = pool, recorder
        self._processes = pool._processes

    def __enter__(self):
        self.pool.__enter__()
        return self

    def __exit__(self, *args):
        return self.pool.__exit__(*args)

    def map(self, function, jobs, chunksize=1):
        results = self.pool.map(function, jobs, chunksize)
        if function.__name__ == '_league_chunk':
            self.recorder.record(results)
        return results


def configuration(workers, champions_dir):
    return dict(generations=3, matches_per_generation=96, matches_per_round=48, block_size=4, workers=workers,
                seed=57001, check_seed_base=58000, target='mixed', lr=3e-4, minibatch=256,
                temperature=0.05, profile_prior=5, patience=40, time_cap_seconds=10 ** 9,
                panel_path=RUN2 / 'panel.json', panel_rotation=False,
                held_out_manifest=RUN2 / 'held-out.json',
                league_shares=json.loads((RUN2 / 'league-shares.json').read_text(encoding='utf-8')),
                timing_log=True, panel_pairs=2, head_to_head_pairs=2, auxiliary=False,
                champions_dir=champions_dir, progress=True, console=None)


def run(workers):
    import torch
    import phase2_trainer as trainer
    recorder = Recorder()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='golden-', dir=SCRATCH) as root:
        directory = Path(root) / 'run'
        champions = Path(root) / 'champions'
        context = trainer.mp.get_context('spawn')

        class Context:
            def Pool(self, *a, **kw):
                return PoolWrapper(context.Pool(*a, **kw), recorder)

        real_chunk = trainer._league_chunk

        def chunk(job):
            result = real_chunk(job)
            recorder.record([result])
            return result

        import random
        import numpy as np
        # The trainer checkpoints the global generators; pin them like a fresh run would.
        random.seed(35)
        np.random.seed(35)
        torch.manual_seed(35)
        start = time.perf_counter()
        if workers == 1:
            with patch.object(trainer, '_league_chunk', chunk):
                result = trainer.train(directory, **configuration(workers, champions))
        else:
            with patch.object(trainer.mp, 'get_context', return_value=Context()):
                result = trainer.train(directory, **configuration(workers, champions))
        wall = time.perf_counter() - start
        logs = [json.loads(line) for line in (directory / 'generations.jsonl').read_text().splitlines()]
        golden = dict(
            model=digest(result.model.state_dict()), optimizer=digest(result.optimizer.state_dict()),
            log=digest(logs), log_entries=[digest(entry) for entry in logs],
            samples=result.samples, updates=result.updates,
            round_samples=[digest(r) for r in recorder.rounds],
            round_matches=[sum(1 for _ in r) for r in recorder.rounds],
            checkpoints={p.name: {key: digest(value) for key, value in
                                  torch.load(p, weights_only=False, map_location='cpu').items()
                                  if key not in TIMING_KEYS}
                         for p in sorted(directory.glob('checkpoint_*.pt'))},
            snapshots={p.name: digest(torch.load(p, weights_only=True, map_location='cpu'))
                       for p in sorted(directory.glob('generation_*.pt'))},
            champion={p.name: (digest(torch.load(p, weights_only=True, map_location='cpu')) if p.suffix == '.pt'
                               else p.read_text(encoding='utf-8'))
                      for p in sorted(champions.glob('reigning_champion.*'))},
            promotions=[entry.get('promoted') for entry in logs],
            checks=[entry['check']['promoted'] if entry.get('check') else None for entry in logs])
    return golden, wall


def main():
    sys.path.insert(0, str(ROOT / 'proyecto'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('record', 'compare'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    golden, wall = run(args.workers)
    print(f'workers={args.workers} samples={golden["samples"]} updates={golden["updates"]} wall={wall:.2f}s')
    if args.mode == 'record':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(golden=golden), indent=1) + '\n', encoding='utf-8')
        return
    expected = json.loads(args.reference.read_text(encoding='utf-8'))['golden']
    different = [key for key in expected if golden.get(key) != expected[key]]
    for name, keys in expected['checkpoints'].items():
        actual = golden['checkpoints'].get(name, {})
        different += [f'{name}:{key}' for key in keys if actual.get(key) != keys[key]]
    if different:
        print('GOLDEN MISMATCH: ' + ', '.join(different))
        sys.exit(1)
    print('GOLDEN OK')


if __name__ == '__main__':
    main()
