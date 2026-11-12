"""Phase 2 baseline, profiles and pre-optimization semantic goldens (ticket #35).

Run from the repository root. Every invocation supervises a fresh child process
and kills its process tree after 175 seconds. Production modules are not edited.
See docs/perf/phase2-baseline.md for commands and interpretation.
"""
import argparse
import cProfile
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import platform
import pstats
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'experiments/phase2-selection-panel/panel-v0-measuring.json'
SCRATCH = ROOT / 'training_output/perf35'


def timed_call(payload):
    """Measure actor compute separately from parent wait; preserve result order."""
    function, job = payload
    start = time.perf_counter()
    result = function(job)
    return result, time.perf_counter() - start


class PoolProbe:
    def __init__(self, pool, records):
        self.pool, self.records = pool, records
        self._processes = pool._processes

    def __enter__(self):
        self.pool.__enter__()
        return self

    def __exit__(self, *args):
        return self.pool.__exit__(*args)

    def map(self, function, jobs, chunksize=1):
        start = time.perf_counter()
        batches = self.pool.map(timed_call, [(function, job) for job in jobs], chunksize)
        wall = time.perf_counter() - start
        results = [result for result, duration in batches]
        durations = [duration for result, duration in batches]
        # Separate microprobe: deliberately outside the measured map duration.
        start = time.perf_counter()
        job_payloads = [pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL) for value in jobs]
        result_payloads = [pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL) for value in results]
        payloads = job_payloads + result_payloads
        dump_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for payload in payloads:
            pickle.loads(payload)
        load_seconds = time.perf_counter() - start
        self.records.append(dict(kind=function.__name__, wall=wall,
                                 actor_sum=sum(durations), actor_max=max(durations),
                                 wait_residual=wall - max(durations),
                                 job_bytes=sum(map(len, job_payloads)), result_bytes=sum(map(len, result_payloads)),
                                 pickle_dump=dump_seconds, pickle_load=load_seconds))
        return results


class RoundProbe:
    """Only trace the trainer frame, using source markers rather than line offsets."""
    def __init__(self, function):
        self.code = function.__code__
        source, first = inspect.getsourcelines(function)
        markers = {'weights = model.state_dict()': 'setup',
                   'results = _play_round(': 'actors',
                   'last = [sample for _, samples in results': 'merge',
                   'for indices in np.array_split': 'learner',
                   'recent_loss = statistics.mean': 'boundary',
                   'reporter.round_done(': 'progress'}
        self.lines = {first + i: phase for i, line in enumerate(source)
                      for marker, phase in markers.items() if line.strip().startswith(marker)}
        if len(self.lines) != len(markers):
            raise RuntimeError('Trainer source markers changed; update RoundProbe')
        self.rounds = []
        self.current = None
        self.phase = None
        self.tick = None

    def trace(self, frame, event, arg):
        if frame.f_code is not self.code:
            return None
        phase = self.lines.get(frame.f_lineno) if event == 'line' else None
        if phase is not None or event == 'return':
            now = time.perf_counter()
            if self.current is not None:
                self.current[self.phase] = self.current.get(self.phase, 0) + now - self.tick
            if phase == 'setup':
                self.current = {}
                self.rounds.append(self.current)
            self.phase, self.tick = phase, now
        return self.trace


def seed_all(seed=35):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def semantic(value):
    """Typed, byte-exact representation; never use allclose or rounded floats."""
    import numpy as np
    import torch
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
        return ['tensor', str(array.dtype), list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()]
    if isinstance(value, np.ndarray):
        return ['array', str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]
    if isinstance(value, dict):
        return ['dict', [[str(key), semantic(item)] for key, item in sorted(value.items(), key=lambda x: str(x[0]))
                         if key not in ('elapsed', 'elapsed_seconds', 'resolved_path')]]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [semantic(item) for item in value]]
    if isinstance(value, float):
        return ['float', value.hex()]
    if isinstance(value, str) and ('/' in value or '\\' in value):
        # Panel paths are location dependent; identity is separately SHA-checked.
        return ['str', value.replace('\\', '/').split('/')[-1]]
    return [type(value).__name__, repr(value)]


def digest(value):
    return hashlib.sha256(json.dumps(semantic(value), sort_keys=True).encode()).hexdigest()


def manifest(directory, result):
    import torch
    logs = [json.loads(line) for line in (directory / 'generations.jsonl').read_text().splitlines()]
    return dict(model=digest(result.model.state_dict()), optimizer=digest(result.optimizer.state_dict()),
                log=digest(logs), samples=result.samples, updates=result.updates,
                checkpoints={p.name: digest(torch.load(p, weights_only=False, map_location='cpu'))
                             for p in sorted(directory.glob('checkpoint_*.pt'))},
                snapshots={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(directory.glob('generation_*.pt'))})


def configuration(args, golden=False):
    return dict(generations=2, matches_per_generation=16 if golden else args.matches,
                matches_per_round=8 if golden else args.matches, block_size=4,
                workers=args.workers, seed=getattr(args, 'seed', 35), check_seed_base=getattr(args, 'check_seed_base', 10000), target='mixed', minibatch=32 if golden else 512,
                panel_path=PANEL, panel_pairs=2 if golden else args.pairs,
                head_to_head_pairs=2 if golden else args.direct_pairs,
                auxiliary=args.auxiliary, card_prediction_situations=8 if golden else 400,
                progress=True, champions_dir=None)


def golden(args):
    import tempfile
    import phase2_trainer as trainer
    if args.partial_directory:
        seed_all()
        trainer.train(args.partial_directory, stop_after_round=args.resume_round, **configuration(args, golden=True))
        return dict(partial=True)
    with tempfile.TemporaryDirectory(prefix='golden-', dir=SCRATCH) as root:
        directory = Path(root)
        config = configuration(args, golden=True)
        seed_all()
        if args.resume:
            command = [sys.executable, str(Path(__file__).resolve()), 'golden', '--child',
                       '--workers', str(args.workers), '--partial-directory', str(directory),
                       '--resume-round', str(args.resume_round),
                       '--output', str(directory / 'partial.json')]
            if args.auxiliary:
                command.append('--auxiliary')
            subprocess.run(command, check=True)
            result = trainer.train(directory, resume=directory / f'checkpoint_{args.resume_round:06d}.pt', **config)
        else:
            result = trainer.train(directory, **config)
        actual = manifest(directory, result)
    if args.reference:
        expected = json.loads(Path(args.reference).read_text())['golden']
        if actual != expected:
            different = [key for key in actual if actual[key] != expected[key]]
            raise AssertionError('Golden mismatch: ' + ', '.join(different))
    return dict(golden=actual, workers=args.workers, resume=args.resume, resume_round=args.resume_round if args.resume else None,
                auxiliary=args.auxiliary, configuration=config)


def bench(args):
    import tempfile
    import phase2_trainer as trainer
    import phase2_card_metrics as metrics
    from unittest.mock import patch
    records, validations, checks = [], [], []
    probe = RoundProbe(trainer.run_rounds)
    context = trainer.mp.get_context('spawn')
    real_validation = metrics.validation_predictions
    real_check = trainer._selection_check

    class Context:
        def Pool(self, *a, **kw):
            return PoolProbe(context.Pool(*a, **kw), records)

    def validation(*a, **kw):
        start = time.perf_counter()
        result = real_validation(*a, **kw)
        validations.append(time.perf_counter() - start)
        return result

    def check(*a, **kw):
        start = time.perf_counter()
        result = real_check(*a, **kw)
        checks.append(time.perf_counter() - start)
        return result

    profile = cProfile.Profile()
    with tempfile.TemporaryDirectory(prefix='bench-', dir=SCRATCH) as root:
        seed_all()
        start = time.perf_counter()
        with patch.object(trainer.mp, 'get_context', return_value=Context()), patch.object(metrics, 'validation_predictions', validation), patch.object(trainer, '_selection_check', check):
            sys.settrace(probe.trace)
            if args.profile:
                profile.enable()
            try:
                result = trainer.train(root, **configuration(args))
            finally:
                profile.disable()
                sys.settrace(None)
        wall = time.perf_counter() - start
        hashes = manifest(Path(root), result)
    if args.profile:
        save_profile(profile, args.output)
    actor_wall = sum(r['wall'] for r in records if r['kind'] == '_league_chunk')
    return dict(configuration=configuration(args), wall=wall, matches=2 * args.matches,
                samples=result.samples, updates=result.updates, matches_per_s=2 * args.matches / wall,
                samples_per_s=result.samples / wall, actor_matches_per_s=2 * args.matches / actor_wall,
                learner_seconds_per_update=sum(r.get('learner', 0) for r in probe.rounds) / result.updates,
                rounds=probe.rounds, maps=records, validation_seconds=validations, check_seconds=checks, hashes=hashes)


def save_profile(profile, output):
    profile.dump_stats(str(output) + '.pstats')
    with open(str(output) + '.profile.txt', 'w', encoding='utf-8') as stream:
        stats = pstats.Stats(profile, stream=stream).strip_dirs()
        stats.sort_stats('cumulative').print_stats(55)
        stats.sort_stats('tottime').print_stats(35)
        stats.sort_stats('cumulative').print_stats('compatible_mask|backward|adam.py|cross_entropy|optimizer.py')
    path = Path(str(output) + '.profile.txt')
    path.write_text(path.read_text(encoding='utf-8').rstrip() + '\n', encoding='utf-8')


def profile_component(args):
    import phase2_trainer as trainer
    import phase2_card_metrics as metrics
    trainer._configure_threads()
    seed_all()
    model = trainer.Phase2Network(seed=35, auxiliary=True)
    profile = cProfile.Profile()
    if args.mode == 'actor':
        spec = dict(identity='current', category='self', neutral_profile=False, style=None, offset=0)
        job = (model.state_dict(), 35, list(range(4)), 'mixed', .3, spec, {}, .1)
        profile.enable()
        start = time.perf_counter()
        results = trainer._league_block(job)
        wall = time.perf_counter() - start
        profile.disable()
        result = dict(wall=wall, matches=4, samples=sum(len(samples) for _, samples in results))
    else:
        timings = []
        for _ in range(2):
            profile.enable()
            start = time.perf_counter()
            values = metrics.validation_predictions(model, schedule_seed=10000, target='mixed', temperature=.1, situation_count=400)
            timings.append(time.perf_counter() - start)
            profile.disable()
        result = dict(seconds=timings, metrics=values)
    save_profile(profile, args.output)
    return result


def auxiliary_probe(args):
    """Time actual 512-sample updates and reject changed gradients/tensors."""
    import numpy as np
    import torch
    import phase2_trainer as trainer
    trainer._configure_threads()
    torch.use_deterministic_algorithms(True)
    seed_all()
    initial = trainer.Phase2Network(seed=35, auxiliary=True)
    spec = dict(identity='current', category='self', neutral_profile=False, style=None, offset=0)
    results = trainer._league_block((initial.state_dict(), 35, list(range(4)), 'mixed', .3, spec, {}, .1))
    samples = [sample for _, batch in results for sample in batch]
    samples = (samples * (512 // len(samples) + 1))[:512]
    x = torch.from_numpy(np.stack([s[0] for s in samples]))
    actions = torch.tensor([s[1] for s in samples])
    targets = torch.tensor([s[2] for s in samples])
    labels = torch.tensor([s[5] for s in samples])
    ids = torch.from_numpy(trainer._IDS)
    from opponent_inference import _hand_tantos
    def legacy_mask(evidence):
        masks = np.ones((len(evidence), len(ids)), dtype=bool)
        tantos = _hand_tantos()
        for row, (own, played, declared, upper) in zip(masks, evidence):
            for card in own:
                row &= ~(trainer._IDS == card).any(axis=1)
            for card in played:
                row &= (trainer._IDS == card).any(axis=1)
            if declared >= 0:
                row &= tantos == declared
            row &= tantos <= upper
        return masks
    records = {}
    from phase2_torch import combination_scores
    for variant in ('original', 'legacy_masks', 'index_select', 'split_sum', 'shared_trunk', 'ordered_backward'):
        model = trainer.Phase2Network(seed=35, auxiliary=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        times, losses, gradients = [], [], []
        for _ in range(8):
            row = {}
            tick = time.perf_counter()
            def mark(name):
                nonlocal tick
                now = time.perf_counter()
                row[name], tick = now - tick, now
            if variant == 'shared_trunk':
                features = model.layers[:4](x)
                q = model.layers[4](features)
            else:
                q = model(x)
            loss = torch.nn.functional.mse_loss(q.gather(1, actions[:, None]).squeeze(1), targets)
            mark('q_forward')
            mask_function = legacy_mask if variant == 'legacy_masks' else trainer.compatible_mask
            masks = torch.from_numpy(mask_function([s[4] for s in samples]))
            mark('masks')
            logits = model.card_head(features) if variant == 'shared_trunk' else model.card_logits(x)
            mark('card_forward')
            if variant == 'ordered_backward':
                scores = combination_scores(logits, ids)
            elif variant == 'index_select':
                scores = logits.index_select(1, ids.flatten()).reshape(512, len(ids), 3).sum(dim=2)
            elif variant == 'split_sum':
                scores = logits[:, ids[:, 0]] + logits[:, ids[:, 1]] + logits[:, ids[:, 2]]
            else:
                scores = logits[:, ids].sum(dim=2)
            mark('combination_scores')
            if not masks[torch.arange(512), labels].all():
                raise AssertionError('Invalid fixture evidence')
            scores = scores.masked_fill(~masks, float('-inf'))
            mark('masked_fill')
            loss = loss + .1 * torch.nn.functional.cross_entropy(scores, labels)
            mark('cross_entropy')
            optimizer.zero_grad()
            loss.backward()
            mark('backward')
            gradients.append(digest([p.grad for p in model.parameters()]))
            tick = time.perf_counter()
            optimizer.step()
            mark('adam')
            losses.append(float(loss.detach()))
            times.append(row)
        records[variant] = dict(mean_seconds={k: sum(r[k] for r in times) / len(times) for k in times[0]},
                                losses=losses, gradients=gradients, model=digest(model.state_dict()),
                                optimizer=digest(optimizer.state_dict()))
        records[variant]['exact'] = all(records[variant][k] == records['original'][k]
                                       for k in ('losses', 'gradients', 'model', 'optimizer'))
    return dict(batch_size=512, combinations=len(ids), updates=8, variants=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('bench', 'golden', 'actor', 'validation', 'auxiliary'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=35)
    parser.add_argument('--check-seed-base', type=int, default=10000)
    parser.add_argument('--matches', type=int, default=96)
    parser.add_argument('--pairs', type=int, default=100)
    parser.add_argument('--direct-pairs', type=int, default=200)
    parser.add_argument('--auxiliary', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-round', type=int, default=1)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--partial-directory', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not args.child:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--child'])
        try:
            return process.wait(timeout=175)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=True)
            else:
                process.kill()
            process.wait()
            raise RuntimeError('175-second benchmark limit exceeded; process tree stopped')
    import torch
    result = golden(args) if args.mode == 'golden' else bench(args) if args.mode == 'bench' else auxiliary_probe(args) if args.mode == 'auxiliary' else profile_component(args)
    result['environment'] = dict(python=sys.version, torch=torch.__version__, platform=platform.platform(),
                                 cpu=platform.processor(), logical_cpus=os.cpu_count(),
                                 revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), **{k: v for k, v in result.items()
                                                    if k in ('wall', 'matches_per_s', 'samples_per_s', 'seconds', 'samples', 'updates')})))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
