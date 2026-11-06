"""Run the whole phase 1 pipeline with one command: train -> reference evaluation -> publish -> commit.

Re-running the same command after an interruption resumes from the latest checkpoint.
All output is also written to <output>/pipeline.log.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def keep_awake():
    """Stop Windows from sleeping while the pipeline runs (no-op elsewhere)."""
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def run(step, command, log):
    banner = f"\n=== {step} === {time.strftime('%H:%M:%S')}\n$ {' '.join(command)}\n"
    print(banner, end='', flush=True)
    log.write(banner)
    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    if process.wait() != 0:
        raise SystemExit(f'{step} failed with exit code {process.returncode}')
    done = f'=== {step} done in {(time.perf_counter() - start) / 60:.1f} min ===\n'
    print(done, end='', flush=True)
    log.write(done)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='training_output/phase1')
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--population', type=int, default=64)
    parser.add_argument('--pairs', type=int, default=16)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validation-pairs', type=int, default=1000)
    parser.add_argument('--check-batch', type=int, default=200)
    parser.add_argument('--promotion-threshold', type=float, default=.55)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--eval-pairs', type=int, default=1000)
    parser.add_argument('--eval-workers', type=int, default=12)
    parser.add_argument('--no-commit', action='store_true', help='publish but do not git commit')
    args = parser.parse_args()

    keep_awake()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    with (output / 'pipeline.log').open('a', encoding='utf-8') as log:
        train = [python, '-u', str(HERE / 'phase1_trainer.py'), '--output', args.output,
                 '--generations', str(args.generations), '--population', str(args.population),
                 '--pairs', str(args.pairs), '--workers', str(args.workers), '--seed', str(args.seed),
                 '--validation-pairs', str(args.validation_pairs), '--check-batch', str(args.check_batch),
                 '--promotion-threshold', str(args.promotion_threshold), '--patience', str(args.patience)]
        checkpoints = sorted(output.glob('checkpoint_*.npz'))
        if checkpoints:
            train += ['--resume', str(checkpoints[-1])]
        run('1/4 training', train, log)

        # A fresh log avoids duplicate rows when the pipeline is re-run.
        (output / 'reference_evaluation.jsonl').unlink(missing_ok=True)
        run('2/4 reference evaluation', [python, '-u', str(HERE / 'reference_evaluation.py'), '--directory', args.output,
                                         '--promoted-only', '--pairs', str(args.eval_pairs),
                                         '--workers', str(args.eval_workers)], log)
        run('3/4 publish', [python, '-u', str(HERE / 'phase1_trainer.py'), '--output', args.output, '--publish'], log)
        if args.no_commit:
            print('=== 4/4 commit skipped (--no-commit) ===', flush=True)
        else:
            run('4/4 commit', ['git', 'add', 'champions/phase1'], log)
            run('4/4 commit', ['git', 'commit', '-m', f'Add phase 1 reigning champion and logs (seed {args.seed})'], log)
    print('\nPipeline finished. Champion and logs are in champions/phase1/.', flush=True)


if __name__ == '__main__':
    main()
