"""Live progress for the phase 2 trainer: an atomic JSON file and terminal lines.

Purely observational. Nothing here feeds the semantic log, the checkpoint or any
random stream, so training stays deterministic with progress on or off.
"""
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

PROGRESS_FILE = 'progress.json'
RATE_WINDOW = 5  # rounds


def format_duration(seconds):
    minutes = int(max(0, seconds) // 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m' if hours else f'{minutes}m'


def generation_line(entry, elapsed):
    """One permanent terminal line per generation."""
    check = entry.get('check')
    if check is None:
        verdict = 'primera campeona' if entry.get('champion') == entry['generation'] else 'sin control'
        return f"gen {entry['generation']} · {verdict} · {format_duration(elapsed)}"
    gain = check['gain'] * 100
    half = (check['upper_bound'] - check['lower_bound']) / 2 * 100
    verdict = 'asciende' if check['promoted'] else 'se mantiene'
    return f"gen {entry['generation']} · mejora {gain:+.1f} pts (±{half:.1f}) · {verdict} · {format_duration(elapsed)}"


def write_json_atomic(path, payload, attempts=5):
    """Write a temp file, fsync it, then os.replace. Best effort: a reader holding the
    file open on Windows must never crash training."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        data = json.dumps(payload, indent=2)
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(attempts):
            try:
                os.replace(temporary, path)
                return True
            except PermissionError:
                time.sleep(0.02 * (attempt + 1))
    except OSError:
        pass
    return False


class ProgressReporter:
    def __init__(self, path=None, console=None, time_cap=None, temperature=None, promotions=0, wall=time.monotonic):
        self.path = Path(path) if path else None
        self.console = console
        self.time_cap = time_cap
        self.temperature = temperature
        self.promotions = promotions
        self.wall = wall
        self.window = deque(maxlen=RATE_WINDOW + 1)
        self.line_width = 0
        self.warned = False

    def round_done(self, *, elapsed, generation, round, matches, samples, updates, recent_loss,
                   league_mix, champion_generation, panel_version, ended=False, stop_reason=None):
        now = self.wall()
        self.window.append((now, matches, samples))
        first, last = self.window[0], self.window[-1]
        span = last[0] - first[0]
        matches_per_s = (last[1] - first[1]) / span if span > 0 else None
        samples_per_s = (last[2] - first[2]) / span if span > 0 else None
        seen = sum(league_mix.values())
        mix = {k: v / seen for k, v in league_mix.items()} if seen else {}
        if self.path is not None:
            ok = write_json_atomic(self.path, dict(
                elapsed=elapsed, time_cap=self.time_cap, generation=generation, round=round,
                matches=matches, samples=samples, updates=updates,
                matches_per_s=matches_per_s, samples_per_s=samples_per_s, recent_loss=recent_loss,
                league_mix=mix, temperature=self.temperature, champion_generation=champion_generation,
                promotions=self.promotions, panel_version=panel_version,
                status='ended' if ended else 'training', stop_reason=stop_reason))
            if not ok and not self.warned:
                self.warned = True
                print(f'aviso: no se pudo escribir {self.path}; el entrenamiento sigue sin archivo de progreso', file=sys.stderr)
        if self.console is not None:
            rate = f' · {matches_per_s:.1f} part/s' if matches_per_s else ''
            shown = generation if ended else generation + 1
            text = f'gen {shown} · ronda {round} · {matches} partidas{rate} · {format_duration(elapsed)}'
            self._write('\r' + text.ljust(self.line_width))
            self.line_width = len(text)

    def generation_done(self, entry, elapsed):
        if entry.get('promoted'):
            self.promotions += 1
        if self.console is not None:
            line = generation_line(entry, elapsed)
            self._write('\r' + line.ljust(self.line_width) + '\n')
            self.line_width = 0

    def _write(self, text):
        try:
            self.console.write(text)
            self.console.flush()
        except (OSError, ValueError):
            self.console = None
