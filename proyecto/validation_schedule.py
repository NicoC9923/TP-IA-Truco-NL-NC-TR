"""Shared validation deals for promotion and reference evaluation."""

def validation_schedule(seed, target, pairs):
    return [(2 * (seed * 1000000000 + p), target if target is not None else (15 if p % 2 == 0 else 30)) for p in range(pairs)]
