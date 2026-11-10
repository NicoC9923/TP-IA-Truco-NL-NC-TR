"""Truco strength percentiles of 1, 2 and 3 cards (ADR 0062), from the rules module only.

The measure is truco_reglas.poder_mano_truco. The percentile of a set of k cards is the
mid-rank of its measure among all C(40, k) sets of k distinct cards: (below + equal / 2) / count.
The table is keyed by the cards' truco ranks, sorted from highest, and committed as
truco_strength_percentiles.json. Regenerate it with ``py truco_strength.py --write``.
"""
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
import truco_reglas as rules

TABLE_PATH = Path(__file__).with_name('truco_strength_percentiles.json')
_DECK = tuple((v, p) for p in rules.PALOS for v in rules.VALORES)
_RANK = {card: rules.valor_carta_truco(*card) for card in _DECK}


def _measure(cards):
    return rules.poder_mano_truco([{'valor': v, 'palo': p} for v, p in cards])


def _key(ranks):
    return ','.join(map(str, sorted(ranks, reverse=True)))


def generate():
    """{'1': {ranks: percentile}, '2': ..., '3': ...} computed from scratch."""
    table = {}
    for size in (1, 2, 3):
        hands = list(combinations(_DECK, size))
        measures = [_measure(hand) for hand in hands]
        counts = Counter(measures)
        order = sorted(counts)
        below, running = {}, 0
        for value in order:
            below[value] = running
            running += counts[value]
        entries = {}
        for hand, measure in zip(hands, measures):
            entries[_key(_RANK[c] for c in hand)] = (below[measure] + counts[measure] / 2) / len(hands)
        table[str(size)] = dict(sorted(entries.items()))
    return table


def _load():
    raw = json.loads(TABLE_PATH.read_text(encoding='utf-8'))
    return {int(size): {tuple(map(int, key.split(','))): value for key, value in entries.items()}
            for size, entries in raw.items()}


_TABLES = _load() if TABLE_PATH.exists() else {}


def percentile(cards):
    """Percentile in [0, 1] of 1-3 (value, suit) cards; 0.0 for no cards."""
    if not cards:
        return 0.0
    return _TABLES[len(cards)][tuple(sorted((_RANK[tuple(c)] for c in cards), reverse=True))]


if __name__ == '__main__':
    if sys.argv[1:] == ['--write']:
        TABLE_PATH.write_text(json.dumps(generate(), indent=0, sort_keys=True) + '\n', encoding='utf-8')
    else:
        print(__doc__)
