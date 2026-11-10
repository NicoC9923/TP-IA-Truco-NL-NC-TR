"""Run 3 hand facts (ADR 0062): columns appended after the 1,628 original phase 2 inputs.

Offsets from the end of the original layout: 0 own envido total of the original three
cards / 33; 1 flush-pair bit (two or more cards of one suit); 2 truco strength percentile of
the three-card hand; 3 percentile of the cards still unplayed (0 when none); 4 points at
stake if an envido call is accepted / 30 and 5 the same for real envido (0 when the call
is not legal now; the pending envido chain is included); 6 (version 3 only) probability
that the own envido total beats the opponent's under the exact-filter beliefs, ties to mano.
"""
import numpy as np
import truco_reglas as rules
import truco_strength

STAKE_RAISES = (('envido', 2), ('real_envido', 3))


def _tantos(hand):
    return rules.calcular_envido([{'valor': v, 'palo': p} for v, p in hand])


def hand_facts(obs, legal_actions, exact_belief=None):
    """Six values, or seven when ``exact_belief`` is given."""
    hand = obs['hand']
    own = _tantos(hand)
    suits = [suit for _, suit in hand]
    flush = float(any(suits.count(s) >= 2 for s in suits))
    pending = obs['pending']
    accepted = pending['accepted'] if pending and pending['kind'] == 'envido' else 0
    values = [own / 33, flush, truco_strength.percentile(hand),
              truco_strength.percentile([hand[i] for i in obs['available']])]
    values.extend((accepted + points) / 30 if call in legal_actions else 0.0 for call, points in STAKE_RAISES)
    if exact_belief is not None:
        values.append(exact_belief.tanto_win_probability(own, ties_win=obs['mano'] == obs['player']))
    return np.asarray(values, dtype=np.float32)
