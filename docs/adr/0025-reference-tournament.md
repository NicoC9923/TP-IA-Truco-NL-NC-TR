# Reference tournament statistics and fixed state

Adaptive Utility uses the agent's default parameter values and an in-memory history in reference matches. It neither reads nor writes the human history; its normal human-match construction is unchanged. Policy factories receive a seed, and a fresh pair is constructed for each pair of mirror matches.

The standalone tournament reports the fraction of individual matches won by each policy, a 95% Wilson interval and its half-width, and the fraction of calls or raises marked as bluffs by the engine. A policy with no calls has an undefined bluff rate, represented by null. Wilson intervals are a common binomial proportion interval and remain informative at zero or all wins; mirror matches share deals, so the interval is an approximate descriptive margin, not a guarantee of independent observations.

Using a normal interval was rejected because it collapses to zero width at extreme rates. Persisting or copying the human history was rejected because a reference opponent must not change with human play. The tournament remains separate from training as required by ADR 0021.
