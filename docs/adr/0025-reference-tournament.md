# Reference tournament statistics and fixed state

Adaptive Utility uses the agent's default parameter values and an in-memory history in reference matches. It neither reads nor writes the human history; its normal human-match construction is unchanged. Policy factories receive a seed, and a fresh pair is constructed for each pair of mirror matches.

The standalone tournament reports the fraction of individual matches won by each policy and a 95% normal margin based on the sample standard error of mirror-pair win rates, consistent with promotion and reference evaluation. The interval is centered on the observed rate and can extend beyond zero or one. With one pair the sample margin is reported as zero because variance cannot be estimated. A policy with no calls has an undefined bluff rate, represented by null.

Wilson intervals over individual matches were rejected because paired matches share deals and the resulting margin differs from the promotion check. Persisting or copying the human history was rejected because a reference opponent must not change with human play. The tournament remains separate from training as required by ADR 0021.
