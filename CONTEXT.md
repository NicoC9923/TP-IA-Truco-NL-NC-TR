# Truco with Ottoman

A 1-vs-1 Argentine truco match (no flor, to 15 or 30 points, or a set number of hands) between a human and Ottoman, an AI agent, or between two AI agents, plus the training of agents that learn to play it.

Docs and code are in English; only the UI is in Spanish. Where a term appears in the UI, its Spanish label is given as **UI:**. Truco's own game vocabulary (truco, retruco, vale cuatro, envido, real envido, falta envido, quiero / no quiero, "son buenas", brava, pardas, tantos, mano / pie) has no English equivalent and is used as-is.

## Language

### Game

**Match**:
A complete game between two players, ending when one reaches the target of 15 or 30 points, or after a chosen number of hands (see **Short match**). UI: "partida".
_Avoid_: game, partido

**Short match**:
A match that lasts a chosen number of hands (1 to 10) instead of a points target. The player with more points after the last hand wins; if points are level, extra hands are played one at a time until they are not (ADR 0069). UI: "partida corta".
_Avoid_: quick game, demo match

**Spectator mode**:
A match between two AI agents that the user only watches, choosing which agent plays each side and controlling the pace. The user may see both players' cards and what each agent is considering. UI: "modo espectador".
_Avoid_: AI vs AI, demo mode, simulation

**Hand**:
One deal: from dealing the cards until its points are scored. UI: "mano".
_Avoid_: round, play

**Trick**:
Each of the up to three card-against-card contests inside a hand. UI: "baza".
_Avoid_: round, turn, truco (as a synonym for trick)

**Mano / pie**:
Each player's role in a hand: the mano plays first and wins ties in tantos.
_Avoid_: "es mano" to mean the deal

**Cards**:
The three cards a player receives in a hand. UI: "cartas".
_Avoid_: hand (for the cards)

**Bluff**:
A call or raise made with an objectively weak hand: fewer than 24 tantos for envido, or a hand-strength percentile below 0.44 for truco. The criterion is the same for every player, human or agent. UI: "mentira" (earlier called "farol", which remains in code identifiers such as `es_farol` and `prob_farol_envido`).
_Avoid_: lie (except in audience-facing text)

**Certain evidence**:
Information about the opponent's cards that cannot be false: declared tantos, a "son buenas", and cards already played.
_Avoid_: hint, confirmed data

**Behavioral signal**:
Information about the opponent that may be a bluff, such as calling, raising or accepting truco.
_Avoid_: tell

**Weighted evidence**:
A behavioral signal given more or less weight according to how often that opponent bluffs. It changes what is believed about their cards but never rules a hand out.
_Avoid_: probable evidence, weak evidence

### Agents

**Agent mode**:
Each AI architecture Ottoman can use in a match, chosen from the UI: Reactive, Adaptive Utility, Genetic-algorithms network and Reinforcement-learning network.
_Avoid_: profile, difficulty, level

**Reactive**:
The existing simple-reflex agent mode. UI: "Reactivo".

**Adaptive Utility**:
The existing agent mode built on thresholds over the 474,000-hand tables, with online adjustment. UI: "Utilidad adaptativa".
_Avoid_: Learning mode, "aprendizaje" (its former name)

**Existing agent**:
Each agent that was already in the app before the trained agent: Reactive and Adaptive Utility. It takes no part in phase 1 or phase 2 run 1 training; from phase 2 run 2 on it trains with the model and sits on its selection panel (ADR 0056).
_Avoid_: legacy agent, old agent

**Trained agent**:
An agent whose policy was not hand-written: it comes from an iterative training process. There is one per phase, and each one is a network mode.
_Avoid_: new AI, trained bot, "Entrenado" (as a mode name)

**Network mode**:
Each of the two agent modes that play with a trained agent: the Genetic-algorithms network (phase 1, UI: "Red neuronal · Algoritmos genéticos") and the Reinforcement-learning network (phase 2, UI: "Red neuronal · Aprendizaje por refuerzo").
_Avoid_: trained mode, AI mode

**Adapt to opponent**:
A network-mode option. When on (the default), the agent uses the opponent profile and weighted evidence; when off, it uses neutral values for both but still reads the current hand. UI: "Adaptarse al rival", in advanced settings.
_Avoid_: adaptive mode, "Entrenado adaptativo"

**Reasoning trace**:
An audience-visible record of why the agent made each decision in a hand.
_Avoid_: log, XAI, thoughts

**Opponent profile**:
Statistics on how a specific opponent plays (bluffs seen, folds, truco accepted, envido called), measured from what is observed of them, starting from a neutral value and accumulated across matches.
_Avoid_: human model, style (for the observed profile)

**Style**:
A fixed way of playing imposed on a styled opponent: bluffer, cautious, aggressive or calculating.
_Avoid_: personality, profile (for the style)

### Training

**Phase**:
Each stage of training, with its own technique: phase 1, genetic algorithms; phase 2, reinforcement learning. Phase 1 never uses phase 2. Phase 2 may train against selected phase 1 networks and uses some in its selection panel (ADR 0043, 0047). Phase 1 and phase 2 run 1 never trained against a reference opponent; phase 2 run 2 does (ADR 0056).
_Avoid_: stage, version

**Self-play**:
Training in which the agent plays against copies of itself or earlier versions of itself.
_Avoid_: playing against a clone

**League**:
The set of opponents an agent plays while training. Phase 1: clones, the hall of fame and styled opponents, all from phase 1. Phase 2: its current self, the snapshot pool (with selected phase 1 networks) and styled snapshots.
_Avoid_: pool, tournament

**Styled opponent**:
A copy of a network with a fixed bias added to its outputs (in phase 2, to its action values) so it plays in a style.
_Avoid_: sparring partner, style bot, reference opponent (for these)

**Generation**:
A full population of individuals evaluated at once; the evaluation yields the parents of the next generation. In phase 2, which has no population, each promotion check after a fixed number of training matches is called a generation.
_Avoid_: epoch, iteration

**Individual**:
A candidate in the population, defined by its genome (its network's weights).
_Avoid_: agent (for a candidate), bot

**Genome**:
The vector of all of an individual's network weights; what the genetic algorithm crosses over and mutates.
_Avoid_: chromosome, DNA

**Fitness**:
An individual's score in a generation, from the matches it wins and its point difference.
_Avoid_: score, reward

**Champion**:
The individual with the best fitness in a generation. In phase 2, the network at the end of a generation, which challenges the reigning champion.
_Avoid_: best bot, winner

**Reigning champion**:
The best agent of a phase so far. A champion replaces it only by beating it clearly (phase 1: 55% against it; phase 2: a measured improvement over it against the selection panel); that replacement is a **promotion**. It is the one that plays in its phase's network mode.
_Avoid_: best champion, active champion

**Hall of fame**:
The set of champions that were promoted to reigning champion, which new generations keep playing against. Phase 1 only; phase 2 uses a snapshot pool.
_Avoid_: archive

**Snapshot pool**:
Phase 2's frozen past versions of itself, one saved every generation whether promoted or not, kept bounded to recent and behaviorally distinct ones, plus selected phase 1 networks. Training opponents only.
_Avoid_: hall of fame (for phase 2), archive

**Selection panel**:
A fixed, versioned set of frozen opponents against which phase 2's challenger and reigning champion are both measured in each promotion check. Its network members are never training opponents. Run 1 used networks only; run 2 adds Reactive and Adaptive Utility (ADR 0057).
_Avoid_: evaluation panel, reference opponents

**Measuring run**:
A short phase 2 run, before the real one, that measures speed and fixes pilot settings; its last snapshot joins the selection panel.
_Avoid_: tuning run, smoke test

**Current-hand events**:
The ordered public actions of the hand in progress (calls, answers, declarations, cards played), as the phase 2 network sees them.
_Avoid_: history (alone), trace

**Reference opponent**:
An existing agent (Reactive or Adaptive Utility) the trained agent is measured against in the reference evaluation. Not a league opponent in phase 1 or phase 2 run 1; from run 2 on it is one, so its scores are familiar-opponent results (ADR 0060).
_Avoid_: benchmark, sparring partner

**Reference evaluation**:
Measuring champions against the reference opponents, done separately from training. Its result is shown but training never reads it.
_Avoid_: benchmark, test, validation (for this measurement)

**Held-out networks**:
The two phase 1 champions (v1s1-g23 and v1s2-g2) never used in any league or panel, kept out of phase 2 run 2 entirely so they remain the only unseen check (ADR 0060).
_Avoid_: test set, exam (alone)

**Acceptance test**:
The one-time, pre-registered test that decides whether a new phase 2 champion replaces the shipped one: a direct match plus the held-out networks (ADR 0058).
_Avoid_: final evaluation, unbiased estimate

**Validation deals**:
A fixed set of deals never used for training, on which promotions are decided and the reference evaluation is run.
_Avoid_: test set, evaluation seeds

**Mirror matches**:
A pair of matches on the same deals in which the players swap seats, to cancel out the luck of the cards.
_Avoid_: duplicate, reversed match

**Observation**:
The information available to a player at one decision, including that player's cards and the public match history, but not the opponent's hidden cards.

**Legal action**:
An action permitted by the game state at a decision; a policy chooses from these actions.

**Revealed hand**:
An opponent's original cards are all publicly known; the engine records this when all three have been played. Only revealed hands contribute to the observed bluff rate.

**Next-trick win probability**:
The weighted probability that a specified card wins against a uniformly selected remaining opponent card, with the relevant tie rule supplied by the caller.


**Population diversity**:
The mean standard deviation of genome weights across individuals in a generation.

**Reference tournament**:
A standalone round-robin comparison of reference opponents and optional policies using mirror match pairs; its bluff rate is the share of calls or raises marked as bluffs by the engine.



**Patience extension**:
An explicitly requested increase of the remaining promotion patience when resuming a training checkpoint, recorded in the generation log.

**Evaluation panel**:
A frozen set of network opponents used to compare a candidate and a baseline across multiple strategies after training.
_Avoid_: league, reference opponents

**Pre-registered experiment decision**:
The decision about whether an experiment is promising, using comparison criteria fixed before its outcome is known.
_Avoid_: promotion, shipping decision

**Offspring diagnostic**:
A frozen-population comparison of offspring with their parents against a shared set of opponents.

**Decision observation bank**:
A fixed collection of legal decision situations used to compare policy action distributions.

**Behavioral diversity**:
Differences in action probability distributions between policies on a shared decision observation bank.

**Encoder version**:
The agreed interpretation of a network's observation features; a trained network is evaluated with the version used during training.

**Truco resolution**:
A response that accepts, declines, folds against, or raises a pending truco call. An envido interruption postpones the response and is not a resolution.

**Current-trick card outcome**:
The signed result of playing an available card against the opposing card in the current trick, or its signed expectation when that opposing card has not yet been played. A parda has value zero.

**Pending envido stakes**:
The points payable if a pending envido is accepted or declined, including the winner-dependent falta envido amount.

**Hand facts**:
Explicit inputs of the run 3 phase 2 encoder (versions 2 and 3, ADR 0062): own envido total of the original three cards, flush-pair bit, truco strength percentile of the hand and of the unplayed cards, envido and real envido stakes, and (version 3) the probability that the own tanto beats the opponent's.

**Match equity**:
The probability that a player wins the match from a score, a target (15 or 30) and who is mano of the next hand, read from a fixed table built from greedy self-play. It is the second half of the run 3 training target, read after the decision's hand is scored.

**Warm start**:
Starting a training run from an existing network. The input layer is widened with zero-filled columns for new inputs, so the starting network plays exactly like the old one.

## Confirmed training-engine rule decisions

- Truco and raises may be called only on the caller's own card turn, or as an answer to a pending bet. Calling between card turns, which the human game allows, is not modeled.
- "El envido va primero" is allowed as an answer to the first truco, even after the answering player has played their first card, as long as no trick is complete. This matches the human game.

These are deliberate training-engine simplifications; see ADR 0055.
