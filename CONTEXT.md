# Truco with Ottoman

A 1-vs-1 Argentine truco match (no flor, to 15 or 30 points) between a human and Ottoman, an AI agent, plus the training of agents that learn to play it.

Docs and code are in English; only the UI is in Spanish. Where a term appears in the UI, its Spanish label is given as **UI:**. Truco's own game vocabulary (truco, retruco, vale cuatro, envido, real envido, falta envido, quiero / no quiero, "son buenas", brava, pardas, tantos, mano / pie) has no English equivalent and is used as-is.

## Language

### Game

**Match**:
A complete game between two players, ending when one reaches the target of 15 or 30 points. UI: "partida".
_Avoid_: game, partido

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
A call or raise made with an objectively weak hand: fewer than 24 tantos for envido, or a hand-strength percentile below 0.44 for truco. The criterion is the same for every player, human or agent. UI: "farol".
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
Each AI architecture Ottoman can use in a match, chosen from the UI: Reactive, Adaptive Utility, Neuroevolution network and Reinforcement-learning network.
_Avoid_: profile, difficulty, level

**Reactive**:
The existing simple-reflex agent mode. UI: "Reactivo".

**Adaptive Utility**:
The existing agent mode built on thresholds over the 474,000-hand tables, with online adjustment. UI: "Utilidad adaptativa".
_Avoid_: Learning mode, "aprendizaje" (its former name)

**Existing agent**:
Each agent that was already in the app before the trained agent: Reactive and Adaptive Utility. It takes no part in training the trained agent.
_Avoid_: legacy agent, old agent

**Trained agent**:
An agent whose policy was not hand-written: it comes from an iterative training process. There is one per phase, and each one is a network mode.
_Avoid_: new AI, trained bot, "Entrenado" (as a mode name)

**Network mode**:
Each of the two agent modes that play with a trained agent: the Neuroevolution network (phase 1, UI: "Red neuronal · Neuroevolución") and the Reinforcement-learning network (phase 2, UI: "Red neuronal · Aprendizaje por refuerzo").
_Avoid_: trained mode, AI mode

**Adapt to opponent**:
A network-mode option. When on, the agent uses the opponent profile and weighted evidence; when off, it uses neutral values for both. UI: "Adaptarse al rival".
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
Each stage of training, with its own technique: phase 1, neuroevolution; phase 2, reinforcement learning. The two phases are independent: neither is used to train the other.
_Avoid_: stage, version

**Self-play**:
Training in which the agent plays against copies of itself or earlier versions of itself.
_Avoid_: playing against a clone

**League**:
The set of opponents an agent plays while training: clones, the hall of fame and styled opponents, all from its own phase.
_Avoid_: pool, tournament

**Styled opponent**:
A copy of a network from the same phase with a fixed bias added to its outputs so it plays in a style.
_Avoid_: sparring partner, style bot, reference opponent (for these)

**Generation**:
A full population of individuals evaluated at once; the evaluation yields the parents of the next generation. In phase 2, which has no population, each periodic champion check is called a generation.
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
The individual with the best fitness in a generation.
_Avoid_: best bot, winner

**Reigning champion**:
The best agent of a phase so far. A champion replaces it only by beating it clearly; that replacement is a **promotion**. It is the one that plays in its phase's network mode.
_Avoid_: best champion, active champion

**Hall of fame**:
The set of champions that were promoted to reigning champion, which new generations keep playing against.
_Avoid_: archive

**Reference opponent**:
An existing agent (Reactive or Adaptive Utility) the trained agent is measured against in the reference evaluation. Never an opponent in the league.
_Avoid_: benchmark, sparring partner

**Reference evaluation**:
Measuring champions against the reference opponents, done separately from training. Its result is shown but training never reads it.
_Avoid_: benchmark, test, validation (for this measurement)

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
