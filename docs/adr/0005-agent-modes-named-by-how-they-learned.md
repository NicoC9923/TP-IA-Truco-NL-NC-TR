# Four agent modes, named by how each one learned

The static "utilidad" mode is removed, because it was the "aprendizaje" mode with its adjustments switched off, and the two are merged into a single **Adaptive Utility** mode. The UI keeps four architectures to compare live (UI labels in Spanish):

- **Reactive** (UI: "Reactivo"): simple-reflex agent, the comparison floor.
- **Adaptive Utility** (UI: "Utilidad adaptativa"): thresholds over the 474,000-hand tables, with bluffing and opponent modelling.
- **Neuroevolution network** (UI: "Red neuronal · Neuroevolución"): the phase 1 network, trained with a genetic algorithm.
- **Reinforcement-learning network** (UI: "Red neuronal · Aprendizaje por refuerzo"): the phase 2 network, trained with Deep Monte-Carlo.

The names of the two new modes say how each network learned, so the genetic-algorithm vs reinforcement-learning comparison is visible right in the selector. Next to the two network modes there is an **Adapt to opponent** switch (UI: "Adaptarse al rival") and a button to reset the opponent profile (ADR 0012). Off, the network plays with neutral values; on, it uses the human's profile. Comparing the two positions shows exactly what adapting adds. Each network mode shows as unavailable while its phase has no saved reigning champion.

## Considered Options

- **"Entrenado" and "Entrenado adaptativo"** (the original version): they don't say which technique trained the network, and with two phases the champion would have to be chosen somewhere else.
- **Six separate modes** (each network, adaptive and not, plus the two existing ones): clutters the selector and hides that the adaptive and neutral versions are the same network.
