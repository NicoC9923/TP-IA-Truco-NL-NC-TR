# A symmetric game engine, separate from the human match

Training uses a new game engine: symmetric (player A vs player B), with no UI, no robot and no threads, and reproducible from a seed. It reuses the rules in `truco_reglas.py`. `PartidaTruco` is not used for training because it is hard-wired to "human vs robot", embeds the AI, and is tied to the simulator, so it cannot run thousands of AI-vs-AI matches. The work happens on a separate branch (`feat/ia-entrenada`) so the demo already working on `main` is not put at risk.

## Consequences

There are two implementations of a hand's flow. A test checks that they score the same points on fixed hands, so what the agent learns in the engine also holds in the human match.
