# Neuroevolución para el agente entrenado

Es la fase 1 del plan de entrenamiento en dos fases (ADR 0015).

El agente entrenado es una red neuronal chica (perceptrón multicapa) cuyos pesos se ajustan con un algoritmo genético a través de generaciones de autojuego. Se eligió porque cubre dos técnicas del programa de la materia a la vez (algoritmos genéticos y redes neuronales), produce generaciones reales que se pueden mostrar en una exposición, y no necesita ejemplos de "jugada correcta": aprende solo del resultado de las partidas, así que puede descubrir faroles que nadie programó.

## Opciones consideradas

- **Algoritmo genético sobre los umbrales del agente de utilidad:** simple, pero solo afina reglas escritas a mano. El techo es bajo.
- **Aprendizaje supervisado:** necesita ejemplos etiquetados con la jugada correcta, y no existen. El dataset T1K22 tiene repartos, no decisiones. Si se etiquetara con las decisiones del agente actual, la red solo aprendería a imitarlo y nunca lo superaría.
- **MCCFR:** es lo más sólido en teoría para juegos con faroles, pero avanza por iteraciones y no por generaciones, necesita abstraer el estado del juego y queda menos alineado con el programa.
- **Aprendizaje por refuerzo (PPO, DQN) con PyTorch:** potente, pero mucho más difícil de poner a punto y suma una dependencia pesada (ver ADR 0003).
