# Entrenamiento en dos fases independientes: neuroevolución y aprendizaje por refuerzo

Las dos fases entrenan una red neuronal; lo que cambia es cómo se ajustan sus pesos. El objetivo es una IA fuerte, no solo una demostración académica, así que el entrenamiento tiene dos fases que usan la misma interfaz de política, el mismo motor, las mismas reglas de liga, el mismo criterio de fin de fase y la misma página de evolución:

1. **Neuroevolución** (ADR 0001): un algoritmo genético ajusta los pesos. Está en el programa de la materia, produce generaciones que se pueden mostrar, es rápida de tener andando y prueba todo el andamiaje. Su techo es medio, porque un algoritmo genético no escala a redes grandes.
2. **Aprendizaje por refuerzo profundo con autojuego:** Deep Monte-Carlo al estilo de DouZero, en PyTorch sobre CPU. Ajusta los pesos por descenso por gradiente, que es la forma habitual de entrenar redes neuronales.

Las dos fases son **corridas independientes**. La fase 2 arranca con pesos al azar y su liga se arma solo con sus propias redes (ADR 0004 y 0013): nunca juega contra el campeón de la fase 1. Las dos se comparan recién al final, en la evaluación de referencia (ADR 0022) y en la página de evolución. Así la comparación entre el algoritmo genético y el aprendizaje por refuerzo es limpia: lo único distinto es cómo aprende cada red.

Es como se hace comúnmente: DouZero y AlphaZero entrenaron con autojuego puro desde cero, sin rivales de otro método.

## Opciones consideradas

- **Usar el campeón de la fase 1 como rival de la fase 2** (la versión original): la fase 2 arrancaría contra un rival fuerte, pero aprendería a ganarle a la fase 1 en particular y la comparación entre fases quedaría sesgada.
- **MCCFR:** es lo más fuerte en teoría (converge a un equilibrio de Nash), pero solo sobre una versión simplificada del juego. Además no explota a los rivales débiles y, al ser una tabla, no admite el perfil del rival, así que se perdería la adaptación al rival. Tampoco reutiliza la red.
- **PPO** en lugar de Deep Monte-Carlo: igual de válido, pero más delicado de ajustar. Deep Monte-Carlo ya probó ser estable en juegos de cartas entrenando en CPU.
- **Quedarse solo con una de las dos fases:** solo neuroevolución tiene fuerza media; solo aprendizaje por refuerzo pierde el algoritmo genético de la materia y las generaciones para exponer.
