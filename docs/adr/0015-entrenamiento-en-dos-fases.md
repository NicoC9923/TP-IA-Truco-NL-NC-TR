# Entrenamiento en dos fases: neuroevolución y después aprendizaje por refuerzo

El objetivo es una IA fuerte, no solo una que sirva de demostración académica. Por eso el entrenamiento va en dos fases que usan la misma interfaz de política, el mismo motor, la misma liga de rivales y la misma página de evolución:

1. **Neuroevolución** (ADR 0001): rápida de tener andando, valida todo el andamiaje y es la técnica más clara para exponer.
2. **Aprendizaje por refuerzo profundo con autojuego**, con Deep Monte-Carlo al estilo de DouZero, en PyTorch sobre CPU. Usa el campeón de la fase 1 como un rival más.

Las dos curvas se comparan contra los mismos rivales de referencia.

## Opciones consideradas

- **MCCFR:** es lo más fuerte en teoría (converge a un equilibrio de Nash), pero solo sobre una versión simplificada del juego. Además no explota a los rivales débiles y, al ser una tabla, no admite el perfil del rival, así que se perdería el modo Entrenado adaptativo. Tampoco reutiliza la red.
- **PPO** en lugar de Deep Monte-Carlo: igual de válido, pero más delicado de ajustar. Deep Monte-Carlo ya probó ser estable en juegos de cartas entrenando en CPU.
- **Quedarse solo con neuroevolución:** su fuerza es media, porque un algoritmo genético no escala a redes grandes.
