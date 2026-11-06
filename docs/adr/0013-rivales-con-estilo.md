# Rivales con estilo: copias del propio agente con sesgos fijos

Para que la red aprenda a leer el perfil del rival (ADR 0012), en el entrenamiento tiene que enfrentar rivales que jueguen distinto. Esos rivales son **copias de las propias redes de la fase** (clones o miembros del salón de la fama) a las que se les suma un sesgo fijo en sus salidas mientras hacen de rival:

- **Mentiroso:** bonificación a cantar truco y envido con cartas débiles.
- **Cauteloso:** penalización al "quiero".
- **Agresivo:** bonificación a subir.
- **Calculador:** penalización a farolear.

El tamaño de cada sesgo es fijo y queda documentado, para que las corridas se puedan comparar. Alrededor de la mitad de las partidas de entrenamiento son contra rivales con estilo, un 12,5% contra cada uno (ADR 0004). No se escribe ningún código de estrategia para esto ni se usa nada de los agentes existentes (ADR 0021).

## Opciones consideradas

- **Cuatro variantes del agente de Utilidad con parámetros fijos** (la versión original): el agente nuevo aprendería a explotar la lógica de Utilidad por la puerta de atrás, y su evaluación contra Utilidad adaptativa quedaría inflada.
- **Cuatro bots nuevos escritos desde cero:** independientes de la app, pero es estrategia escrita a mano que hay que mantener, y los sesgos sobre las propias redes alcanzan para dar variedad.
- **Sin estilos (autojuego puro):** es lo estándar cuando solo se busca fuerza, pero los clones juegan parecido, el perfil no tendría señal y la adaptación al rival casi no aprendería nada.
