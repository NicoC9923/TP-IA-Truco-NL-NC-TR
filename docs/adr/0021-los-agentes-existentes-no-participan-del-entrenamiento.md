# Los agentes existentes no participan del entrenamiento

Reactivo, Utilidad adaptativa y cualquier cosa construida con su código quedan fuera del entrenamiento del agente nuevo: no son rivales de la liga, no forman parte del fitness, no deciden cuándo termina una fase y no eligen qué campeón llega a la app. El entrenamiento solo usa redes de su propia fase (ADR 0004, 0013 y 0017).

El motivo es que los modos de la app sean arquitecturas independientes. Si el agente nuevo practicara contra los existentes, aprendería a explotarlos en particular, y compararlos en vivo dejaría de ser justo: uno habría estudiado al otro.

Los agentes existentes sí se usan **después**, como evaluación de referencia (ADR 0022): miden al campeón, pero el entrenador nunca lee ese resultado. Es la práctica habitual del área: AlphaZero entrenó solo contra sí mismo y su fuerza se midió después contra Stockfish, que nunca vio al entrenar; DouZero se comparó contra bots existentes que tampoco usó para entrenar; Libratus y Pluribus se midieron contra jugadores profesionales de póker.

Lo que el entrenamiento sí reutiliza es el módulo de reglas (jerarquía de cartas, tantos, Falta Envido), porque eso es el juego y no una estrategia.

## Opciones consideradas

- **Liga con Reactivo, Utilidad adaptativa y cuatro variantes de Utilidad** (el diseño original): anclas externas contra los ciclos de estrategia, pero con exactamente el acoplamiento que se quiere evitar.
- **Validar y terminar la fase contra Utilidad adaptativa:** no cambia los pesos de forma directa, pero decide cuándo parar y qué campeón se usa, así que mete al agente existente en el circuito.
