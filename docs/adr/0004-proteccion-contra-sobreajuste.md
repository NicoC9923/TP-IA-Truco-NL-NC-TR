# Protección contra el sobreajuste en el autojuego

En autojuego puro aparecen dos problemas: ciclos de estrategia (A le gana a B, B le gana a C y C le gana a A) y la especialización en las manías del propio clon. Para evitarlos:

1. Cada individuo juega contra una mezcla de rivales: clones de su generación, el salón de la fama y los rivales de referencia.
2. El progreso se mide en repartos de validación que nunca se usan para entrenar. La curva que se muestra es esa, no el fitness de entrenamiento.
3. Cada reparto se juega en partidas espejo, cambiando de asiento, para compensar la suerte de las cartas.

## Opciones consideradas

- **Jugar solo contra la generación anterior:** es lo más simple, pero produce exactamente los ciclos y la especialización que se quieren evitar.
