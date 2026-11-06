# Protección contra el sobreajuste en el autojuego

En autojuego puro aparecen dos problemas: ciclos de estrategia (A le gana a B, B le gana a C y C le gana a A) y la especialización en las manías del propio clon. Para evitarlos, sin usar a los agentes que ya existen en la app (ADR 0021):

1. Cada individuo juega contra una liga hecha solo con redes de su propia fase: más o menos la mitad de las partidas contra **rivales sin sesgo** (clones de su generación y el salón de la fama) y la otra mitad contra **rivales con estilo**, que son copias de esas mismas redes con sesgos fijos (ADR 0013).
2. El salón de la fama guarda solo a los campeones que fueron promovidos a campeón vigente (ADR 0017), así que es una lista corta de mejoras reales y no cientos de casi-copias.
3. El progreso se mide en repartos de validación que nunca se usan para entrenar.
4. Cada reparto se juega en partidas espejo, cambiando de asiento, para compensar la suerte de las cartas.

Es lo que se hace comúnmente: OpenAI Five jugaba cerca del 80% de sus partidas contra su versión actual y el 20% contra versiones anteriores suyas, y AlphaStar entrenaba en una liga formada solo por versiones propias.

## Opciones consideradas

- **Jugar solo contra la generación anterior:** es lo más simple, pero produce exactamente los ciclos y la especialización que se quieren evitar.
- **Sumar a la liga a Reactivo, Utilidad adaptativa y cuatro variantes de Utilidad** (la versión original de este ADR): las anclas externas ayudan contra los ciclos, pero el agente nuevo aprendería a explotar a los agentes existentes, y su comparación en la app dejaría de ser entre arquitecturas independientes. Se descartó por la ADR 0021.
