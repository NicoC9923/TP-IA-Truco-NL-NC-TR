# Campeón vigente y fin de fase por autojuego

El progreso y el fin de una fase se deciden solo con redes de la propia fase, sin los agentes existentes (ADR 0021):

- Hay un **campeón vigente**, que al principio es el primer campeón.
- Al final de cada generación, su campeón juega 1.000 pares de partidas espejo contra el campeón vigente, sobre los repartos de validación (2.000 partidas). Si gana al menos el 55%, es **promovido**: pasa a ser el campeón vigente y entra al salón de la fama.
- La fase **termina** cuando pasan 20 generaciones seguidas sin promociones. El número es un punto de partida, a ajustar cuando se mida la velocidad del motor.
- Los modos de red de la app cargan el campeón vigente de su fase.

Con 1.000 pares, el margen de error es de unos ±2 puntos porcentuales, así que un 55% indica una mejora real y no suerte de las cartas. La fase 2 no tiene generaciones: hace el mismo control cada cierta cantidad fija de partidas de entrenamiento, y cada control cuenta como una generación en la página de evolución.

Es el método de AlphaGo Zero: una red nueva reemplazaba a la mejor solo si le ganaba al menos el 55% de 400 partidas.

## Opciones consideradas

- **Terminar cuando el campeón le gane al 55% a Utilidad adaptativa** (la versión original): mete a un agente existente en el circuito del entrenamiento, porque decide cuándo parar y qué campeón llega a la app. Se descartó por la ADR 0021. Utilidad adaptativa pasa a la evaluación de referencia (ADR 0022).
- **Medir sobre pocas partidas (por ejemplo, 50):** no permite distinguir habilidad de azar.
- **Promover al campeón de cada generación sin control:** el campeón de una generación puede ser peor que uno anterior y el salón de la fama se llenaría de casi-copias.
