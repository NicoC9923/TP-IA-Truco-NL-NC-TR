# Entradas de la red de decisión

La red de decisión ve seis grupos de datos, codificados como números (unas 60 entradas):

1. **Sus cartas:** jerarquía, palo, si ya la jugó y tantos de cada una.
2. **La mesa:** cartas jugadas en cada baza, quién ganó cada una, pardas y en qué baza está.
3. **Los cantos:** estado del envido (qué se cantó, quién, si está resuelto, los tantos declarados) y del truco (nivel y quién tiene el quiero).
4. **El tanteador.**
5. **La inferencia sobre el rival:** probabilidad de que tenga una brava, fuerza esperada de sus cartas y probabilidad de ganar la próxima baza.
6. **El perfil del rival:** en neutro para el modo Entrenado.

Se le dan resúmenes ya calculados (inferencia, tanteador) en lugar del historial crudo, porque una red chica evolucionada no puede deducirlos sola (ver ADR 0010).
