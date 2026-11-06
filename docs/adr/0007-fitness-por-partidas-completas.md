# Fitness = partidas completas a 15 ganadas, más la diferencia de puntos

Cada individuo juega partidas completas a 15 puntos. Su fitness es el porcentaje de partidas ganadas más 0,1 × la diferencia de puntos normalizada. Las partidas completas son necesarias para que aprenda a jugar según el tanteador (cuándo conviene la Falta Envido, cuándo arriesgar al final de la partida). El término de diferencia de puntos desempata y reduce el ruido.

## Opciones consideradas

- **Solo diferencia de puntos por mano:** tiene menos ruido, pero nunca aprende a jugar según el tanteador.
