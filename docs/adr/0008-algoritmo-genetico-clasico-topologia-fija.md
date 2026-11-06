# Algoritmo genético clásico con topología de red fija

El genoma de cada individuo es el vector de todos los pesos de una red de forma fija. Cada generación aplica selección por torneo, elitismo (los mejores pasan sin cambios), cruce de pesos y mutación gaussiana, sobre una población de unos 64 individuos. Se eligió por ser el algoritmo genético del programa de la materia y por ser fácil de explicar y de visualizar.

## Opciones consideradas

- **NEAT** (también evoluciona la forma de la red): muy vistoso, pero mucho más complejo de implementar. Queda como posible extensión.
- **Estrategias evolutivas** (CMA-ES, OpenAI-ES): suelen converger mejor, pero se alejan del algoritmo genético visto en la materia.
