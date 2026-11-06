# Fase 1 con numpy en CPU, sin PyTorch ni GPU

Vale para la fase 1 (neuroevolución). La fase 2 (ADR 0015) usa PyTorch, también en CPU, porque el aprendizaje por refuerzo sí necesita gradientes.

La red y el algoritmo genético se implementan con numpy (que ya es dependencia del proyecto). Las generaciones se evalúan en paralelo con `multiprocessing` sobre los 16 hilos del CPU. El cuello de botella es simular el juego en Python, lógica con muchas ramas que una GPU no acelera; la red es chica (pocos miles de pesos), así que mandar datos a la GPU tardaría más de lo que ahorra. La ventaja principal de PyTorch, el cálculo automático de gradientes, no se usa en un algoritmo genético. Además, PyTorch con GPU AMD en Windows depende de ROCm, cuyo soporte es reciente y frágil.

## Consecuencias

La política del agente se define detrás de una interfaz propia, para poder reemplazarla más adelante por una red de PyTorch (por ejemplo, para aprendizaje por refuerzo) sin tocar el motor de juego.
