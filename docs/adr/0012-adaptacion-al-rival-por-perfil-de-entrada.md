# Adaptación al rival a través de la entrada, no reentrenando

El agente entrenado no cambia sus pesos mientras juega contra un humano: un algoritmo genético necesita miles de partidas y un humano juega pocas. En cambio, la red recibe como entrada un perfil del rival (tasa de faroles vistos, de idas al mazo, de truco aceptado, acumuladas entre partidas). Durante el entrenamiento enfrenta rivales de estilos variados, así que aprende a leer ese perfil. Contra el humano los pesos quedan fijos pero el perfil cambia, y con él el juego del agente.
