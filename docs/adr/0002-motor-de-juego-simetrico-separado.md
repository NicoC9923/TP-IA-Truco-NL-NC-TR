# Motor de juego simétrico, separado de la partida contra humanos

El entrenamiento usa un motor de juego nuevo: simétrico (jugador A contra jugador B), sin interfaz, sin robot ni hilos, y reproducible con semilla. Reutiliza las reglas de `truco_reglas.py`. `PartidaTruco` no se usa para entrenar porque está fijada a "humano contra robot", lleva la IA adentro y está atada al simulador, así que no sirve para correr miles de partidas IA contra IA. El trabajo se hace en una branch aparte (`feat/ia-entrenada`) para no poner en riesgo la demo que ya funciona en `main`.

## Consecuencias

Hay dos implementaciones del flujo de una mano. Un test verifica que dan los mismos puntos en manos fijas, para que lo que el agente aprende en el motor valga también en la partida contra el humano.
