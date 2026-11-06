# Una sola red decide todas las acciones, con máscara de acciones legales

Una única red tiene una salida por cada acción posible: tirar la carta 1, 2 o 3; envido, real envido y falta envido; quiero y no quiero; truco, retruco y vale cuatro; e irse al mazo. En cada decisión se descartan las acciones ilegales y se elige entre las que quedan. Se prefirió a tres redes separadas (envido, truco y cartas) porque comparte lo aprendido entre decisiones (por ejemplo, la fuerza de la mano sirve tanto para cantar truco como para elegir carta) y no triplica la cantidad de pesos a evolucionar.
