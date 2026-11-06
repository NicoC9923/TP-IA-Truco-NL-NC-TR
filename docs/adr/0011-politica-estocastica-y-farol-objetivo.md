# Política estocástica y detección objetiva del farol

La red elige su acción al azar, en proporción a sus puntajes (softmax), tanto al entrenar como al jugar. Así puede aprender faroles con frecuencia mixta ("mentir el 25% de las veces con esta mano"). Una política determinista sería predecible y explotable.

Como una red no tiene una "intención" de mentir que se pueda inspeccionar, el farol se juzga desde afuera, con el mismo criterio para todos los modos: un canto o una subida de envido con menos de 24 tantos, o de truco con una fuerza de mano (el percentil del corpus) menor a 0,44. Son los mismos umbrales que el juego ya usa para juzgar los faroles del humano, así que las tasas de farol de todos los modos y del humano se pueden comparar.
