# La red de inferencia supervisada se construye en la fase 2, como salida auxiliar

En la fase 1 la red de decisión usa solo el módulo bayesiano. La predicción supervisada de las cartas del rival (ADR 0010) se agrega en la fase 2, como una salida auxiliar de la misma red, y se entrena junto con la decisión. Hacerla en la fase 1 habría obligado a escribir el cálculo de gradientes a mano en numpy; en la fase 2, PyTorch ya lo trae. Además, entrenar las dos cosas juntas suele mejorar la decisión.
