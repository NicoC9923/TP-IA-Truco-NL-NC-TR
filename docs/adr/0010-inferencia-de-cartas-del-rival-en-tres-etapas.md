# Inferencia de las cartas del rival en tres etapas

Deducir qué cartas tiene el rival se separa de decidir la jugada, y cada parte usa la técnica que le sirve:

1. **Módulo bayesiano exacto:** usa solo la evidencia segura (tantos declarados, "son buenas", cartas vistas) para descartar manos imposibles entre las 7.770 combinaciones.
2. **Red de inferencia supervisada:** ve el historial completo, incluidos los cantos, y predice las cartas del rival. Se entrena con las cartas reales de las partidas simuladas, así que las etiquetas salen gratis. Su predicción se filtra con el módulo bayesiano para que nunca proponga una carta imposible.
3. **Red de decisión:** recibe la inferencia como entrada.

Se descartó que la red de decisión aprenda a inferir sola desde el historial crudo: con neuroevolución eso exige redes mucho más grandes, y un algoritmo genético no escala a ese tamaño. Los cantos de truco no entran como evidencia segura, porque pueden ser faroles.
