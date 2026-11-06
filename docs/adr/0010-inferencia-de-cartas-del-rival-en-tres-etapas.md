# Inferencia de las cartas del rival en tres etapas

Deducir qué cartas tiene el rival se separa de decidir la jugada, y cada parte usa la técnica que le sirve:

1. **Módulo bayesiano**, en dos capas que se mantienen separadas:
   - **Filtro exacto:** usa solo la evidencia segura (tantos declarados, "son buenas", cartas vistas) para descartar manos imposibles entre las 7.770 combinaciones. Una mano descartada no vuelve nunca.
   - **Ponderación:** a las manos que quedan les da más o menos peso según las señales de comportamiento del rival (por ejemplo, un canto de truco), en proporción a la tasa de farol que se le midió (ADR 0012). Si el rival casi no farolea, su truco pesa mucho; si farolea seguido, pesa poco. Con la adaptación apagada se usa una tasa neutra.
2. **Red de inferencia supervisada** (fase 2, ADR 0019): ve el historial completo y predice las cartas del rival. Se entrena con las cartas reales de las partidas simuladas, así que las etiquetas salen gratis. Su predicción pasa por el filtro exacto para que nunca proponga una carta imposible.
3. **Red de decisión:** recibe como entrada los resúmenes de la inferencia ponderada, calculados con el mismo código al entrenar y al jugar contra el humano.

Los cantos de truco siguen sin ser evidencia segura, porque pueden ser faroles: entran como evidencia ponderada y nunca descartan una mano.

## Opciones consideradas

- **Que la red de decisión aprenda a inferir sola desde el historial crudo:** con neuroevolución eso exige redes mucho más grandes, y un algoritmo genético no escala a ese tamaño.
- **Solo el filtro exacto** (la versión original): ignora lo que el agente ya sabe de cuánto farolea este rival en particular.
