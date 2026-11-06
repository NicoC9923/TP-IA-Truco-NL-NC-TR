# La evaluación de referencia corre en un proceso aparte

La fuerza del agente nuevo se mide contra los agentes existentes, Reactivo y Utilidad adaptativa, en un **proceso aparte del entrenador**. Ese proceso toma cada campeón guardado, lo hace jugar 1.000 pares de partidas espejo contra cada rival de referencia sobre los repartos de validación y escribe su propio log, con el porcentaje de victorias y su margen de error. La página de evolución junta ese log con el del entrenador (ADR 0009).

La separación queda garantizada por la estructura y no por disciplina: el entrenador no tiene ningún camino de código hacia los agentes existentes, y un test verifica que el módulo del entrenador no los importa. Para la exposición: "el examen se corrige en otra sala".

Para que la vara no se mueva, en cada corrida Utilidad adaptativa arranca desde el mismo estado inicial fijo, sin lo aprendido contra humanos, y nunca escribe su historial persistente. Así una medición de una semana se puede comparar con la de la semana siguiente, y las partidas de evaluación no contaminan lo que aprendió de los humanos.

## Opciones consideradas

- **Evaluar dentro del entrenador con la regla de "no leer el resultado":** funciona, pero depende de que nadie la rompa; con un proceso aparte no se puede romper sin que lo detecte el test.
- **Medir la fuerza solo contra versiones propias:** sirve para decidir promociones (ADR 0017), pero los resultados entre versiones propias pueden dar vueltas en círculo y no prueban que el agente sea fuerte.
- **Medir contra humanos:** harían falta unas 2.000 partidas para tener ±2 puntos de margen.
