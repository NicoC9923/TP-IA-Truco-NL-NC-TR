# Adaptación al rival a través de la entrada y de la inferencia, no reentrenando

Los pesos de la red no cambian mientras juega contra un humano: entrenar necesita miles de partidas y un humano juega pocas. El agente se adapta leyendo al rival, de dos maneras:

1. **Perfil del rival como entrada:** tasa de faroles vistos, de idas al mazo, de truco aceptado y de envido cantado. Cambia **cómo juega** contra ese rival.
2. **Inferencia ponderada** (ADR 0010): el módulo bayesiano pesa los cantos de truco del rival según su tasa de farol medida. Cambia **lo que cree** de sus cartas.

Reglas del perfil:

- Se calcula a partir de lo observado, con el mismo código al entrenar y al jugar. A la red nunca se le dice el estilo verdadero de un rival, porque contra el humano nunca lo va a saber.
- Cada estadística arranca en un valor neutro, con un prior que vale unas 20 observaciones, y se acerca a lo observado a medida que se junta evidencia. Así el agente no reacciona de más a las primeras manos.
- La tasa de farol se cuenta solo en las manos en las que se vieron las cartas del rival, con el criterio objetivo de farol (ADR 0011).
- Hay un único perfil del humano, guardado entre sesiones, con un botón para reiniciarlo cuando juega otra persona. La app no tiene cuentas de usuario.
- Con el interruptor **Adaptarse al rival** apagado (ADR 0005), las dos maneras usan valores neutros, así que las dos posiciones solo difieren en la adaptación.

Para que la red aprenda a leer el perfil, en el entrenamiento enfrenta rivales con estilos distintos (ADR 0013).

## Opciones consideradas

- **Reentrenar con las partidas del humano:** son muy pocas partidas y el resultado es inestable.
- **Cartera de campeones** (elegir durante la sesión el que mejor le va contra el humano): necesita varios campeones bien distintos, y el cambio recién rinde después de muchas partidas.
- **Solo el perfil como entrada, o solo la inferencia ponderada:** cada una muestra la adaptación desde un solo ángulo. Juntas muestran que cambia cómo juega y también lo que cree.
