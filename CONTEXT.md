# Truco con Ottoman

Partida de truco argentino 1 contra 1 (sin flor, a 15 o 30 puntos) entre un humano y Ottoman, un agente de IA, y el entrenamiento de agentes que aprenden a jugarla.

## Lenguaje

### Juego

**Partida**:
Enfrentamiento completo entre dos jugadores, que termina cuando uno llega a la meta de 15 o 30 puntos.
_Evitar_: partido, juego, match

**Mano**:
Un reparto: desde que se dan las cartas hasta que se anotan sus puntos.
_Evitar_: ronda, jugada, hand

**Baza**:
Cada uno de los hasta tres enfrentamientos de una carta contra otra dentro de una mano.
_Evitar_: ronda, vuelta, truco (como sinónimo de baza)

**Ser mano / ser pie**:
Rol de cada jugador en una mano: el que es mano juega primero y gana los empates de tantos.
_Evitar_: "es mano" para referirse al reparto

**Cartas** (o **manojo**):
Las tres cartas que recibe un jugador en una mano.
_Evitar_: mano (para las cartas)

**Farol**:
Canto o subida hecho con una mano objetivamente débil: menos de 24 tantos en el envido, o una fuerza de mano menor al percentil 0,44 en el truco. El criterio es el mismo para cualquier jugador, humano o agente.
_Evitar_: bluff, mentira (salvo en textos para el público)

**Evidencia segura**:
Información sobre las cartas del rival que no puede ser falsa: los tantos declarados, un "son buenas" y las cartas ya jugadas.
_Evitar_: pista, dato confirmado

**Señal de comportamiento**:
Información sobre el rival que puede ser un farol, como cantar, subir o aceptar el truco.
_Evitar_: tell, indicio

**Evidencia ponderada**:
Una señal de comportamiento a la que se le da más o menos peso según cuánto farolea ese rival. Cambia lo que se cree de sus cartas, pero nunca descarta una mano.
_Evitar_: evidencia probable, evidencia débil

### Agentes

**Modo de agente**:
Cada arquitectura de IA que Ottoman puede usar en una partida y que se elige desde la interfaz: Reactivo, Utilidad adaptativa, Red neuronal · Neuroevolución o Red neuronal · Aprendizaje por refuerzo.
_Evitar_: perfil, dificultad, nivel

**Agente existente**:
Cada agente que ya estaba en la app antes del agente entrenado: Reactivo y Utilidad adaptativa. No participa del entrenamiento del agente entrenado.
_Evitar_: legacy, agente viejo, agente heredado

**Agente entrenado**:
Agente cuya política no fue escrita a mano: surge de un proceso de entrenamiento iterativo. Hay uno por fase, y cada uno es un modo de red.
_Evitar_: IA nueva, bot entrenado, Entrenado (como nombre de modo)

**Modo de red**:
Cada uno de los dos modos de agente que juegan con un agente entrenado: Red neuronal · Neuroevolución (fase 1) y Red neuronal · Aprendizaje por refuerzo (fase 2).
_Evitar_: modo entrenado, modo IA

**Adaptarse al rival**:
Opción de un modo de red que, prendida, hace que el agente use el perfil del rival y la evidencia ponderada; apagada, usa valores neutros en los dos.
_Evitar_: modo adaptativo, Entrenado adaptativo

**Traza de razonamiento**:
Registro, visible para el público, de por qué el agente tomó cada decisión en una mano.
_Evitar_: log, XAI, pensamiento

**Perfil del rival**:
Estadísticas de cómo juega un rival concreto (faroles vistos, idas al mazo, truco aceptado, envido cantado), medidas a partir de lo que se observa de él, que arrancan en un valor neutro y se acumulan entre partidas.
_Evitar_: modelo del humano, estilo (para el perfil observado)

**Estilo**:
Forma fija de jugar que se le impone a un rival con estilo: mentiroso, cauteloso, agresivo o calculador.
_Evitar_: personalidad, perfil (para el estilo)

### Entrenamiento

**Fase**:
Cada etapa del entrenamiento, con su propia técnica: fase 1, neuroevolución; fase 2, aprendizaje por refuerzo. Las dos fases son independientes: no se usan una a la otra para entrenar.
_Evitar_: etapa, versión

**Autojuego**:
Entrenamiento en el que el agente juega contra copias de sí mismo o contra versiones anteriores suyas.
_Evitar_: self-play, jugar contra un clon

**Liga**:
El conjunto de rivales contra los que juega un agente mientras entrena: clones, el salón de la fama y rivales con estilo, todos de su misma fase.
_Evitar_: pool, torneo

**Rival con estilo**:
Una copia de una red de la propia fase a la que se le suma un sesgo fijo en sus salidas para que juegue con un estilo.
_Evitar_: sparring, bot de estilo, rival de referencia (para estos rivales)

**Generación**:
Una población completa de individuos que se evalúa de una vez; de la evaluación salen los padres de la generación siguiente. En la fase 2, que no tiene población, se llama generación a cada control periódico del campeón.
_Evitar_: época, iteración

**Individuo**:
Un candidato de la población, definido por su genoma (los pesos de su red).
_Evitar_: agente (para un candidato), bot

**Genoma**:
El vector con todos los pesos de la red de un individuo; es lo que el algoritmo genético cruza y muta.
_Evitar_: cromosoma, ADN

**Fitness**:
Puntaje de un individuo en una generación, que sale de las partidas que gana y de su diferencia de puntos.
_Evitar_: aptitud, score, recompensa

**Campeón**:
El individuo con mejor fitness de una generación.
_Evitar_: mejor bot, ganador

**Campeón vigente**:
El mejor agente de una fase hasta el momento. Un campeón lo reemplaza solo si le gana con claridad; ese reemplazo es una **promoción**. Es el que juega en el modo de red de su fase.
_Evitar_: mejor campeón, campeón activo

**Salón de la fama**:
Conjunto de los campeones que fueron promovidos a campeón vigente, contra los que siguen jugando las generaciones nuevas.
_Evitar_: hall of fame, archivo

**Rival de referencia**:
Agente existente (Reactivo o Utilidad adaptativa) contra el que se mide al agente entrenado en la evaluación de referencia. Nunca es rival en la liga.
_Evitar_: benchmark, sparring

**Evaluación de referencia**:
Medición de los campeones contra los rivales de referencia, hecha aparte del entrenamiento. Su resultado se muestra, pero el entrenamiento nunca lo lee.
_Evitar_: benchmark, test, validación (para esta medición)

**Repartos de validación**:
Conjunto fijo de repartos que nunca se usa para entrenar y sobre el que se deciden las promociones y se hace la evaluación de referencia.
_Evitar_: test set, seeds de evaluación

**Partidas espejo**:
Par de partidas sobre los mismos repartos en las que los jugadores intercambian asientos, para compensar la suerte de las cartas.
_Evitar_: duplicado, partida invertida
