# Cuatro modos de agente, nombrados por cómo aprendió cada uno

Se elimina el modo "utilidad" estático, porque era el modo "aprendizaje" con los ajustes apagados, y se fusiona en un único modo **Utilidad adaptativa**. En la interfaz quedan cuatro arquitecturas para comparar en vivo:

- **Reactivo:** agente reflejo simple, piso de comparación.
- **Utilidad adaptativa:** umbrales sobre las tablas de 474.000 manos, con faroles y modelado del rival.
- **Red neuronal · Neuroevolución:** la red de la fase 1, entrenada con un algoritmo genético.
- **Red neuronal · Aprendizaje por refuerzo:** la red de la fase 2, entrenada con Deep Monte-Carlo.

Los nombres de los dos modos nuevos dicen cómo aprendió cada red, así la comparación entre el algoritmo genético y el aprendizaje por refuerzo se ve en el mismo selector. Junto a los dos modos de red hay un interruptor **Adaptarse al rival** y un botón para reiniciar el perfil del rival (ADR 0012). Apagado, la red juega con valores neutros; prendido, usa el perfil del humano. Comparar las dos posiciones muestra directamente cuánto suma adaptarse. Cada modo de red aparece como no disponible mientras su fase no tenga un campeón vigente guardado.

## Opciones consideradas

- **"Entrenado" y "Entrenado adaptativo"** (la versión original): no dicen con qué técnica se entrenó la red, y con dos fases habría que elegir el campeón en otro lado.
- **Seis modos separados** (cada red, adaptativa y no adaptativa, más los dos existentes): llena el selector y oculta que la versión adaptativa y la neutra son la misma red.
