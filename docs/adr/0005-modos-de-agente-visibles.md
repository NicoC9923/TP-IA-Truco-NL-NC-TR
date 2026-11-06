# Cuatro modos de agente: reactivo, utilidad adaptativa, entrenado y entrenado adaptativo

Se elimina el modo "utilidad" estático, porque era el modo "aprendizaje" con los ajustes apagados, y se fusiona en un único modo **Utilidad adaptativa**. En la interfaz quedan cuatro arquitecturas para comparar en vivo:

- **Reactivo:** agente reflejo simple, piso de comparación.
- **Utilidad adaptativa:** umbrales sobre las tablas de 474.000 manos, con faroles y modelado del rival.
- **Entrenado:** la red entrenada, con el perfil del rival siempre neutro.
- **Entrenado adaptativo:** la misma red, con el perfil del rival real (ver ADR 0012).

Comparar los dos últimos muestra directamente cuánto suma adaptarse al rival.
