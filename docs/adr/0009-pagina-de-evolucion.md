# Página de evolución aparte, alimentada por dos logs

Una página HTML aparte de la interfaz de juego muestra el entrenamiento. Lee dos registros: el log del entrenador, con una línea por generación, y el log de la evaluación de referencia, que escribe un proceso aparte (ADR 0022). Por generación se ve:

- el fitness máximo, promedio y mínimo;
- la sucesión de campeones vigentes (en qué generación hubo una promoción y con qué porcentaje ganó);
- la curva de la evaluación de referencia: el campeón contra Reactivo y contra Utilidad adaptativa, con su margen de error;
- el comportamiento que va surgiendo (frecuencia de envido, de farol y de aceptación del truco);
- la diversidad de la población;
- un selector para ver una mano de ejemplo jugada por el campeón de cualquier generación, decisión por decisión.

La fase 1 y la fase 2 se muestran lado a lado y, cuando existe la fase 2, se comparan las tres predicciones de las cartas del rival (ADR 0019). La página se separó de la interfaz de juego para que entrenar no dependa de que la partida esté corriendo.
