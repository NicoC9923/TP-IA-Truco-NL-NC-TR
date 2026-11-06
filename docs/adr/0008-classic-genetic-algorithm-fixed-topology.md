# Classic genetic algorithm with a fixed network topology

Each individual's genome is the vector of all weights of a fixed-shape network. Each generation applies tournament selection, elitism (the best pass through unchanged), weight crossover and gaussian mutation, over a population of about 64 individuals. Chosen because it is the genetic algorithm from the course syllabus and is easy to explain and visualize.

## Considered Options

- **NEAT** (also evolves the network's shape): very showy, but much more complex to implement. Left as a possible extension.
- **Evolution strategies** (CMA-ES, OpenAI-ES): usually converge better, but move away from the genetic algorithm taught in the course.
