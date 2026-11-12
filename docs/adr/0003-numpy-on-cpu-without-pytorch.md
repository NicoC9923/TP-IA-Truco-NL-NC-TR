# Phase 1 with numpy on CPU, without PyTorch or a GPU

This applies to phase 1 (neuroevolution). Phase 2 (ADR 0015) uses PyTorch, also on CPU, because reinforcement learning does need gradients.

The network and the genetic algorithm are implemented with numpy (already a project dependency). Generations are evaluated in parallel with `multiprocessing` across the CPU's 16 threads. The bottleneck is simulating the game in Python, branch-heavy logic a GPU does not speed up; the network is small (a few thousand weights), so shipping data to a GPU would cost more than it saves. PyTorch's main advantage, automatic gradients, is not used by a genetic algorithm. Also, PyTorch on an AMD GPU under Windows depends on ROCm, whose support is recent and fragile.

## Consequences

The agent's policy sits behind its own interface, so it can later be replaced by a PyTorch network (for reinforcement learning) without touching the game engine.
