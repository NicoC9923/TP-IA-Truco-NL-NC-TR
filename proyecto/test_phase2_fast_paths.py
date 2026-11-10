"""Policy forward fast path equals model() byte for byte, and steps aside when hooked."""
import unittest
import numpy as np
import torch
from phase2_torch import Phase2Network, _q_values


class ForwardFastPathTests(unittest.TestCase):
    def test_matches_module_call_for_trained_like_weights(self):
        rng = np.random.default_rng(5)
        for seed, auxiliary in ((0, False), (3, True), (11, False)):
            model = Phase2Network(seed=seed, auxiliary=auxiliary)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(torch.from_numpy(rng.normal(0, .05, parameter.shape).astype(np.float32)))
                for _ in range(200):
                    x = (rng.random(1628) < .07).astype(np.float32) * rng.normal(0, 1, 1628).astype(np.float32)
                    x = torch.from_numpy(x)
                    self.assertEqual(_q_values(model, x).numpy().tobytes(), model(x).numpy().tobytes())

    def test_hooks_and_subclasses_use_the_module(self):
        model = Phase2Network(seed=1)
        x = torch.zeros(1628)
        handle = model.layers[2].register_forward_hook(lambda module, inputs, output: output + 1)
        with torch.no_grad():
            self.assertEqual(_q_values(model, x).numpy().tobytes(), model(x).numpy().tobytes())
        handle.remove()

        class Shifted(Phase2Network):
            def forward(self, inputs):
                return super().forward(inputs) + 2
        shifted = Shifted(seed=1)
        with torch.no_grad():
            self.assertEqual(_q_values(shifted, x).numpy().tobytes(), shifted(x).numpy().tobytes())


if __name__ == '__main__':
    unittest.main()
