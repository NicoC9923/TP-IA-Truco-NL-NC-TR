"""Warm start (ADR 0062): a wider input layer with zero new columns computes the old function exactly."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import phase2_network as net
from phase2_torch import Phase2Network, Phase2Policy, widen_state_dict, _q_values
from symmetric_engine import play_match


def random_network(seed=0):
    model = Phase2Network(seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model


class Recorder(Phase2Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = []

    def record_decision(self, inputs, action):
        self.inputs.append(inputs)


class WarmStartTest(unittest.TestCase):
    def test_network_size_follows_the_encoder_version(self):
        for version in (1, 2, 3):
            self.assertEqual(Phase2Network(input_size=net.INPUT_SIZES[version]).layers[0].in_features,
                             net.INPUT_SIZES[version])

    def test_load_infers_the_encoder_from_the_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            for version in (1, 2, 3):
                path = Path(folder) / f'{version}.pt'
                Phase2Network(input_size=net.INPUT_SIZES[version]).save(path)
                self.assertEqual(Phase2Network.load(path).encoder_version, version)

    def test_widening_zero_fills_new_columns_only(self):
        old = random_network().state_dict()
        wide = widen_state_dict(old, 3)
        first = wide['layers.0.weight']
        self.assertEqual(tuple(first.shape), (256, net.INPUT_SIZES[3]))
        self.assertTrue(torch.equal(first[:, :net.INPUT_SIZE], old['layers.0.weight']))
        self.assertEqual(float(first[:, net.INPUT_SIZE:].abs().max()), 0.0)
        for key in old:
            if key != 'layers.0.weight':
                self.assertTrue(torch.equal(wide[key], old[key]))
        with self.assertRaises(ValueError):
            widen_state_dict(wide, 2)  # narrowing is refused

    def test_outputs_are_bit_identical_on_real_observations(self):
        old = random_network()
        for version in (2, 3):
            wide = Phase2Network(input_size=net.INPUT_SIZES[version])
            wide.load_state_dict(widen_state_dict(old.state_dict(), version))
            policies = [Recorder(old, seed=5, temperature=.3), Recorder(old, seed=6, temperature=.3)]
            play_match(*policies, seed=11, target=15)
            inputs = [x for p in policies for x in p.inputs]
            self.assertGreater(len(inputs), 20)
            # The wide policy must pick its own encoding, so rebuild the inputs through it.
            twins = [Recorder(wide, seed=5, temperature=.3), Recorder(wide, seed=6, temperature=.3)]
            play_match(*twins, seed=11, target=15)
            wide_inputs = [x for p in twins for x in p.inputs]
            self.assertEqual(len(inputs), len(wide_inputs))
            for x_old, x_new in zip(inputs, wide_inputs):
                self.assertEqual(x_new.shape[0], net.INPUT_SIZES[version])
                self.assertEqual(x_new[:net.INPUT_SIZE].tobytes(), x_old.tobytes())
                a = _q_values(old, torch.from_numpy(x_old)).detach().numpy()
                b = _q_values(wide, torch.from_numpy(x_new)).detach().numpy()
                self.assertEqual(a.tobytes(), b.tobytes())
            batch_old = torch.from_numpy(np.stack(inputs))
            batch_new = torch.from_numpy(np.stack(wide_inputs))
            self.assertEqual(old(batch_old).detach().numpy().tobytes(), wide(batch_new).detach().numpy().tobytes())

    def test_policies_play_the_same_match(self):
        old = random_network(3)
        wide = Phase2Network(input_size=net.INPUT_SIZES[2])
        wide.load_state_dict(widen_state_dict(old.state_dict(), 2))
        first = play_match(Phase2Policy(old, seed=1, temperature=.2), Phase2Policy(old, seed=2, temperature=.2), seed=4, target=15)
        second = play_match(Phase2Policy(wide, seed=1, temperature=.2), Phase2Policy(wide, seed=2, temperature=.2), seed=4, target=15)
        self.assertEqual([d['action'] for d in first.decisions], [d['action'] for d in second.decisions])


if __name__ == '__main__':
    unittest.main()
