"""Optional PyTorch phase 2 network and policy."""
import numpy as np
import torch
from torch import nn
from opponent_inference import IncrementalInference
from opponent_profile import OpponentProfile
from phase1_network import ACTIONS
from phase2_network import INPUT_SIZE, encode_observation
from symmetric_engine import bluff

class Phase2Network(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.layers = nn.Sequential(nn.Linear(INPUT_SIZE, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, len(ACTIONS)))
            nn.init.uniform_(self.layers[-1].weight, -0.001, 0.001)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, inputs):
        return self.layers(inputs)

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        return model

class Phase2Policy:
    """Sample with a seeded RNG that continues across matches."""
    def __init__(self, model=None, *, seed=0, temperature=0.1, profile=None,
                 weighted_beliefs=True, style=None, style_strength=0.1):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be positive and finite')
        if style not in (None, 'bluffer', 'calculating', 'cautious', 'aggressive'):
            raise ValueError('Unknown style')
        self.model = model if model is not None else Phase2Network(seed=seed)
        self.profile = profile if profile is not None else OpponentProfile()
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        self.weighted_beliefs = weighted_beliefs
        self.style, self.style_strength = style, style_strength
        self.exact_inference = IncrementalInference()
        self.weighted_inference = IncrementalInference()

    def encode(self, observation, legal_actions):
        return encode_observation(observation, self.profile, legal_actions,
                                  weighted_beliefs=self.weighted_beliefs,
                                  exact_inference=self.exact_inference,
                                  weighted_inference=self.weighted_inference)

    def __call__(self, observation, legal_actions):
        inputs = self.encode(observation, legal_actions)
        with torch.no_grad():
            values = self.model(torch.from_numpy(inputs)).numpy()
        scores = []
        for action in legal_actions:
            offset = 0.0
            if self.style in ('bluffer', 'calculating') and bluff(action, observation['hand']):
                offset += self.style_strength * (1 if self.style == 'bluffer' else -1)
            if self.style == 'cautious' and action == 'quiero':
                offset -= self.style_strength
            if self.style == 'aggressive' and action in ('retruco', 'vale_cuatro', 'real_envido', 'falta_envido'):
                offset += self.style_strength
            scores.append(float(values[ACTIONS.index(action)]) + offset)
        logits = np.asarray(scores, dtype=np.float64) / self.temperature
        weights = np.exp(logits - logits.max())
        probabilities = weights / weights.sum()
        self.last_probabilities = dict(zip(legal_actions, map(float, probabilities)))
        return legal_actions[self.rng.choice(len(legal_actions), p=probabilities)]

    def observe_hand(self, decisions, opponent):
        self.profile.observe(decisions, opponent, semantics='phase2')
        self.exact_inference.reset()
        self.weighted_inference.reset()
