"""Optional PyTorch phase 2 network and policy."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from opponent_inference import IncrementalInference, exact_inference_from_observation, _CARD_IDS
from opponent_profile import OpponentProfile
from phase1_network import ACTIONS
from phase2_network import INPUT_SIZE, encode_observation
from symmetric_engine import bluff


class _CombinationScores(torch.autograd.Function):
    """Three ordered additions with the original gather's gradient accumulation."""
    @staticmethod
    def forward(ctx, logits, ids):
        ctx.save_for_backward(ids)
        ctx.logits_shape = logits.shape
        # The reduction starts at +0, which also preserves signed-zero bits.
        return 0.0 + logits[:, ids[:, 0]] + logits[:, ids[:, 1]] + logits[:, ids[:, 2]]

    @staticmethod
    def backward(ctx, gradient):
        (ids,) = ctx.saved_tensors
        result = gradient.new_zeros(ctx.logits_shape)
        rows = torch.arange(gradient.shape[0], device=gradient.device)[:, None, None]
        # Match IndexBackward's row/hand/card order, including duplicate cards.
        result.index_put_((rows, ids[None, :, :]), gradient[:, :, None].expand(-1, -1, 3), accumulate=True)
        return result, None


def combination_scores(logits, ids):
    """Sum ordered three-card logits with the original CPU gradient order."""
    return _CombinationScores.apply(logits, ids)


class Phase2Network(nn.Module):
    def __init__(self, seed=0, auxiliary=False):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.layers = nn.Sequential(nn.Linear(INPUT_SIZE, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, len(ACTIONS)))
            nn.init.uniform_(self.layers[-1].weight, -0.001, 0.001)
            nn.init.zeros_(self.layers[-1].bias)
            if auxiliary:
                self.card_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 40))

    def forward(self, inputs):
        return self.layers(inputs)

    def card_logits(self, inputs):
        return self.card_head(self.layers[:4](inputs))

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path):
        weights = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(auxiliary='card_head.0.weight' in weights)
        model.load_state_dict(weights)
        return model

def card_distribution(logits, observation):
    """Return compatible original hands, their probabilities and hidden-card marginals."""
    belief = exact_inference_from_observation(observation)
    opponent = 1 - observation['player']
    played = [card for player, card in observation['table'] if player == opponent]
    hands = tuple(belief.weights)
    if not hands:
        raise ValueError('No hands compatible with observation')
    ids = torch.tensor([[_CARD_IDS[c] for c in hand] for hand in hands], device=logits.device)
    probabilities = torch.softmax(logits[ids].sum(dim=1), dim=0)
    marginals = torch.zeros(40, dtype=logits.dtype, device=logits.device)
    marginals = marginals.scatter_add(0, ids.flatten(), probabilities.repeat_interleave(3))
    if played:
        marginals = marginals.scatter(0, torch.tensor([_CARD_IDS[c] for c in played], device=logits.device), 0.0)
    return hands, probabilities, marginals

def _q_values(model, inputs):
    """model(inputs) for one decision without nn.Module dispatch.

    A plain, hook-free Phase2Network runs the same linear and ReLU kernels in
    the same order as nn.Sequential; anything else goes through model().
    """
    if type(model) is not Phase2Network or _hooked(model) or tuple(map(type, model.layers)) != _PLAIN_LAYERS:
        return model(inputs)
    first, _, second, _, third = model.layers
    hidden = torch.relu(F.linear(inputs, first.weight, first.bias))
    hidden = torch.relu(F.linear(hidden, second.weight, second.bias))
    return F.linear(hidden, third.weight, third.bias)


_PLAIN_LAYERS = (nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear)


def _hooked(model):
    from torch.nn.modules import module as nn_module
    if (nn_module._global_forward_hooks or nn_module._global_forward_pre_hooks
            or nn_module._global_forward_hooks_always_called):
        return True
    return any(m._forward_hooks or m._forward_pre_hooks for m in (model, model.layers, *model.layers))


_DEFAULT_PROFILE_PRIOR = object()


class Phase2Policy:
    """Sample with a seeded RNG that continues across matches."""
    def __init__(self, model=None, *, seed=0, temperature=0.1, profile=None,
                 weighted_beliefs=True, style=None, style_strength=0.1, adapt=True,
                 profile_prior=_DEFAULT_PROFILE_PRIOR, greedy=False):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be positive and finite')
        if style not in (None, 'bluffer', 'calculating', 'cautious', 'aggressive'):
            raise ValueError('Unknown style')
        if profile_prior is _DEFAULT_PROFILE_PRIOR:
            profile_prior = profile.prior_strength if profile is not None else 20
        if not np.isfinite(profile_prior) or profile_prior <= 0:
            raise ValueError('profile_prior must be positive and finite')
        if profile is not None and profile.prior_strength != profile_prior:
            raise ValueError('profile_prior does not match profile')
        self.model = model if model is not None else Phase2Network(seed=seed)
        self.profile = profile if profile is not None else OpponentProfile(prior_strength=profile_prior)
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        self.greedy = greedy
        self.weighted_beliefs = weighted_beliefs
        self.adapt = adapt
        self.neutral_profile = OpponentProfile(prior_strength=profile_prior)
        self.style, self.style_strength = style, style_strength
        self.exact_inference = IncrementalInference()
        self.weighted_inference = IncrementalInference()

    def encode(self, observation, legal_actions):
        return encode_observation(observation, self.profile if self.adapt else self.neutral_profile, legal_actions,
                                  weighted_beliefs=self.weighted_beliefs,
                                  exact_inference=self.exact_inference,
                                  weighted_inference=self.weighted_inference)

    def __call__(self, observation, legal_actions):
        inputs = self.encode(observation, legal_actions)
        with torch.no_grad():
            values = _q_values(self.model, torch.from_numpy(inputs)).numpy()
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
        if self.greedy:
            # Highest legal Q; ties go to the lowest action index. No RNG is consumed.
            best = max(scores)
            action = min((a for a, s in zip(legal_actions, scores) if s == best), key=ACTIONS.index)
            self.last_probabilities = {a: float(a == action) for a in legal_actions}
            self.record_decision(inputs, action)
            return action
        logits = np.asarray(scores, dtype=np.float64) / self.temperature
        weights = np.exp(logits - logits.max())
        probabilities = weights / weights.sum()
        self.last_probabilities = dict(zip(legal_actions, map(float, probabilities)))
        action = legal_actions[self.rng.choice(len(legal_actions), p=probabilities)]
        self.record_decision(inputs, action)
        return action

    def record_decision(self, inputs, action):
        """Hook for consumers that collect decisions during training."""
        pass

    def observe_hand(self, decisions, opponent):
        self.profile.observe(decisions, opponent, semantics='phase2')
        self.exact_inference.reset()
        self.weighted_inference.reset()
