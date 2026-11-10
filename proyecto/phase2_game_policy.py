"""Human-match adapter for the published frozen phase 2 policy."""
from phase2_torch import Phase2Network, Phase2Policy


class GamePhase2Policy(Phase2Policy):
    encoder = 'phase2'

    @classmethod
    def load(cls, path, *, seed):
        model = Phase2Network.load(path).eval()
        return cls(model, seed=seed, temperature=.05, profile_prior=5)

    @property
    def inference(self):
        # The game clears this at hand boundaries; both caches must be reset.
        return self

    def reset(self):
        self.exact_inference.reset()
        self.weighted_inference.reset()

    def __call__(self, observation, legal_actions):
        action = super().__call__(observation, legal_actions)
        card = observation['hand'][observation['available'][0] if observation['available'] else 0]
        self.last_belief = self.weighted_inference.belief.summaries(card, ties_win=observation['mano'] == observation['player'])
        return action
