"""Print a reproducible match's public opponent inference at each decision."""
import argparse
import json
import random
from opponent_inference import inference_from_observation
from opponent_profile import OpponentProfile
from symmetric_engine import play_match




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--target', type=int, default=2)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    def policy(obs, legal):
        choices = [action for action in legal if action != 'fold']
        return rng.choice(choices)
    match = play_match(policy, policy, seed=args.seed, target=args.target)
    profiles = [OpponentProfile(), OpponentProfile()]
    previous_hand = 0
    hand_decisions = []
    for decision in match.decisions:
        if decision['hand_index'] != previous_hand:
            for player in (0, 1):
                profiles[player].observe(hand_decisions, 1 - player)
            hand_decisions = []
            previous_hand = decision['hand_index']
        obs = decision['observation']
        profile = profiles[obs['player']].rates()
        belief = inference_from_observation(obs, profile['bluff'])
        card = obs['hand'][obs['available'][0]] if obs['available'] else None
        summaries = belief.summaries(card) if card else None
        print(json.dumps({'hand': previous_hand, 'player': obs['player'], 'action': decision['action'], 'possible_hands': len(belief.distribution()), 'summaries': summaries, 'profile': profile}))
        hand_decisions.append(decision)

if __name__ == '__main__':
    main()
