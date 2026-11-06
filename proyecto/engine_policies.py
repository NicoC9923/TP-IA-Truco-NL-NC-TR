"""Reference policies for the symmetric engine."""
import random
from pathlib import Path
from truco_ia import AgenteTrucoIA

class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
    def __call__(self, observation, actions):
        return self.rng.choice(actions)

class ReactivePolicy:
    def __init__(self, seed=0):
        self.agent = AgenteTrucoIA(modo_agente='reactivo', historial_path=str(Path(__file__).with_name('historial_partidas.seed.json')), rng=random.Random(seed))
    def __call__(self, obs, actions):
        hand = [{'valor': v, 'palo': p} for v, p in obs['hand']]
        cards = [hand[i] for i in obs['available']]
        pending = obs['pending']
        opponent = None
        if len(obs['table']) % 2:
            v, p = obs['table'][-1][1]
            opponent = {'valor': v, 'palo': p}
        winners = ['parda' if w is None else 'robot' if w == obs['player'] else 'humano' for w in obs['tricks']]
        if pending:
            if pending['kind'] == 'envido':
                choice = self.agent.responder_envido(hand, pending['call'].upper(), obs['mano'] == obs['player'])[0].lower()
            else:
                level = ('TRUCO', 'RETRUCO', 'VALE_CUATRO')[pending['level']-2]
                choice = self.agent.responder_canto_truco(cards, level, winners, opponent)[0].lower()
            return choice if choice in actions else 'no_quiero'
        if 'envido' in actions:
            choice = self.agent.decidir_iniciar_envido(hand, obs['mano'] == obs['player'], obs['score'][1-obs['player']], obs['score'][obs['player']], obs['target'])[0]
            if choice and choice.lower() in actions:
                return choice.lower()
        level = ('NADA', 'TRUCO', 'RETRUCO', 'VALE_CUATRO')[obs['truco']-1]
        choice = self.agent.decidir_iniciar_truco(cards, level, winners, opponent)[0]
        if choice and choice.lower() in actions:
            return choice.lower()
        card = self.agent.elegir_carta_a_jugar(cards, len(obs['tricks'])+1, opponent, winners)
        return 'card_%d' % hand.index(card)

