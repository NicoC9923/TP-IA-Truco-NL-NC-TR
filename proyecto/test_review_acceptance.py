import io, json, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from phase1_trainer import train
from symmetric_engine import play_match

class AcceptanceReviewTests(unittest.TestCase):
    def test_observations_match_deals_and_only_played_cards_reach_table(self):
        def play(obs, actions):
            return next(a for a in actions if a.startswith('card_'))
        result = play_match(play, play, seed=4, target=4)
        tables = {}
        for decision in result.decisions:
            obs = decision['observation']; hand = decision['hand_index']
            self.assertEqual(obs['hand'], result.deals[hand][decision['player']])
            self.assertEqual(obs['table'], tuple(tables.get(hand, [])))
            if decision['action'].startswith('card_'):
                tables.setdefault(hand, []).append((decision['player'], obs['hand'][int(decision['action'][-1])]))

    def test_training_schedules_targets_scores_and_disjoint_partitions(self):
        jobs = []
        def evaluate(job):
            jobs.append(job)
            seed = job[2]
            decision = {'player': 0, 'hand_index': 0, 'bluff': None, 'action': 'fold', 'observation': {'pending': None}}
            r = SimpleNamespace(winner=0, points=(2, 0), decisions=[decision])
            return [(r, 0), (r, 1)]
        with tempfile.TemporaryDirectory() as d, patch('phase1_trainer._evaluate', side_effect=evaluate), redirect_stdout(io.StringIO()):
            train(d, population_size=64, generations=2, pairs=4, workers=1, seed=9, validation_pairs=8, promotion_threshold=0)
            rows = [json.loads(line) for line in Path(d, 'generations.jsonl').read_text().splitlines()]
        training = [j for j in jobs if j[2] % 2]
        validation = [j for j in jobs if not j[2] % 2]
        self.assertEqual({j[3] for j in training}, {15, 30})
        self.assertTrue(.23 < sum(j[4] != (0,0) for j in training) / len(training) < .37)
        self.assertEqual(len(validation), 8)
        self.assertTrue(set(j[2] for j in training).isdisjoint(j[2] for j in validation))
        from collections import Counter
        for row in rows:
            self.assertEqual(Counter(row['league']), {'unbiased': 128, 'bluffer': 32, 'cautious': 32, 'aggressive': 32, 'calculating': 32})
        self.assertNotEqual(rows[0]['league'][:8], rows[1]['league'][:8])
    def test_human_falta_envido_at_30_and_tied_tantos(self):
        import contextlib, random
        from partida_truco import PartidaTruco
        for seed, target, scores, call in [(3, 30, (4, 8), 'falta_envido'), (9, 2, (0, 0), 'envido')]:
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                human = PartidaTruco(meta_puntos=target, historial_path=str(Path(d, 'history.json')), rng=random.Random(seed))
                human.puntos_humano, human.puntos_robot = scores
                with patch.object(human.ia, 'responder_envido', return_value=('QUIERO', '')):
                    human.cantar_envido_humano(call.upper())
                result = play_match(lambda o,a: call if call in a else 'quiero', lambda o,a: 'quiero', seed=seed, target=target, starting_score=scores)
                score = human.estado_publico()['tanteador']
                self.assertEqual(result.points, (score['humano'], score['robot']))
    def test_train_passes_partition_matching_training_and_validation_seeds(self):
        import phase1_trainer
        calls = []
        real = phase1_trainer.play_match
        def recording(*args, **kwargs):
            calls.append((kwargs['seed'], kwargs['deal_partition']))
            return real(*args, **kwargs)
        with tempfile.TemporaryDirectory() as d, patch('phase1_trainer.play_match', side_effect=recording), redirect_stdout(io.StringIO()):
            train(d, population_size=2, generations=2, workers=1, target=1, seed=3, validation_pairs=2)
        self.assertEqual({partition for _, partition in calls}, {0, 1})
        self.assertTrue(all(seed % 2 == partition for seed, partition in calls))
