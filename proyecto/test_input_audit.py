import unittest

from input_audit import card_strength, determinable_winners
import truco_reglas as rules


class InputAuditTests(unittest.TestCase):
    def test_hierarchy_matches_rules_for_the_deck(self):
        for suit in rules.PALOS:
            for value in rules.VALORES:
                self.assertEqual(card_strength((value, suit)), rules.valor_carta_truco(value, suit))

    def test_opponent_card_on_table_determines_winners(self):
        observation = {'player': 0, 'hand': ((1, 'Espada'), (4, 'Oro'), (3, 'Copa')),
                       'available': (0, 1, 2), 'table': ((1, (2, 'Oro')),)}
        self.assertEqual(determinable_winners(observation), (0, 2))

    def test_no_opponent_card_means_unknown(self):
        observation = {'player': 0, 'hand': ((1, 'Espada'),), 'available': (0,), 'table': ()}
        self.assertIsNone(determinable_winners(observation))

    def test_mistakes_use_only_legal_actions_and_certain_facts(self):
        from input_audit import mistake_actions
        obs = {'player': 0, 'hand': ((7, 'Oro'), (6, 'Oro'), (4, 'Copa')),
               'available': (0, 1, 2), 'table': (), 'tricks': (),
               'pending': {'kind': 'envido'}, 'envido_reveal': None}
        self.assertEqual(mistake_actions(obs, ('quiero', 'no_quiero', 'fold'))['decline_high_envido'], ('no_quiero', 'fold'))

    def test_near_aliasing_respects_relevant_state(self):
        from input_audit import aliasing
        import numpy as np
        result = aliasing(np.array([[0., 0.], [0., 0.], [0.001, 0.], [1., 1.]]),
                          [('a',), ('b',), ('a',), ('c',)], radius=0.01)
        self.assertEqual(result['exact']['observations'], 2)
        self.assertEqual(result['near']['observations'], 3)

    def test_card_choice_reports_winning_and_cheapest_probability(self):
        from input_audit import card_choice_audit
        obs = {'player': 0, 'hand': ((1, 'Espada'), (4, 'Oro'), (3, 'Copa')),
               'available': (0, 1, 2), 'table': ((1, (2, 'Oro')),)}
        p = {'card_0': .2, 'card_1': .5, 'card_2': .3}
        result = card_choice_audit([(obs, p)])
        self.assertEqual(result['opportunities'], 1)
        self.assertAlmostEqual(result['winning_mass'], .5)
        self.assertAlmostEqual(result['cheapest_winning_mass'], .3)
        self.assertAlmostEqual(result['stronger_winning_mass'], .2)
        self.assertEqual(result['pairwise']['disagree']['pairs'], 2)

    def test_per_card_uses_table_when_opponent_has_played(self):
        from input_audit import per_card_win_probabilities
        obs = {'player': 0, 'hand': ((1, 'Espada'), (4, 'Oro'), (3, 'Copa')),
               'available': (0, 1, 2), 'table': ((1, (2, 'Oro')),)}
        self.assertEqual(per_card_win_probabilities(obs), {0: 1., 1: 0., 2: 1.})

    def test_per_card_comparison_separates_first_and_other(self):
        from input_audit import per_card_audit
        rows = [({'available': (0, 1)}, {'card_0': .2, 'card_1': .8}, {0: .8, 1: .1}),
                ({'available': (0, 1)}, {'card_0': .7, 'card_1': .3}, {0: .1, 1: .8})]
        result = per_card_audit(rows)
        self.assertEqual(result['first']['opportunities'], 1)
        self.assertEqual(result['other']['opportunities'], 1)
        self.assertEqual(result['first']['best_card_argmax_rate'], 0.)
        self.assertEqual(result['other']['best_card_argmax_rate'], 0.)
        self.assertAlmostEqual(result['other']['best_card_mass'], .3)


    def test_preference_probe_holds_other_features_fixed(self):
        import numpy as np
        from input_audit import preference_probe
        from phase1_network import ACTIONS, GENOME_SIZE, INPUTS, HIDDEN
        genome = np.zeros(GENOME_SIZE)
        genome[0] = 1.0
        genome[INPUTS * HIDDEN + HIDDEN] = 1.0
        result = preference_probe(genome, np.zeros(INPUTS), slot=0)
        self.assertGreater(result['face_value_correlation'], .99)
        self.assertGreater(result['face_value_correlation'], result['strength_correlation'])

    def test_inference_probabilities_are_for_each_available_card(self):
        from input_audit import per_card_win_probabilities
        obs = {'player': 0, 'mano': 0, 'hand': ((1, 'Espada'), (4, 'Oro'), (3, 'Copa')),
               'available': (0, 1, 2), 'table': (), 'history': (), 'envido_reveal': None}
        p = per_card_win_probabilities(obs)
        self.assertGreater(p[0], p[2])
        self.assertGreater(p[2], p[1])

    def test_mistake_rates_track_opportunities_and_legal_mass(self):
        from input_audit import mistake_audit
        obs = {'player': 0, 'hand': ((7, 'Oro'), (6, 'Oro'), (4, 'Copa')),
               'available': (0, 1, 2), 'table': (), 'tricks': (),
               'pending': {'kind': 'envido'}, 'envido_reveal': None}
        result = mistake_audit([(obs, {'quiero': .4, 'no_quiero': .5, 'fold': .1})])
        self.assertEqual(result['decline_high_envido']['opportunities'], 1)
        self.assertAlmostEqual(result['decline_high_envido']['mean_probability'], .6)
        self.assertEqual(result['accept_known_losing_envido']['opportunities'], 0)

    def test_last_trick_tie_is_not_a_losing_card(self):
        from input_audit import mistake_actions
        obs = {'player': 0, 'hand': ((1, 'Espada'), (3, 'Oro'), (4, 'Copa')),
               'available': (0, 1, 2), 'table': ((0, (2, 'Oro')), (1, (5, 'Oro')), (1, (3, 'Copa'))),
               'tricks': (0, 1), 'pending': None, 'envido_reveal': None}
        mistakes = mistake_actions(obs, ('card_0', 'card_1', 'card_2'))
        self.assertEqual(mistakes['lose_last_trick'], ('card_2',))

    def test_aliasing_reports_groups_and_fact_examples(self):
        from input_audit import aliasing
        import numpy as np
        result = aliasing(np.array([[0.], [0.], [.005]]), [('a',), ('b',), ('a',)], radius=.01)
        self.assertEqual(result['exact']['groups'], 1)
        self.assertEqual(result['exact']['examples'][0]['facts'], [('a',), ('b',)])

    def test_cli_rejects_invalid_counts(self):
        import subprocess, sys
        from pathlib import Path
        script = Path(__file__).with_name('input_audit.py')
        process = subprocess.run([sys.executable, str(script), '--workers', '0', '--output', 'ignored.json'], capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('positive', process.stderr)

    def test_reserved_namespace_is_separate(self):
        from input_audit import BANK_SEED, bank_seed
        from behavior_diversity import BANK_SEED as old_seed
        from offspring_diagnostic import SEED_BASE
        self.assertGreater(BANK_SEED, max(old_seed, SEED_BASE) + 10**6)
        self.assertEqual(bank_seed(0) % 2, 1)
        self.assertEqual(bank_seed(1) - bank_seed(0), 2)

    def test_strategic_facts_capture_rank_legality_and_envido(self):
        from input_audit import strategic_facts
        obs = {'hand': ((7, 'Oro'), (6, 'Oro'), (4, 'Copa'))}
        self.assertEqual(strategic_facts(obs, ('card_0', 'fold')),
                         ((10, 2, 0), ('card_0', 'fold'), 33))

    def test_hand_cannot_lose_only_with_a_secured_trick(self):
        from input_audit import cannot_lose_hand
        obs = {'player': 0, 'hand': ((1, 'Espada'), (1, 'Basto'), (7, 'Espada')),
               'available': (0,), 'table': ((0, (1, 'Basto')), (1, (4, 'Oro')),
                                          (0, (7, 'Espada')), (1, (3, 'Oro'))),
               'tricks': (0, 1)}
        self.assertTrue(cannot_lose_hand(obs))
        obs['tricks'] = (1, 0)
        self.assertFalse(cannot_lose_hand(obs))

    def test_audit_runs_all_measurements_on_frozen_synthetic_bank(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        from input_audit import audit
        from phase1_network import GENOME_SIZE, INPUTS
        obs = {'player': 0, 'hand': ((1, 'Espada'), (4, 'Oro'), (3, 'Copa')),
               'available': (0, 1, 2), 'table': ((1, (2, 'Oro')),), 'tricks': (),
               'pending': None, 'envido_reveal': None}
        bank = [(obs, ('card_0', 'card_1', 'card_2', 'fold'))]
        class Pool:
            def __init__(self, max_workers): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def map(self, function, items, chunksize=1):
                return [function(item) for item in items]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.npz'
            np.savez(checkpoint, population=np.zeros((2, GENOME_SIZE)))
            with patch('input_audit.build_bank', return_value=(bank, 8)), \
                 patch('input_audit._prepare_observation', return_value=(np.zeros(INPUTS), {0: 1., 1: 0., 2: 1.})), \
                 patch('concurrent.futures.ProcessPoolExecutor', Pool):
                result = audit(checkpoint, card_decisions=1, workers=1, members=1, champions=[])
        network = next(iter(result['networks'].values()))
        self.assertEqual(result['card_decisions'], 1)
        self.assertAlmostEqual(network['h1']['winning_mass'], .5)
        self.assertEqual(network['h1']['opportunities'], 1)
        self.assertIn('first', network['h2'])
        self.assertIn('decline_high_envido', network['mistakes'])
        self.assertEqual(result['aliasing']['exact']['observations'], 0)

    def test_cli_writes_json_and_passes_worker_count(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from input_audit import main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            with patch('input_audit.audit', return_value={'ok': True}) as mocked:
                main(['--output', str(output), '--workers', '3'])
            self.assertEqual(json.loads(output.read_text()), {'ok': True})
            self.assertEqual(mocked.call_args.kwargs['workers'], 3)

    def test_interpretation_requires_sizable_effects(self):
        from input_audit import interpret
        h1 = {'error_rate_ratio': 2.5, 'error_rate_ratio_ci95': [2.1, 3.2]}
        h2 = {'first': {'opportunities': 1000, 'best_card_argmax_rate': .8},
              'other': {'opportunities': 1000, 'best_card_argmax_rate': .5}}
        result = interpret(h1, h2, {'mean_face_value_correlation': .1, 'mean_strength_correlation': .2},
                           {'near': {'rate': .06}})
        self.assertEqual(result, {'H1': True, 'H2': True, 'H3': True})
        h1['error_rate_ratio_ci95'] = [1.5, 3.2]
        h2['other']['best_card_argmax_rate'] = .79
        self.assertEqual(interpret(h1, h2, {}, {'near': {'rate': .01}}),
                         {'H1': False, 'H2': False, 'H3': False})
