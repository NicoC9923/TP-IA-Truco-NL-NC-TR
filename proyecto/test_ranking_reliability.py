import subprocess
import sys
import unittest
import numpy as np
from ranking_reliability import spearman, summarize, diagnostic_schedule


class RankingReliabilityTest(unittest.TestCase):
    def test_spearman_ties(self):
        self.assertAlmostEqual(spearman([1, 2, 2, 4], [4, 2, 2, 1]), -1)
        self.assertIsNone(spearman([1, 1], [1, 2]))

    def test_report_and_verdict(self):
        ref = np.arange(10)
        good = np.arange(10)
        bad = good[::-1]
        report = summarize({16: [(good, bad)] * 3, 32: [(good, good)] * 3}, ref)
        self.assertEqual(report['verdict'], 'proceed')
        self.assertEqual(report['pairs']['32']['top_8_overlap'], [8, 8, 8])
        self.assertEqual(report['pairs']['32']['mean_reference_score_top_8'], 5.5)
        self.assertEqual(summarize({16: [(good, good)], 32: [(good, good)]}, ref)['verdict'], 'do not proceed')

    def test_seed_namespace(self):
        config = [None] * 15
        config[2], config[14] = 2, True
        seeds = [item['seed'] for stream in range(13) for item in diagnostic_schedule(4, 64, config, [], None, stream)]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(s % 2 and s > 2 * 10**17 for s in seeds))

    def test_invalid_cli(self):
        proc = subprocess.run([sys.executable, 'proyecto/ranking_reliability.py', 'none', '--output', 'none', '--splits', '0'], capture_output=True)
        self.assertNotEqual(proc.returncode, 0)


if __name__ == '__main__':
    unittest.main()
