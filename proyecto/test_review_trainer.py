import io, json, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path
from phase1_trainer import train

class TrainerReviewTests(unittest.TestCase):
    def test_resume_truncates_log_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            opts = dict(population_size=2, pairs=1, workers=1, target=1, seed=7, validation_pairs=1, patience=30, promotion_threshold=0)
            train(d, generations=11, **opts)
            train(d, generations=11, resume=Path(d, 'checkpoint_0010.npz'), **opts)
            self.assertEqual([json.loads(x)['generation'] for x in Path(d, 'generations.jsonl').read_text().splitlines()], list(range(1,12)))
    def test_progress_shows_promotion_without_mojibake(self):
        with tempfile.TemporaryDirectory() as d:
            output = io.StringIO()
            with redirect_stdout(output):
                train(d, population_size=2, generations=2, workers=1, target=1, validation_pairs=1)
            self.assertIn('first champion', output.getvalue())
            self.assertIn('promoted', output.getvalue())
            self.assertNotIn('Â', output.getvalue())
