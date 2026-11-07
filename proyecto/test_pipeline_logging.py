import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_phase1


class PipelineLoggingTests(unittest.TestCase):
    def test_failure_is_written_to_pipeline_log(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(Path(run_phase1.__file__)), '--output', directory,
                                     '--population', '1', '--fitness-mode', 'mixed', '--points-weight', '.5',
                                     '--style-strength', '.25', '--no-styles-after-first-promotion'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            log = (Path(directory) / 'pipeline.log').read_text(encoding='utf-8')
            self.assertIn('1/4 training failed with exit code', log)
            for option in ('--fitness-mode mixed', '--points-weight 0.5', '--style-strength 0.25', '--no-styles-after-first-promotion'):
                self.assertIn(option, log)

    def test_success_is_written_to_pipeline_log(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'argv', ['run_phase1', '--output', directory, '--no-commit']), \
                 patch.object(run_phase1, 'run'), patch.object(run_phase1, 'keep_awake'), contextlib.redirect_stdout(io.StringIO()):
                run_phase1.main()
            self.assertIn('Pipeline finished', (Path(directory) / 'pipeline.log').read_text(encoding='utf-8'))
