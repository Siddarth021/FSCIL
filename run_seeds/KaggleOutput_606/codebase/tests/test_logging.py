"""
Stage 1 tests for utils/logging.py

Covers: logging initialization (and basic structured-record round trip).
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logging import ExperimentLogger, make_run_id, setup_python_logger  # noqa: E402


class TestLoggingInitialization(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_setup_creates_run_dir_and_log_file(self):
        run_dir = os.path.join(self.tmpdir, "run1")
        logger = setup_python_logger(run_dir, name="test_logger_init")
        logger.info("hello")
        self.assertTrue(os.path.isdir(run_dir))
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "run.log")))
        with open(os.path.join(run_dir, "run.log")) as f:
            content = f.read()
        self.assertIn("hello", content)

    def test_experiment_logger_writes_jsonl(self):
        run_dir = os.path.join(self.tmpdir, "run2")
        exp_logger = ExperimentLogger(run_dir, run_id="unit_test_run")
        exp_logger.log("test_event", foo=1, bar="baz")

        jsonl_path = os.path.join(run_dir, "events.jsonl")
        self.assertTrue(os.path.isfile(jsonl_path))
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["event"], "test_event")
        self.assertEqual(lines[0]["payload"], {"foo": 1, "bar": "baz"})
        self.assertEqual(lines[0]["run_id"], "unit_test_run")

    def test_log_loss_components_convention(self):
        run_dir = os.path.join(self.tmpdir, "run3")
        exp_logger = ExperimentLogger(run_dir, run_id="loss_run")
        exp_logger.log_loss_components(
            session=0, epoch=1, l_total=1.23, l_ce=1.0, l_recon=0.1, l_adv=0.08, l_energy=0.05
        )
        events = exp_logger.read_events()
        self.assertEqual(len(events), 1)
        payload = events[0]["payload"]
        self.assertEqual(payload["L_total"], 1.23)
        self.assertEqual(payload["L_CE"], 1.0)
        self.assertEqual(payload["L_recon"], 0.1)
        self.assertEqual(payload["L_adv"], 0.08)
        self.assertEqual(payload["L_energy"], 0.05)

    def test_log_session_accuracy(self):
        run_dir = os.path.join(self.tmpdir, "run4")
        exp_logger = ExperimentLogger(run_dir, run_id="acc_run")
        exp_logger.log_session_accuracy(session=2, accuracy=0.734)
        events = exp_logger.read_events()
        self.assertEqual(events[0]["event"], "cumulative_session_accuracy")
        self.assertEqual(events[0]["payload"]["session"], 2)
        self.assertAlmostEqual(events[0]["payload"]["accuracy"], 0.734)

    def test_read_events_empty_when_no_log_written(self):
        run_dir = os.path.join(self.tmpdir, "run5")
        exp_logger = ExperimentLogger(run_dir, run_id="empty_run")
        self.assertEqual(exp_logger.read_events(), [])

    def test_make_run_id_format(self):
        run_id = make_run_id("cifar100", git_hash="abcdef123456")
        parts = run_id.split("_")
        self.assertEqual(parts[0], "cifar100")
        self.assertTrue(run_id.endswith("abcdef123456"))

    def test_setup_logger_idempotent_no_duplicate_handlers(self):
        run_dir = os.path.join(self.tmpdir, "run6")
        logger1 = setup_python_logger(run_dir, name="idempotent_logger")
        n_handlers_1 = len(logger1.handlers)
        logger2 = setup_python_logger(run_dir, name="idempotent_logger")
        n_handlers_2 = len(logger2.handlers)
        self.assertEqual(n_handlers_1, n_handlers_2)


if __name__ == "__main__":
    unittest.main()
