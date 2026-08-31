"""
Stage 1 tests for utils/seed.py

Covers: seed reproducibility.
"""

import os
import sys
import random
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from utils.seed import set_seed, seed_from_config  # noqa: E402
from utils.config_tracking import ConfigManager  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_ROOT = os.path.join(REPO_ROOT, "configs")


class TestSeedReproducibility(unittest.TestCase):
    def test_python_random_reproducible(self):
        set_seed(1234, deterministic=False)
        seq1 = [random.random() for _ in range(10)]
        set_seed(1234, deterministic=False)
        seq2 = [random.random() for _ in range(10)]
        self.assertEqual(seq1, seq2)

    def test_numpy_random_reproducible(self):
        set_seed(999, deterministic=False)
        arr1 = np.random.rand(20)
        set_seed(999, deterministic=False)
        arr2 = np.random.rand(20)
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_seeds_diverge(self):
        set_seed(1, deterministic=False)
        a = np.random.rand(20)
        set_seed(2, deterministic=False)
        b = np.random.rand(20)
        self.assertFalse(np.array_equal(a, b))

    def test_set_seed_rejects_non_int(self):
        with self.assertRaises(TypeError):
            set_seed(3.14, deterministic=False)  # type: ignore[arg-type]

    def test_seed_report_contents(self):
        report = set_seed(7, deterministic=True)
        self.assertEqual(report.seed, 7)
        self.assertTrue(report.deterministic)
        # torch_seeded should track torch_available exactly (both True or both False)
        self.assertEqual(report.torch_available, report.torch_seeded)

    def test_seed_from_config_uses_experiment_seed(self):
        resolved = ConfigManager(config_root=CONFIG_ROOT).load(dataset="cifar100")
        expected_seed = resolved.get("experiment.seed")
        report = seed_from_config(resolved)
        self.assertEqual(report.seed, expected_seed)


if __name__ == "__main__":
    unittest.main()
