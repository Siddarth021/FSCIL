"""
Stage 1 tests for utils/config_tracking.py

Covers (per Stage 1 requirements):
  - configuration loading
  - configuration validation
  - provenance tracking
  - missing required parameter detection
  - experiment configuration saving/loading
"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config_tracking import (  # noqa: E402
    ConfigManager,
    ConfigParseError,
    ConfigValidationError,
    UnresolvedParameterError,
    ParamEntry,
    ResolvedConfig,
    PROVENANCE_TAGS,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_ROOT = os.path.join(REPO_ROOT, "configs")


class TestConfigLoading(unittest.TestCase):
    def setUp(self):
        self.mgr = ConfigManager(config_root=CONFIG_ROOT)

    def test_loads_without_error(self):
        resolved = self.mgr.load()
        self.assertIsInstance(resolved, ResolvedConfig)
        self.assertGreater(len(resolved.entries), 0)

    def test_known_paper_fact_values_load_correctly(self):
        resolved = self.mgr.load()
        self.assertEqual(resolved.get("model.gin.num_layers"), 4)
        self.assertEqual(resolved.get("model.gin.hidden_dim"), 16)
        self.assertAlmostEqual(resolved.get("model.graph.adjacency_threshold"), 0.8)
        self.assertEqual(resolved.get("optim.optimizer"), "Lion")
        self.assertAlmostEqual(resolved.get("optim.learning_rate"), 0.000325)
        self.assertAlmostEqual(resolved.get("loss.lambda1_recon"), 0.04)
        self.assertAlmostEqual(resolved.get("loss.lambda2_adv"), 0.04)
        self.assertAlmostEqual(resolved.get("loss.lambda3_energy"), 0.04)

    def test_load_with_dataset_merges_data_config(self):
        resolved = self.mgr.load(dataset="cifar100")
        self.assertEqual(resolved.get("data.cifar100.base_classes"), 60)
        self.assertEqual(resolved.get("data.cifar100.num_incremental_sessions"), 8)
        self.assertEqual(resolved.dataset, "cifar100")

    def test_load_unknown_dataset_raises(self):
        with self.assertRaises(Exception):
            self.mgr.load(dataset="not_a_real_dataset")

    def test_all_three_datasets_load(self):
        for ds in ["cifar100", "mini_imagenet", "cub200"]:
            resolved = self.mgr.load(dataset=ds)
            self.assertIn(f"data.{ds}.total_classes", resolved.entries)

    def test_targets_file_loaded_separately_and_not_merged(self):
        resolved = self.mgr.load()
        # targets must NOT leak into the trainable/resolved config
        self.assertFalse(any(k.startswith("targets.") for k in resolved.entries.keys()))
        targets = self.mgr.load_targets()
        self.assertIn("targets.cifar100.mean_accuracy_pct", targets)
        self.assertEqual(targets["targets.cifar100.mean_accuracy_pct"].value, 82.30)


class TestProvenanceTracking(unittest.TestCase):
    def setUp(self):
        self.mgr = ConfigManager(config_root=CONFIG_ROOT)
        self.resolved = self.mgr.load(dataset="cifar100")

    def test_every_entry_has_recognized_provenance(self):
        for key, entry in self.resolved.entries.items():
            self.assertIn(entry.provenance, PROVENANCE_TAGS, msg=f"key={key}")

    def test_paper_fact_tagged_correctly(self):
        self.assertEqual(self.resolved.provenance_of("model.gin.num_layers"), "PAPER_FACT")
        self.assertEqual(self.resolved.provenance_of("optim.optimizer"), "PAPER_FACT")

    def test_implementation_choice_blockers_tagged_correctly(self):
        # The five frozen blocker resolutions must be visibly distinguishable as
        # IMPLEMENTATION_CHOICE, never silently presented as PAPER_FACT.
        self.assertEqual(
            self.resolved.provenance_of("model.graph.num_nodes_mode"), "IMPLEMENTATION_CHOICE"
        )
        self.assertEqual(
            self.resolved.provenance_of("model.prompts.vision_prompt_insertion_mode"),
            "IMPLEMENTATION_CHOICE",
        )
        self.assertEqual(
            self.resolved.provenance_of("model.mlp_bridge.role"), "IMPLEMENTATION_CHOICE"
        )
        self.assertEqual(
            self.resolved.provenance_of("model.acga.acga_hgnec_data_wiring"),
            "IMPLEMENTATION_CHOICE",
        )
        self.assertEqual(
            self.resolved.provenance_of("model.acga.modality_scope"), "IMPLEMENTATION_CHOICE"
        )

    def test_unresolved_entries_have_null_value(self):
        for key, entry in self.resolved.entries.items():
            if entry.is_unresolved():
                self.assertIsNone(entry.value, msg=f"UNRESOLVED key {key} must have null value")

    def test_unresolved_keys_are_discoverable(self):
        unresolved = self.resolved.unresolved_keys()
        self.assertIn("model.hgn_ec.dt", unresolved)
        self.assertIn("model.acga.latent_dim_K", unresolved)

    def test_get_raises_on_unresolved_without_flag(self):
        with self.assertRaises(UnresolvedParameterError):
            self.resolved.get("model.hgn_ec.dt")

    def test_get_allows_unresolved_when_explicitly_requested(self):
        value = self.resolved.get("model.hgn_ec.dt", allow_unresolved=True)
        self.assertIsNone(value)

    def test_unresolved_entry_with_nonnull_value_is_rejected_at_parse_time(self):
        with self.assertRaises(ConfigParseError):
            ParamEntry(key="bad.key", value=1.0, provenance="UNRESOLVED")


class TestConfigValidation(unittest.TestCase):
    def setUp(self):
        self.mgr = ConfigManager(config_root=CONFIG_ROOT)

    def test_valid_config_passes(self):
        resolved = self.mgr.load(dataset="cifar100")
        warnings = self.mgr.validate(resolved)
        self.assertIsInstance(warnings, list)
        # UNRESOLVED entries should show up as warnings, not hard failures, by default.
        self.assertTrue(any("UNRESOLVED" in w for w in warnings))

    def test_missing_required_key_detected(self):
        resolved = self.mgr.load(dataset="cifar100")
        del resolved.entries["model.gin.num_layers"]
        with self.assertRaises(ConfigValidationError) as ctx:
            self.mgr.validate(resolved)
        self.assertIn("model.gin.num_layers", str(ctx.exception))

    def test_untagged_key_detected(self):
        resolved = self.mgr.load(dataset="cifar100")
        # Bypass ParamEntry's own __post_init__ guard to simulate a corrupted/untagged entry.
        entry = resolved.entries["model.gin.hidden_dim"]
        entry.provenance = ""
        with self.assertRaises(ConfigValidationError) as ctx:
            self.mgr.validate(resolved)
        self.assertIn("no provenance tag", str(ctx.exception))

    def test_invalid_provenance_value_detected(self):
        resolved = self.mgr.load(dataset="cifar100")
        entry = resolved.entries["model.gin.hidden_dim"]
        entry.provenance = "MADE_UP_TAG"
        with self.assertRaises(ConfigValidationError) as ctx:
            self.mgr.validate(resolved)
        self.assertIn("invalid provenance", str(ctx.exception))

    def test_strict_unresolved_mode_raises_when_unresolved_present(self):
        resolved = self.mgr.load(dataset="cifar100")
        with self.assertRaises(ConfigValidationError):
            self.mgr.validate(resolved, strict_unresolved=True)

    def test_strict_unresolved_mode_passes_when_no_unresolved(self):
        resolved = self.mgr.load(dataset="cifar100")
        # Simulate a fully-resolved config for this test only.
        for entry in resolved.entries.values():
            if entry.is_unresolved():
                entry.provenance = "IMPLEMENTATION_CHOICE"
                entry.value = "resolved_for_test"
        warnings = self.mgr.validate(resolved, strict_unresolved=True)
        self.assertEqual(warnings, [])


class TestConfigSaveLoad(unittest.TestCase):
    def setUp(self):
        self.mgr = ConfigManager(config_root=CONFIG_ROOT)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_reload_round_trip(self):
        resolved = self.mgr.load(dataset="cub200")
        resolved.meta = self.mgr.build_meta(seed=123, extra={"note": "unit test"})
        save_path = os.path.join(self.tmpdir, "config.yaml")
        self.mgr.save(resolved, save_path)
        self.assertTrue(os.path.isfile(save_path))

        reloaded = ConfigManager.load_saved(save_path)
        self.assertEqual(reloaded.dataset, "cub200")
        self.assertEqual(set(reloaded.entries.keys()), set(resolved.entries.keys()))
        for key in resolved.entries:
            self.assertEqual(reloaded.entries[key].value, resolved.entries[key].value)
            self.assertEqual(reloaded.entries[key].provenance, resolved.entries[key].provenance)
        self.assertEqual(reloaded.meta["seed"], 123)
        self.assertEqual(reloaded.meta["note"], "unit test")

    def test_reload_missing_file_raises(self):
        with self.assertRaises(Exception):
            ConfigManager.load_saved(os.path.join(self.tmpdir, "does_not_exist.yaml"))

    def test_reloaded_config_still_enforces_unresolved_guard(self):
        resolved = self.mgr.load(dataset="cifar100")
        save_path = os.path.join(self.tmpdir, "config.yaml")
        self.mgr.save(resolved, save_path)
        reloaded = ConfigManager.load_saved(save_path)
        with self.assertRaises(UnresolvedParameterError):
            reloaded.get("model.hgn_ec.dt")


if __name__ == "__main__":
    unittest.main()
