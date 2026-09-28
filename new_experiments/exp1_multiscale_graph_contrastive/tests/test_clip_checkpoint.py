"""
Stage 2 tests for CLIPWrapper checkpoint/state handling
(`CLIPWrapper.save_checkpoint` / `.load_checkpoint` / `.from_checkpoint`).

Kept in a separate file from `test_clip_wrapper.py` because it touches the filesystem
(temp dir), matching the pattern `test_config_tracking.py` uses for its save/load tests.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from models.clip import CLIPWrapper, build_mock_clip_config, build_mock_clip_wrapper  # noqa: E402
from models.clip.clip_wrapper import CLIPConfigError  # noqa: E402
from utils.seed import set_seed  # noqa: E402


class TestCheckpointRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="achg_clip_stage2_ckpt_")
        self.ckpt_path = os.path.join(self.tmp_dir, "clip_wrapper.pt")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_creates_file(self):
        model = build_mock_clip_wrapper()
        model.save_checkpoint(self.ckpt_path)
        self.assertTrue(os.path.isfile(self.ckpt_path))

    def test_load_into_same_architecture_restores_identical_output(self):
        set_seed(0, deterministic=False)
        model_a = build_mock_clip_wrapper()
        model_a.save_checkpoint(self.ckpt_path)

        # A freshly (differently) initialized model with the SAME config.
        set_seed(1, deterministic=False)
        model_b = CLIPWrapper(build_mock_clip_config())

        config = model_a.config
        patches = torch.randn(2, 3, config.patch_dim)
        tokens = torch.randint(0, config.vocab_size, (2, 3))

        out_a_before = model_a.encode_image(patches)
        out_b_before = model_b.encode_image(patches)
        # Different seeds -> different random init -> different output, sanity check.
        self.assertFalse(torch.allclose(out_a_before, out_b_before))

        model_b.load_checkpoint(self.ckpt_path)
        out_b_after = model_b.encode_image(patches)
        out_t_a = model_a.encode_text(tokens)
        out_t_b = model_b.encode_text(tokens)

        torch.testing.assert_close(out_a_before, out_b_after)
        torch.testing.assert_close(out_t_a, out_t_b)

    def test_load_checkpoint_returns_extra_meta(self):
        model = build_mock_clip_wrapper()
        model.save_checkpoint(self.ckpt_path, extra_meta={"session": 0, "note": "stage2-test"})

        model_b = CLIPWrapper(model.config)
        meta = model_b.load_checkpoint(self.ckpt_path)
        self.assertEqual(meta["session"], 0)
        self.assertEqual(meta["note"], "stage2-test")

    def test_from_checkpoint_reconstructs_wrapper(self):
        model = build_mock_clip_wrapper()
        model.save_checkpoint(self.ckpt_path)

        restored = CLIPWrapper.from_checkpoint(self.ckpt_path)
        self.assertIsInstance(restored, CLIPWrapper)
        self.assertEqual(restored.config.to_dict(), model.config.to_dict())

        patches = torch.randn(1, 2, model.config.patch_dim)
        torch.testing.assert_close(model.encode_image(patches), restored.encode_image(patches))

    def test_mismatched_config_refuses_to_load(self):
        model_a = build_mock_clip_wrapper(d_model=8, d_e=4)
        model_a.save_checkpoint(self.ckpt_path)

        model_b = CLIPWrapper(build_mock_clip_config(d_model=16, d_e=4))
        with self.assertRaises(CLIPConfigError):
            model_b.load_checkpoint(self.ckpt_path)

    def test_restored_wrapper_preserves_frozen_state(self):
        model = build_mock_clip_wrapper()  # frozen=True by default
        model.save_checkpoint(self.ckpt_path)
        restored = CLIPWrapper.from_checkpoint(self.ckpt_path)
        self.assertTrue(restored.is_frozen)


if __name__ == "__main__":
    unittest.main()
