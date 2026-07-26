import tempfile
import unittest
from pathlib import Path

import yaml

import infer_distilled
from distilled_config import (
    DistilledInferenceConfig,
    load_inference_config,
    profile_runtime_settings,
)


class DistilledConfigTest(unittest.TestCase):
    def _write_config(self, root: Path, **overrides):
        payload = {
            "profile": "standard",
            "schedule": [1000, 667, 333],
        }
        payload.update(overrides)
        path = root / "infer.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def test_public_config_is_small_and_valid(self):
        config = load_inference_config(infer_distilled.DEFAULT_CONFIG)
        self.assertEqual(config.profile, "standard")
        self.assertEqual(config.schedule, (1000, 667, 333))
        self.assertEqual(config.student_cfg_scale, 3.0)
        self.assertTrue(config.negative_prompt)
        self.assertEqual(config.dynamic_context_selection, "latest")
        self.assertFalse(config.nonlocal_memory_context)
        self.assertEqual(config.context_pose_pool_size, 5)

    def test_config_rejects_experiment_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(Path(tmp), source_run_dir="/tmp/run")
            with self.assertRaisesRegex(ValueError, "unknown inference config keys"):
                load_inference_config(path)

    def test_schedule_must_start_at_1000_and_descend(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(Path(tmp), schedule=[999, 500])
            with self.assertRaises(ValueError):
                load_inference_config(path)

    def test_hiar_scales_match_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(
                Path(tmp), profile="hiar-sde", hiar_scales=[1.0, 0.5]
            )
            with self.assertRaisesRegex(ValueError, "one value per schedule step"):
                load_inference_config(path)

    def test_non_hiar_profile_rejects_hiar_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(Path(tmp), hiar_scales=[1.0, 0.5, 0.0])
            with self.assertRaisesRegex(ValueError, "only valid"):
                load_inference_config(path)

    def test_context_selection_and_cfg_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_selection = self._write_config(
                root, dynamic_context_selection="global_oldest"
            )
            with self.assertRaisesRegex(ValueError, "dynamic_context_selection"):
                load_inference_config(bad_selection)
            bad_cfg = self._write_config(root, student_cfg_scale=0.5)
            with self.assertRaisesRegex(ValueError, "student_cfg_scale"):
                load_inference_config(bad_cfg)

    def test_profiles_map_to_only_their_intended_policy(self):
        standard = profile_runtime_settings(DistilledInferenceConfig())
        hiar = profile_runtime_settings(
            DistilledInferenceConfig(profile="hiar-sde")
        )
        sink = profile_runtime_settings(
            DistilledInferenceConfig(profile="sink-anchor-context")
        )
        self.assertEqual(standard["prefix_noise_mode"], "none")
        self.assertFalse(standard["force_original_anchor"])
        self.assertEqual(hiar["prefix_noise_mode"], "hiar_sde")
        self.assertTrue(hiar["noise_dynamic_context"])
        self.assertTrue(sink["force_original_anchor"])
        self.assertEqual(sink["prefix_noise_mode"], "none")


class PublicCliTest(unittest.TestCase):
    def test_cli_has_only_explicit_input_mode(self):
        options = {
            option
            for action in infer_distilled.build_parser()._actions
            for option in action.option_strings
        }
        for expected in ("--checkpoint", "--image", "--camera", "--output"):
            self.assertIn(expected, options)
        for removed in ("--source-run-dir", "--match-video", "--run-name"):
            self.assertNotIn(removed, options)

    def test_default_output_is_result_mp4(self):
        parser = infer_distilled.build_parser()
        output_action = next(
            action for action in parser._actions if action.dest == "output"
        )
        self.assertEqual(output_action.default, Path("result.mp4"))

    def test_output_must_be_mp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("checkpoint.safetensors", "input.png", "camera.npz"):
                (root / name).touch()
            args = infer_distilled.build_parser().parse_args(
                [
                    "--config", str(infer_distilled.DEFAULT_CONFIG),
                    "--checkpoint", str(root / "checkpoint.safetensors"),
                    "--image", str(root / "input.png"),
                    "--camera", str(root / "camera.npz"),
                    "--prompt", "test",
                    "--output", str(root / "result.json"),
                ]
            )
            with self.assertRaisesRegex(SystemExit, "must be an .mp4"):
                infer_distilled.validate_cli(args)


if __name__ == "__main__":
    unittest.main()
