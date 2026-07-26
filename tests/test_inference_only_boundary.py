import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_FILES = (
    ROOT / "distilled_config.py",
    ROOT / "infer_distilled.py",
    ROOT / "tools/run_distilled_inference.py",
    ROOT / "examples/wanvideo/pipeline/mosaic/causal_config.py",
    ROOT / "examples/wanvideo/pipeline/mosaic/causal_inference.py",
    ROOT / "examples/wanvideo/pipeline/mosaic/causal_memory.py",
    ROOT / "examples/wanvideo/pipeline/mosaic/causal_rollout.py",
    ROOT / "diffsynth/core/data/nonlocal_memory_context.py",
)
REMOVED_TRAINING_FILES = (
    ROOT / "examples/wanvideo/model_training/train_mosaic.py",
    ROOT / "diffsynth/diffusion/causal_dmd_loss.py",
    ROOT / "diffsynth/diffusion/causal_loss.py",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "examples.wanvideo.model_training",
    "diffsynth.diffusion.causal_dmd_loss",
    "diffsynth.diffusion.causal_loss",
)


class InferenceOnlyBoundaryTest(unittest.TestCase):
    def test_training_files_are_not_shipped(self):
        for path in REMOVED_TRAINING_FILES:
            with self.subTest(path=path):
                self.assertFalse(path.exists())

    def test_inference_modules_do_not_import_training_stack(self):
        for path in INFERENCE_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            with self.subTest(path=path):
                self.assertFalse(
                    any(
                        name.startswith(prefix)
                        for name in imports
                        for prefix in FORBIDDEN_IMPORT_PREFIXES
                    ),
                    imports,
                )

    def test_inference_modules_define_no_training_step_or_loss(self):
        forbidden_names = {"training_step", "train_step", "backward", "build_optimizer"}
        for path in INFERENCE_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            definitions = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            with self.subTest(path=path):
                self.assertTrue(forbidden_names.isdisjoint(definitions), definitions)

    def test_experiment_replay_code_is_not_shipped(self):
        forbidden = (
            "source_run_dir",
            "args_resolved.json",
            "match_validation_video",
            "validation_manifest_path",
            "_sections.json",
        )
        merged = "\n".join(
            path.read_text(encoding="utf-8") for path in INFERENCE_FILES
        )
        for marker in forbidden:
            with self.subTest(marker=marker):
                self.assertNotIn(marker, merged)


if __name__ == "__main__":
    unittest.main()
