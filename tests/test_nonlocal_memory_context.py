import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "diffsynth"
    / "core"
    / "data"
    / "nonlocal_memory_context.py"
)
SPEC = importlib.util.spec_from_file_location("nonlocal_memory_context", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RollingAnchorTests(unittest.TestCase):
    def test_standard_seven_chunk_window(self):
        kwargs = {"frames_per_chunk": 3, "window_chunks": 7}
        self.assertEqual(
            MODULE.rolling_anchor_source_position(local_chunk_idx=0, **kwargs), 0
        )
        self.assertEqual(
            MODULE.rolling_anchor_source_position(local_chunk_idx=6, **kwargs), 0
        )
        self.assertEqual(
            MODULE.rolling_anchor_source_position(local_chunk_idx=7, **kwargs), 3
        )
        self.assertEqual(
            MODULE.rolling_anchor_source_position(local_chunk_idx=8, **kwargs), 6
        )

    def test_nonlocal_prefix_excludes_boundary_and_visible_window(self):
        positions = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        self.assertEqual(
            MODULE.strictly_nonlocal_prefix_count(
                positions,
                max_source_position_exclusive=3,
                expected_count=len(positions),
            ),
            7,
        )


class ContextSelectionTests(unittest.TestCase):
    @staticmethod
    def _pose_score(rep, target):
        return abs(float(rep) - float(target))

    def test_pose_near_pool_then_oldest(self):
        candidates = [
            {
                "name": "globally_old_but_far",
                "source": "history",
                "rep_extrinsic": 100.0,
                "source_timeline_position": -20,
                "position": 1,
            },
            {
                "name": "nearest_recent",
                "source": "generated",
                "rep_extrinsic": 0.1,
                "source_timeline_position": 8,
                "position": 9,
            },
            {
                "name": "second_nearest_older",
                "source": "generated",
                "rep_extrinsic": 0.2,
                "source_timeline_position": 2,
                "position": 3,
            },
        ]
        selected = MODULE.select_pose_near_oldest(
            candidates,
            target_extrinsics=0.0,
            pose_score_fn=self._pose_score,
            pose_pool_size=2,
        )
        self.assertEqual(selected["name"], "second_nearest_older")

    def test_policy_normalization_is_explicit(self):
        self.assertEqual(
            MODULE.normalize_memory_context_selection_policy("nonlocal-oldest"),
            "nonlocal_oldest",
        )
        self.assertEqual(
            MODULE.normalize_dynamic_context_selection_policy("oldest"), "oldest"
        )
        with self.assertRaises(ValueError):
            MODULE.normalize_dynamic_context_selection_policy("global_oldest")


if __name__ == "__main__":
    unittest.main()
