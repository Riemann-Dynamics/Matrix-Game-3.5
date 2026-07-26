import unittest

import numpy as np

from examples.wanvideo.pipeline.mosaic.causal_memory import (
    _CausalKVDynamicContextPool,
    _causal_kv_context_pose_score,
)


def _pose(x):
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(x)
    return pose


class DynamicContextPoolTests(unittest.TestCase):
    def setUp(self):
        self.target = np.repeat(_pose(0.0)[None], 12, axis=0)
        self.entries = [
            {
                "name": "old-near",
                "source": "generated",
                "chunk_idx": 1,
                "position": 3,
                "camera_frame": 3,
                "rep_extrinsic": _pose(0.2),
                "source_timeline_position": 2,
            },
            {
                "name": "new-nearest",
                "source": "generated",
                "chunk_idx": 2,
                "position": 9,
                "camera_frame": 9,
                "rep_extrinsic": _pose(0.1),
                "source_timeline_position": 8,
            },
            {
                "name": "too-new",
                "source": "generated",
                "chunk_idx": 3,
                "position": 12,
                "camera_frame": 12,
                "rep_extrinsic": _pose(0.05),
                "source_timeline_position": 11,
            },
        ]

    def _pool(self, **kwargs):
        pool = _CausalKVDynamicContextPool(**kwargs)
        pool.generated_entries = list(self.entries)
        return pool

    def test_latest_without_nonlocal_matches_legacy_ranking(self):
        selected = self._pool(
            dynamic_context_selection_policy="latest"
        ).select_for_chunk(8, self.target)[0]
        legacy_selected = min(
            self.entries,
            key=lambda entry: (
                _causal_kv_context_pose_score(
                    entry["rep_extrinsic"], self.target
                ),
                0,
                -entry["position"],
            ),
        )
        self.assertIs(selected, legacy_selected)
        self.assertEqual(selected["name"], "too-new")

    def test_oldest_chooses_oldest_inside_pose_near_pool(self):
        selected = self._pool(
            dynamic_context_selection_policy="oldest",
            context_pose_pool_size=3,
        ).select_for_chunk(8, self.target)[0]
        self.assertEqual(selected["name"], "old-near")

    def test_nonlocal_filters_visible_prefix_before_oldest_selection(self):
        selected = self._pool(
            memory_context_selection_policy="nonlocal_oldest",
            dynamic_context_selection_policy="oldest",
            context_pose_pool_size=3,
        ).select_for_chunk(
            8,
            self.target,
            max_source_position_exclusive=8,
        )[0]
        self.assertEqual(selected["name"], "old-near")


if __name__ == "__main__":
    unittest.main()
