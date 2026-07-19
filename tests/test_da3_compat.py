import sys
import unittest
from pathlib import Path

import numpy as np


DA3_SRC = Path(__file__).resolve().parents[1] / "third_party/depth-anything-3/src"
sys.path.insert(0, str(DA3_SRC))

from addict import Dict
from depth_anything_3.utils import pose_align


class DA3CompatibilityTest(unittest.TestCase):
    def test_local_addict_compat_converts_nested_mappings(self):
        value = Dict({"model": {"width": 8}, "entries": [{"enabled": True}]})
        self.assertEqual(value.model.width, 8)
        self.assertTrue(value.entries[0].enabled)
        self.assertEqual(value.to_dict()["model"]["width"], 8)

    def test_numpy_umeyama_fallback_recovers_sim3(self):
        pose_est = np.repeat(np.eye(4)[None], 4, axis=0)
        pose_est[:, :3, 3] = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        scale = 2.5
        translation = np.array([4.0, -3.0, 2.0])
        pose_ref = pose_align._apply_sim3_to_poses(
            pose_est, rotation, translation, scale
        )

        original_evo = pose_align.PosePath3D
        pose_align.PosePath3D = None
        try:
            recovered_r, recovered_t, recovered_s, aligned = (
                pose_align._umeyama_sim3_from_paths(pose_ref, pose_est)
            )
        finally:
            pose_align.PosePath3D = original_evo

        self.assertTrue(np.allclose(recovered_r, rotation, atol=1e-7))
        self.assertTrue(np.allclose(recovered_t, translation, atol=1e-7))
        self.assertAlmostEqual(recovered_s, scale, places=7)
        self.assertTrue(np.allclose(aligned, pose_ref, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
