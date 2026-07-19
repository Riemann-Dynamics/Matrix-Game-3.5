import unittest

from frustum.frustum_handler import _is_da3_compatible_estimator


class _MarkedAdapter:
    da3_compatible = True

    def inference(self):
        pass


class _VideoDepthAnything:
    def infer_video_depth(self):
        pass


class DepthAnything3:
    def inference(self):
        pass


# Mirror the upstream class identity without importing its optional stack.
DepthAnything3.__module__ = "depth_anything_3.api"


class DA3DetectionTest(unittest.TestCase):
    def test_upstream_da3_is_recognized_after_late_import(self):
        self.assertTrue(_is_da3_compatible_estimator(DepthAnything3()))

    def test_explicit_compatible_adapter_is_recognized(self):
        self.assertTrue(_is_da3_compatible_estimator(_MarkedAdapter()))

    def test_legacy_video_depth_estimator_is_not_da3(self):
        self.assertFalse(_is_da3_compatible_estimator(_VideoDepthAnything()))


if __name__ == "__main__":
    unittest.main()
