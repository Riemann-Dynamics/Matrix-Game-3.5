import unittest

import torch

from examples.wanvideo.pipeline.mosaic.module_materialize import (
    _MaterializeMixin,
)


class DistilledReferenceAlignmentTests(unittest.TestCase):
    def test_uint8_frames_normalize_before_bfloat16_cast(self):
        frames = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
        actual = _MaterializeMixin._frames_to_vae_input(
            frames,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            pixel_range="uint8_or_255",
        )
        expected = frames.float().mul(1.0 / 127.5).sub(1.0).to(torch.bfloat16)
        cast_first = frames.to(torch.bfloat16).mul(1.0 / 127.5).sub(1.0)

        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, cast_first))


if __name__ == "__main__":
    unittest.main()
