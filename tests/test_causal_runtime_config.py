from pathlib import Path
import unittest

from distilled_config import DistilledInferenceConfig
from examples.wanvideo.pipeline.mosaic.causal_config import build_runtime_args


class CausalRuntimeConfigTests(unittest.TestCase):
    def test_context_pool_does_not_change_mosaic_candidate_budget(self):
        config = DistilledInferenceConfig(context_pose_pool_size=9)
        args = build_runtime_args(
            config,
            checkpoint=Path("/tmp/model.safetensors"),
            dataset_index=Path("/tmp/data.sqlite"),
            workspace=Path("/tmp/workspace"),
            wan_dir=Path("/tmp/wan"),
            tokenizer_dir=Path("/tmp/tokenizer"),
            memory_cache_dir=Path("/tmp/cache"),
        )
        self.assertEqual(args.candidates_per_query_group_val, 5)
        self.assertEqual(args.causal_dynamic_context_pose_pool_size, 9)
        self.assertEqual(args.causal_memory_context_selection_policy, "legacy")
        self.assertEqual(args.causal_dynamic_context_selection_policy, "latest")
        self.assertTrue(args.causal_dmd_validation_use_cfg)
        self.assertEqual(args.validation_cfg_scale, 3.0)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.validation_seed, config.seed)
        self.assertTrue(args.vae_decode_tiled)
        self.assertEqual(args.vae_clean_context_blocks_max, 2)


if __name__ == "__main__":
    unittest.main()
