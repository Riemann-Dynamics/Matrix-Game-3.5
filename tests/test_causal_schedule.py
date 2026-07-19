import unittest

import torch

from diffsynth.inference.causal_schedule import (
    hiar_sde_corrupt_clean_latents,
    resolve_wrapped_timesteps,
    validate_causal_dmd_denoising_schedule,
)


class _Scheduler:
    def __init__(self):
        self.training = True
        self.linear_timesteps_weights = torch.tensor([3.0])
        self.set_timesteps(1000, training=False)

    def set_timesteps(self, count, training=False):
        self.timesteps = torch.arange(count, 0, -1, dtype=torch.float32)
        self.sigmas = self.timesteps / count
        self.training = training

    def add_noise(self, clean, noise, timestep):
        return clean + noise


class CausalScheduleTest(unittest.TestCase):
    def test_stage3_schedule_contract(self):
        self.assertEqual(
            validate_causal_dmd_denoising_schedule((1000, 667, 333), num_steps=3),
            (1000, 667, 333),
        )

    def test_schedule_must_start_at_1000_and_descend(self):
        for schedule in ((999, 500), (1000, 500, 750)):
            with self.subTest(schedule=schedule):
                with self.assertRaises(ValueError):
                    validate_causal_dmd_denoising_schedule(schedule)

    def test_wrapped_ids_select_expected_scheduler_rows(self):
        timesteps, sigmas = resolve_wrapped_timesteps(
            _Scheduler(), (1000, 667, 333)
        )
        self.assertTrue(torch.equal(timesteps, torch.tensor([1000.0, 667.0, 333.0])))
        self.assertTrue(torch.allclose(sigmas, torch.tensor([1.0, 0.667, 0.333])))

    def test_hiar_corruption_preserves_anchor_and_interpolates(self):
        clean = torch.zeros((1, 1, 3, 1, 1))
        noise = torch.ones_like(clean)
        result = hiar_sde_corrupt_clean_latents(
            _Scheduler(),
            clean,
            667,
            keep_first_clean=True,
            corruption_scale=0.5,
            noise=noise,
        )
        self.assertEqual(result[0, 0, 0, 0, 0].item(), 0.0)
        self.assertTrue(torch.equal(result[:, :, 1:], torch.full_like(result[:, :, 1:], 0.5)))

    def test_hiar_rejects_bad_scale_or_noise_shape(self):
        clean = torch.zeros((1, 1, 2, 1, 1))
        with self.assertRaises(ValueError):
            hiar_sde_corrupt_clean_latents(
                _Scheduler(), clean, 667, keep_first_clean=False, corruption_scale=1.1
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            hiar_sde_corrupt_clean_latents(
                _Scheduler(),
                clean,
                667,
                keep_first_clean=False,
                noise=torch.zeros((1,)),
            )


if __name__ == "__main__":
    unittest.main()
