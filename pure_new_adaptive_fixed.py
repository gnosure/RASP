import torch

from pure_new import Purifica as BasePurifica


class Purifica(BasePurifica):
    """Purification model with a stable, directionally correct adaptive radius."""

    def get_adaptive_radius(
        self,
        base_radius,
        amplitude_ref,
        phase_ref,
        amplitude_pred,
        phase_pred,
    ):
        batch, channel, height, width = amplitude_ref.shape
        base_radius = torch.full(
            (batch,),
            float(base_radius),
            device=amplitude_ref.device,
            dtype=amplitude_ref.dtype,
        )
        if not self.use_adaptive_radius:
            return base_radius

        max_radius_mask = self.build_batch_frequency_mask(
            height, width, self.radius_max, amplitude_ref.device
        )
        num_values = (
            max_radius_mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * channel
        )

        log_amplitude_ref = torch.log1p(amplitude_ref)
        log_amplitude_pred = torch.log1p(amplitude_pred)
        amplitude_diff = torch.abs(log_amplitude_pred - log_amplitude_ref)
        phase_diff = torch.abs(
            self.wrap_phase(phase_pred - phase_ref)
        ) / torch.pi

        amplitude_scale = (
            (log_amplitude_ref.abs() * max_radius_mask).sum(dim=(1, 2, 3))
            / num_values
        ).clamp_min(1e-3)
        amplitude_score = (
            (amplitude_diff * max_radius_mask).sum(dim=(1, 2, 3))
            / num_values
        ) / amplitude_scale
        phase_score = (
            (phase_diff * max_radius_mask).sum(dim=(1, 2, 3))
            / num_values
        )

        score_weight = max(1e-8, self.lambda_amp + self.lambda_phase)
        frequency_score = (
            self.lambda_amp * amplitude_score
            + self.lambda_phase * phase_score
        ) / score_weight

        temperature = max(1e-8, self.adaptive_radius_temperature)
        expansion = torch.sigmoid(
            (frequency_score - self.adaptive_radius_threshold) / temperature
        )
        spectral_radius = self.radius_min + (
            self.radius_max - self.radius_min
        ) * expansion
        radius = (
            (1.0 - self.adaptive_radius_mix) * base_radius
            + self.adaptive_radius_mix * spectral_radius
        )
        return torch.clamp(
            radius, min=self.radius_min, max=self.radius_max
        )
