from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import PhysicsConfig


@dataclass(slots=True)
class PhysicsLossOutput:
    total: torch.Tensor
    depth_loss: torch.Tensor
    silhouette_loss: torch.Tensor
    energy_loss: torch.Tensor


class HybridHamiltonianLoss:
    def __init__(self, config: PhysicsConfig) -> None:
        self.config = config

    def __call__(
        self,
        predicted_depth: torch.Tensor,
        predicted_acc: torch.Tensor,
        target_depth: torch.Tensor,
        target_mask: torch.Tensor,
        teacher_total_energy: torch.Tensor,
    ) -> PhysicsLossOutput:
        target_mask = target_mask.float()
        fg_weight = target_mask + 0.1
        depth_loss = ((predicted_depth - target_depth).abs() * fg_weight).mean()
        silhouette_loss = F.binary_cross_entropy(predicted_acc.clamp(1e-5, 1 - 1e-5), target_mask.clamp(0.0, 1.0))

        depth_proxy = 1.0 / predicted_depth.clamp_min(1e-2)
        proxy_energy = self.config.proxy_gravity_scale * depth_proxy
        teacher_norm = teacher_total_energy / teacher_total_energy.detach().mean().clamp_min(1e-5)
        proxy_norm = proxy_energy / proxy_energy.detach().mean().clamp_min(1e-5)
        energy_loss = F.mse_loss(proxy_norm, teacher_norm)

        total = (
            self.config.depth_weight * depth_loss
            + self.config.silhouette_weight * silhouette_loss
            + self.config.energy_weight * energy_loss
        ) * self.config.weight
        return PhysicsLossOutput(
            total=total,
            depth_loss=depth_loss,
            silhouette_loss=silhouette_loss,
            energy_loss=energy_loss,
        )
