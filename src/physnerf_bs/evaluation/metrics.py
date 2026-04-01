from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.float(), target.float()).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).item())


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.float()
    target = target.float()
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var(unbiased=False)
    sigma_y = target.var(unbiased=False)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / denominator.clamp_min(1e-12)).item())


def compute_lpips_like(pred: torch.Tensor, target: torch.Tensor) -> float:
    try:
        import lpips  # type: ignore
    except ImportError:
        return float(F.l1_loss(pred.float(), target.float()).item())

    model = lpips.LPIPS(net="alex")
    pred_n = pred.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    target_n = target.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    with torch.no_grad():
        return float(model(pred_n, target_n).mean().item())


def compute_body_selectivity(latent_states: torch.Tensor, masks: torch.Tensor) -> dict[str, float]:
    if latent_states.ndim != 5:
        raise ValueError("latent_states must have shape [B, T, C, H, W].")
    if masks.ndim != 5:
        raise ValueError("masks must have shape [B, T, 1, H, W].")

    body_mask = masks.expand(-1, -1, latent_states.shape[2], -1, -1)
    bg_mask = 1.0 - body_mask

    body_sum = (latent_states * body_mask).sum(dim=(0, 1, 3, 4))
    bg_sum = (latent_states * bg_mask).sum(dim=(0, 1, 3, 4))
    body_count = body_mask.sum(dim=(0, 1, 3, 4)).clamp_min(1.0)
    bg_count = bg_mask.sum(dim=(0, 1, 3, 4)).clamp_min(1.0)

    body_mean = body_sum / body_count
    bg_mean = bg_sum / bg_count
    selectivity = (body_mean - bg_mean).abs() / (body_mean.abs() + bg_mean.abs() + 1e-6)
    body_preferred = (body_mean > bg_mean).float().mean()

    return {
        "mean_selectivity": float(selectivity.mean().item()),
        "max_selectivity": float(selectivity.max().item()),
        "body_preferred_fraction": float(body_preferred.item()),
    }
