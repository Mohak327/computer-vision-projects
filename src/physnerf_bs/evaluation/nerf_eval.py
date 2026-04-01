from __future__ import annotations

from typing import Any

import torch

from ..config import EvaluationConfig, NerfConfig, PhysicsConfig
from ..data.datasets import FrameDataset, _pixels_to_rays
from ..models.nerf import PoseConditionedNeRF, render_image_grid, render_rays_hierarchical
from ..physics.hamiltonian import HybridHamiltonianLoss
from .metrics import compute_lpips_like, compute_psnr, compute_ssim


@torch.no_grad()
def evaluate_nerf_model(
    model_coarse: PoseConditionedNeRF,
    model_fine: PoseConditionedNeRF,
    dataset: FrameDataset,
    nerf_config: NerfConfig,
    physics_config: PhysicsConfig,
    eval_config: EvaluationConfig,
    device: str,
    limit: int = 2,
) -> dict[str, Any]:
    physics_loss = HybridHamiltonianLoss(physics_config)
    metrics: list[dict[str, float]] = []

    for index in range(min(limit, len(dataset))):
        item = dataset[index]
        rgb = item["rgb"].to(device)
        depth = item["depth"].to(device)
        mask = item["segmentation"].to(device)
        pose = None if nerf_config.mode == "static" else item["qpos"].to(device)
        h, w = rgb.shape[:2]
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        uv = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).float().to(device)
        ray_o, ray_d = _pixels_to_rays(item["intrinsics"].to(device), item["extrinsics"].to(device), uv)
        pose_grid = None if pose is None else pose.unsqueeze(0).expand(ray_o.shape[0], -1)

        def render_fn(chunk_o: torch.Tensor, chunk_d: torch.Tensor, chunk_pose: torch.Tensor | None):
            return render_rays_hierarchical(
                chunk_o,
                chunk_d,
                pose=chunk_pose,
                model_coarse=model_coarse,
                model_fine=model_fine,
                n_coarse=nerf_config.n_coarse,
                n_fine=nerf_config.n_fine,
                near=nerf_config.near,
                far=nerf_config.far,
                perturb=False,
                white_background=nerf_config.white_background,
            )

        render = render_image_grid(ray_o, ray_d, pose_grid, render_fn, nerf_config.chunk_size, (h, w))
        physics = physics_loss(
            predicted_depth=render["depth"].reshape(-1),
            predicted_acc=render["acc"].reshape(-1),
            target_depth=depth.reshape(-1),
            target_mask=mask.reshape(-1),
            teacher_total_energy=item["physics_total_energy"].reshape(1).expand(h * w).to(device),
        )
        occluded_mask = ((1.0 - mask) * (depth > 0).float()).bool()
        occluded_limb = torch.tensor(0.0, device=device)
        if occluded_mask.any():
            occluded_limb = torch.mean((render["rgb"][occluded_mask] - rgb[occluded_mask]).abs())
        metrics.append(
            {
                "psnr": compute_psnr(render["rgb"].cpu(), rgb.cpu()),
                "ssim": compute_ssim(render["rgb"].cpu(), rgb.cpu()),
                "lpips": compute_lpips_like(render["rgb"].cpu(), rgb.cpu()),
                "energy_consistency": float(physics.energy_loss.item()),
                "occluded_limb_error": float(occluded_limb.item()),
            }
        )

    return {
        key: float(sum(item[key] for item in metrics) / max(len(metrics), 1))
        for key in metrics[0].keys()
    } if metrics else {}
