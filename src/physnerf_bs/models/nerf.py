from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .encodings import positional_encoding


@dataclass(slots=True)
class NerfRenderResult:
    rgb_coarse: torch.Tensor
    depth_coarse: torch.Tensor
    acc_coarse: torch.Tensor
    rgb_fine: torch.Tensor
    depth_fine: torch.Tensor
    acc_fine: torch.Tensor
    weights_fine: torch.Tensor | None = None
    t_fine: torch.Tensor | None = None


class PoseConditionedNeRF(nn.Module):
    def __init__(
        self,
        pose_dim: int,
        mode: str = "pose",
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        pose_freqs: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 6,
        skip_layer: int = 3,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        self.mode = mode
        self.pos_freqs = pos_freqs
        self.dir_freqs = dir_freqs
        self.pose_freqs = pose_freqs
        self.skip_layer = skip_layer
        self.pose_dim = pose_dim

        pos_in = 3 * (2 * pos_freqs + 1)
        dir_in = 3 * (2 * dir_freqs + 1)
        pose_in = 0 if mode == "static" else pose_dim * (2 * pose_freqs + 1)

        self.point_layers = nn.ModuleList()
        self.point_layers.append(nn.Linear(pos_in + pose_in, hidden_dim))
        for layer_idx in range(1, n_layers):
            in_dim = hidden_dim + pos_in + pose_in if layer_idx == skip_layer else hidden_dim
            self.point_layers.append(nn.Linear(in_dim, hidden_dim))

        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        self.color_layers = nn.Sequential(
            nn.Linear(hidden_dim + dir_in + pose_in, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xyz: torch.Tensor, ray_dirs: torch.Tensor, pose: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        n_rays, n_samples, _ = xyz.shape
        xyz_flat = xyz.reshape(-1, 3)
        dirs_flat = ray_dirs[:, None, :].expand(-1, n_samples, -1).reshape(-1, 3)
        pos_enc = positional_encoding(xyz_flat, self.pos_freqs)
        dir_enc = positional_encoding(dirs_flat, self.dir_freqs)

        if self.mode == "static":
            pose_enc = None
            h = pos_enc
        else:
            if pose is None:
                raise ValueError("PoseConditionedNeRF requires pose input in non-static mode.")
            pose_flat = pose[:, None, :].expand(-1, n_samples, -1).reshape(-1, self.pose_dim)
            pose_enc = positional_encoding(pose_flat, self.pose_freqs)
            h = torch.cat([pos_enc, pose_enc], dim=-1)

        inputs = h
        for idx, layer in enumerate(self.point_layers):
            if idx == self.skip_layer:
                h = torch.cat([h, inputs], dim=-1)
            h = self.relu(layer(h))

        sigma = self.sigma_head(h).reshape(n_rays, n_samples, 1)
        features = self.feature_head(h)
        if pose_enc is not None:
            rgb_inputs = torch.cat([features, dir_enc, pose_enc], dim=-1)
        else:
            rgb_inputs = torch.cat([features, dir_enc], dim=-1)
        rgb = self.color_layers(rgb_inputs).reshape(n_rays, n_samples, 3)
        return sigma, rgb


def sample_along_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_vals = torch.linspace(near, far, n_samples, device=ray_origins.device, dtype=ray_origins.dtype)
    t_vals = t_vals.unsqueeze(0).expand(ray_origins.shape[0], -1)
    if perturb:
        mids = 0.5 * (t_vals[:, 1:] + t_vals[:, :-1])
        lower = torch.cat([t_vals[:, :1], mids], dim=-1)
        upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
        t_vals = lower + (upper - lower) * torch.rand_like(t_vals)
    xyz = ray_origins[:, None, :] + ray_directions[:, None, :] * t_vals[..., None]
    return xyz, t_vals


def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, n_importance: int, deterministic: bool) -> torch.Tensor:
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
    if deterministic:
        u = torch.linspace(0.0, 1.0, n_importance, device=bins.device, dtype=bins.dtype).expand(cdf.shape[0], -1)
    else:
        u = torch.rand(cdf.shape[0], n_importance, device=bins.device, dtype=bins.dtype)
    idx = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below = torch.clamp_min(idx - 1, 0)
    above = torch.clamp_max(idx, cdf.shape[-1] - 1)
    cdf_below = torch.gather(cdf, 1, below)
    cdf_above = torch.gather(cdf, 1, above)
    bins_pad = torch.cat([bins[:, :1], bins], dim=-1)
    bins_below = torch.gather(bins_pad, 1, below)
    bins_above = torch.gather(bins_pad, 1, above)
    denom = (cdf_above - cdf_below).clamp_min(1e-5)
    return bins_below + (u - cdf_below) / denom * (bins_above - bins_below)


def volume_render(
    sigma: torch.Tensor,
    rgb: torch.Tensor,
    t_vals: torch.Tensor,
    ray_directions: torch.Tensor,
    white_background: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma = torch.relu(sigma.squeeze(-1))
    dists = t_vals[:, 1:] - t_vals[:, :-1]
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)
    dists = dists * torch.linalg.norm(ray_directions, dim=-1, keepdim=True)

    alpha = 1.0 - torch.exp(-sigma * dists)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    weights = alpha * trans

    rgb_map = torch.sum(weights[..., None] * rgb, dim=1)
    depth_map = torch.sum(weights * t_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)
    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map)[..., None]
    return rgb_map, depth_map, acc_map, weights


def render_rays_hierarchical(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    pose: torch.Tensor | None,
    model_coarse: PoseConditionedNeRF,
    model_fine: PoseConditionedNeRF | None,
    n_coarse: int,
    n_fine: int,
    near: float,
    far: float,
    perturb: bool,
    white_background: bool,
) -> NerfRenderResult:
    xyz_c, t_c = sample_along_rays(ray_origins, ray_directions, near, far, n_coarse, perturb=perturb)
    sigma_c, rgb_c = model_coarse(xyz_c, ray_directions, pose=pose)
    rgb_map_c, depth_c, acc_c, weights_c = volume_render(sigma_c, rgb_c, t_c, ray_directions, white_background)

    if model_fine is None or n_fine <= 0:
        return NerfRenderResult(
            rgb_coarse=rgb_map_c,
            depth_coarse=depth_c,
            acc_coarse=acc_c,
            rgb_fine=rgb_map_c,
            depth_fine=depth_c,
            acc_fine=acc_c,
            weights_fine=weights_c,
            t_fine=t_c,
        )

    mids = 0.5 * (t_c[:, 1:] + t_c[:, :-1])
    t_fine = sample_pdf(mids, weights_c[:, 1:-1].detach(), n_importance=n_fine, deterministic=not perturb)
    t_all = torch.sort(torch.cat([t_c, t_fine], dim=-1), dim=-1).values
    xyz_f = ray_origins[:, None, :] + ray_directions[:, None, :] * t_all[..., None]
    sigma_f, rgb_f = model_fine(xyz_f, ray_directions, pose=pose)
    rgb_map_f, depth_f, acc_f, weights_f = volume_render(sigma_f, rgb_f, t_all, ray_directions, white_background)
    return NerfRenderResult(
        rgb_coarse=rgb_map_c,
        depth_coarse=depth_c,
        acc_coarse=acc_c,
        rgb_fine=rgb_map_f,
        depth_fine=depth_f,
        acc_fine=acc_f,
        weights_fine=weights_f,
        t_fine=t_all,
    )


@torch.no_grad()
def render_image_grid(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    pose: torch.Tensor | None,
    render_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], NerfRenderResult],
    chunk_size: int,
    image_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {}
    for start in range(0, ray_origins.shape[0], chunk_size):
        end = min(start + chunk_size, ray_origins.shape[0])
        chunk_pose = None if pose is None else pose[start:end]
        result = render_fn(ray_origins[start:end], ray_directions[start:end], chunk_pose)
        chunk_dict = {
            "rgb": result.rgb_fine,
            "depth": result.depth_fine,
            "acc": result.acc_fine,
        }
        for key, value in chunk_dict.items():
            outputs.setdefault(key, []).append(value)
    h, w = image_hw
    return {
        "rgb": torch.cat(outputs["rgb"], dim=0).reshape(h, w, 3),
        "depth": torch.cat(outputs["depth"], dim=0).reshape(h, w),
        "acc": torch.cat(outputs["acc"], dim=0).reshape(h, w),
    }
