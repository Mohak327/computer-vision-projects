from __future__ import annotations

from typing import Callable

import torch


def sample_along_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 3D points along rays.

    Args:
        ray_origins: (N, 3)
        ray_directions: (N, 3)
        near: near bound
        far: far bound
        n_samples: samples per ray
        perturb: stratified jitter in each interval

    Returns:
        xyz_samples: (N, n_samples, 3)
        t_vals: (N, n_samples)
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")

    device = ray_origins.device
    t_lin = torch.linspace(
        near,
        far,
        n_samples,
        device=device,
        dtype=ray_origins.dtype,
    )
    t_vals = t_lin.unsqueeze(0).expand(ray_origins.shape[0], -1)

    if perturb:
        mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
        lower = torch.cat([t_vals[:, :1], mids], dim=-1)
        upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand

    xyz_samples = ray_origins[:, None, :] + ray_directions[:, None, :] * t_vals[..., None]
    return xyz_samples, t_vals


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_importance: int,
    deterministic: bool = False,
) -> torch.Tensor:
    """Importance sample using inverse-CDF from piecewise-constant PDF.

    Args:
        bins: (N, M) interval boundaries (usually mids of coarse t values)
        weights: (N, M) non-negative weights for PDF
        n_importance: number of samples to draw
        deterministic: use deterministic uniformly-spaced samples

    Returns:
        samples: (N, n_importance)
    """
    if n_importance <= 0:
        raise ValueError("n_importance must be > 0")

    eps = 1e-5
    weights = weights + eps
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # (N, M+1)

    if deterministic:
        u = torch.linspace(0.0, 1.0, n_importance, device=bins.device, dtype=bins.dtype)
        u = u.expand(cdf.shape[0], -1)
    else:
        u = torch.rand(cdf.shape[0], n_importance, device=bins.device, dtype=bins.dtype)

    cdf = cdf.contiguous()
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1] - 1)

    cdf_g0 = torch.gather(cdf, 1, below)
    cdf_g1 = torch.gather(cdf, 1, above)

    bins_pad = torch.cat([bins[:, :1], bins], dim=-1)  # (N, M+1)
    bins_g0 = torch.gather(bins_pad, 1, below)
    bins_g1 = torch.gather(bins_pad, 1, above)

    denom = torch.where((cdf_g1 - cdf_g0) < eps, torch.ones_like(cdf_g1), (cdf_g1 - cdf_g0))
    t = (u - cdf_g0) / denom
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples


def volume_render(
    sigmas: torch.Tensor,
    rgbs: torch.Tensor,
    t_vals: torch.Tensor,
    ray_directions: torch.Tensor,
    white_bkgd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite colors and depth along rays.

    Args:
        sigmas: (N, S, 1)
        rgbs: (N, S, 3)
        t_vals: (N, S)
        ray_directions: (N, 3)
        white_bkgd: add white background color when opacity < 1

    Returns:
        rgb_map: (N, 3)
        depth_map: (N,)
        acc_map: (N,)
        weights: (N, S)
    """
    sigma = torch.relu(sigmas.squeeze(-1))

    dists = t_vals[:, 1:] - t_vals[:, :-1]
    infinity_pad = torch.full_like(dists[:, :1], 1e10)
    dists = torch.cat([dists, infinity_pad], dim=-1)

    ray_norms = torch.linalg.norm(ray_directions, dim=-1, keepdim=True)
    dists = dists * ray_norms

    alpha = 1.0 - torch.exp(-sigma * dists)
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )
    trans = trans[:, :-1]
    weights = alpha * trans

    rgb_map = torch.sum(weights[..., None] * rgbs, dim=1)
    depth_map = torch.sum(weights * t_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map)[..., None]

    return rgb_map, depth_map, acc_map, weights


def render_rays_hierarchical(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    model_coarse: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    model_fine: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None,
    n_coarse: int,
    n_fine: int,
    near: float,
    far: float,
    perturb: bool,
    white_bkgd: bool,
) -> dict[str, torch.Tensor]:
    """Render rays with coarse and optional fine importance sampling."""
    param = next(model_coarse.parameters())
    ray_origins = ray_origins.to(device=param.device, dtype=param.dtype)
    ray_directions = ray_directions.to(device=param.device, dtype=param.dtype)

    xyz_c, t_c = sample_along_rays(
        ray_origins,
        ray_directions,
        near=near,
        far=far,
        n_samples=n_coarse,
        perturb=perturb,
    )
    sigma_c, rgb_c = model_coarse(xyz_c, ray_directions)
    rgb_map_c, depth_map_c, acc_map_c, weights_c = volume_render(
        sigma_c,
        rgb_c,
        t_c,
        ray_directions,
        white_bkgd=white_bkgd,
    )

    out: dict[str, torch.Tensor] = {
        "rgb_coarse": rgb_map_c,
        "depth_coarse": depth_map_c,
        "acc_coarse": acc_map_c,
        "weights_coarse": weights_c,
        "t_coarse": t_c,
    }

    if model_fine is None or n_fine <= 0:
        out["rgb_fine"] = rgb_map_c
        out["depth_fine"] = depth_map_c
        out["acc_fine"] = acc_map_c
        return out

    t_mids = 0.5 * (t_c[:, :-1] + t_c[:, 1:])
    # Avoid boundary spikes when building PDF.
    pdf_weights = weights_c[:, 1:-1].detach() + 1e-5
    t_fine = sample_pdf(t_mids, pdf_weights, n_importance=n_fine, deterministic=not perturb)

    t_all = torch.cat([t_c, t_fine], dim=-1)
    t_all, _ = torch.sort(t_all, dim=-1)
    xyz_f = ray_origins[:, None, :] + ray_directions[:, None, :] * t_all[..., None]

    sigma_f, rgb_f = model_fine(xyz_f, ray_directions)
    rgb_map_f, depth_map_f, acc_map_f, weights_f = volume_render(
        sigma_f,
        rgb_f,
        t_all,
        ray_directions,
        white_bkgd=white_bkgd,
    )

    out.update(
        {
            "rgb_fine": rgb_map_f,
            "depth_fine": depth_map_f,
            "acc_fine": acc_map_f,
            "weights_fine": weights_f,
            "t_fine": t_all,
        }
    )
    return out


@torch.no_grad()
def render_image_by_chunks(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    render_fn: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    """Render many rays in chunks to avoid OOM.

    Args:
        ray_origins: (N, 3)
        ray_directions: (N, 3)
        render_fn: callable taking chunked origins/directions and returning dict outputs
        chunk_size: number of rays per chunk

    Returns:
        Dict with concatenated tensor outputs.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    outputs: dict[str, list[torch.Tensor]] = {}
    n_rays = ray_origins.shape[0]

    for start in range(0, n_rays, chunk_size):
        end = min(start + chunk_size, n_rays)
        chunk_out = render_fn(ray_origins[start:end], ray_directions[start:end])
        for key, val in chunk_out.items():
            outputs.setdefault(key, []).append(val)

    return {key: torch.cat(val_list, dim=0) for key, val_list in outputs.items()}
