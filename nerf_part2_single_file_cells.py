# %% [markdown]
# NeRF Part 2 Single-File Notebook
#
# Run these cells top to bottom. Train first, then render RGB, depth, or both.

# %%
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

# %%
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_training_curves(
    loss_hist: list[float],
    eval_steps: list[int],
    val_psnr_hist: list[float],
    output_dir: str | Path,
) -> None:
    output_dir = ensure_dir(output_dir)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(loss_hist)
    ax.set_title("Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    if eval_steps and val_psnr_hist:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(eval_steps, val_psnr_hist)
        ax.set_title("Validation PSNR")
        ax.set_xlabel("Step")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "psnr_curve.png", dpi=150)
        plt.close(fig)


def save_rgb_png(image: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(image, 0.0, 1.0)
    img_uint8 = (img * 255.0).round().astype(np.uint8)
    Image.fromarray(img_uint8, mode="RGB").save(output_path)


def save_depth_png(depth: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    d = depth.astype(np.float32)
    d_min = float(np.min(d))
    d_max = float(np.max(d))
    if d_max > d_min:
        d = (d - d_min) / (d_max - d_min)
    else:
        d = np.zeros_like(d)

    d_rgba = plt.cm.inferno(d)
    d_rgb_uint8 = (d_rgba[..., :3] * 255.0).round().astype(np.uint8)
    Image.fromarray(d_rgb_uint8, mode="RGB").save(output_path)

# %%
def load_data(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    data = np.load(data_path)
    images_train = data["images_train"] / 255.0
    c2ws_train = data["c2ws_train"]
    images_val = data["images_val"] / 255.0
    c2ws_val = data["c2ws_val"]
    c2ws_test = data["c2ws_test"]
    focal = data["focal"]

    h, w = images_train.shape[1], images_train.shape[2]
    o_x = w / 2
    o_y = h / 2
    k = torch.as_tensor([[focal.item(), 0, o_x], [0, focal.item(), o_y], [0, 0, 1]])
    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, k


def pixels_to_rays(
    k: torch.Tensor,
    c2w: torch.Tensor,
    uvs: torch.Tensor,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    k = k.to(device)
    c2w = c2w.to(torch.float64).to(device)
    uvs = uvs.to(device)

    num_pixels = uvs.shape[0]
    r = c2w[:3, :3]
    r_os = c2w[:3, 3].unsqueeze(0).expand(num_pixels, -1)

    homog_uvs = torch.hstack((uvs, torch.ones(num_pixels, 1, device=device)))
    k_inv = torch.linalg.inv(k).to(device)
    dirs = ((r @ k_inv.to(torch.float64)) @ homog_uvs.to(torch.float64).T).T
    r_ds = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True)
    return r_os, r_ds


def image_to_rays(
    image: torch.Tensor,
    c2w: torch.Tensor,
    k: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    image = image.to(device)
    h, w = image.shape[:2]
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    uvs = torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=-1).to(torch.float32)
    r_os, r_ds = pixels_to_rays(k=k, c2w=c2w, uvs=uvs, device=device)
    return torch.cat((r_os, r_ds), dim=1).reshape(h, w, 6)


def images_to_rays(
    images: torch.Tensor,
    c2ws: torch.Tensor,
    k: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    rays_per_image = [image_to_rays(images[i], c2ws[i], k, device=device) for i in range(images.shape[0])]
    return torch.stack(rays_per_image, dim=0)


class RaysData(Dataset):
    def __init__(
        self,
        images: torch.Tensor,
        k: torch.Tensor,
        c2ws: torch.Tensor,
        device: str = "cuda",
    ) -> None:
        self.images = torch.as_tensor(images, dtype=torch.float32, device=device)
        self.k = torch.as_tensor(k, dtype=torch.float32, device=device)
        self.c2ws = torch.as_tensor(c2ws, dtype=torch.float32, device=device)
        self.h, self.w = self.images.shape[1:3]
        self.num_images = self.images.shape[0]

        ys, xs = torch.meshgrid(
            torch.arange(self.h, device=device),
            torch.arange(self.w, device=device),
            indexing="ij",
        )
        single_image_uvs = torch.stack((xs.reshape(-1), ys.reshape(-1)), dim=-1)
        self.uvs = single_image_uvs.repeat(self.num_images, 1)

        rays = images_to_rays(self.images, self.c2ws, self.k, device=device)
        self.rays_o = rays[..., :3].reshape(-1, 3)
        self.rays_d = rays[..., 3:].reshape(-1, 3)
        self.gt_rgbs = self.images.reshape(-1, 3)

    def __len__(self) -> int:
        return self.num_images * self.h * self.w

    def sample_rays(self, num_rays: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_indices = torch.randint(0, len(self), (num_rays,), device=self.rays_o.device)
        return (
            self.rays_o[sample_indices],
            self.rays_d[sample_indices],
            self.gt_rgbs[sample_indices],
        )

# %%
def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    if num_freqs <= 0:
        return x
    enc = [x]
    freq_bands = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)
    for freq in freq_bands:
        enc.append(torch.sin(freq * x))
        enc.append(torch.cos(freq * x))
    return torch.cat(enc, dim=-1)


class NeRFMLP(nn.Module):
    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 8,
        skip_layer: int = 4,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")

        self.pos_freqs = pos_freqs
        self.dir_freqs = dir_freqs
        self.skip_layer = skip_layer

        pos_in_dim = 3 * (2 * pos_freqs + 1)
        dir_in_dim = 3 * (2 * dir_freqs + 1)

        self.pts_layers = nn.ModuleList()
        self.pts_layers.append(nn.Linear(pos_in_dim, hidden_dim))
        for i in range(1, n_layers):
            in_dim = hidden_dim + pos_in_dim if i == skip_layer else hidden_dim
            self.pts_layers.append(nn.Linear(in_dim, hidden_dim))

        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        self.color_layers = nn.Sequential(
            nn.Linear(hidden_dim + dir_in_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xyz: torch.Tensor, ray_dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        param_dtype = self.pts_layers[0].weight.dtype
        param_device = self.pts_layers[0].weight.device
        xyz = xyz.to(device=param_device, dtype=param_dtype)
        ray_dirs = ray_dirs.to(device=param_device, dtype=param_dtype)

        n_rays, n_samples, _ = xyz.shape
        xyz_flat = xyz.reshape(-1, 3)
        dirs_expanded = ray_dirs[:, None, :].expand(-1, n_samples, -1).reshape(-1, 3)

        xyz_enc = positional_encoding(xyz_flat, self.pos_freqs)
        dir_enc = positional_encoding(dirs_expanded, self.dir_freqs)

        h = xyz_enc
        for i, layer in enumerate(self.pts_layers):
            if i == self.skip_layer:
                h = torch.cat([h, xyz_enc], dim=-1)
            h = self.relu(layer(h))

        sigma = self.sigma_head(h).reshape(n_rays, n_samples, 1)
        features = self.feature_head(h)
        rgb = self.color_layers(torch.cat([features, dir_enc], dim=-1)).reshape(n_rays, n_samples, 3)
        return sigma, rgb


def sample_along_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")

    t_lin = torch.linspace(near, far, n_samples, device=ray_origins.device, dtype=ray_origins.dtype)
    t_vals = t_lin.unsqueeze(0).expand(ray_origins.shape[0], -1)

    if perturb:
        mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
        lower = torch.cat([t_vals[:, :1], mids], dim=-1)
        upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
        t_vals = lower + (upper - lower) * torch.rand_like(t_vals)

    xyz_samples = ray_origins[:, None, :] + ray_directions[:, None, :] * t_vals[..., None]
    return xyz_samples, t_vals


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_importance: int,
    deterministic: bool = False,
) -> torch.Tensor:
    eps = 1e-5
    weights = weights + eps
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

    if deterministic:
        u = torch.linspace(0.0, 1.0, n_importance, device=bins.device, dtype=bins.dtype).expand(cdf.shape[0], -1)
    else:
        u = torch.rand(cdf.shape[0], n_importance, device=bins.device, dtype=bins.dtype)

    inds = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1] - 1)

    cdf_g0 = torch.gather(cdf, 1, below)
    cdf_g1 = torch.gather(cdf, 1, above)

    bins_pad = torch.cat([bins[:, :1], bins], dim=-1)
    bins_g0 = torch.gather(bins_pad, 1, below)
    bins_g1 = torch.gather(bins_pad, 1, above)

    denom = torch.where((cdf_g1 - cdf_g0) < eps, torch.ones_like(cdf_g1), cdf_g1 - cdf_g0)
    return bins_g0 + ((u - cdf_g0) / denom) * (bins_g1 - bins_g0)


def volume_render(
    sigmas: torch.Tensor,
    rgbs: torch.Tensor,
    t_vals: torch.Tensor,
    ray_directions: torch.Tensor,
    white_bkgd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma = torch.relu(sigmas.squeeze(-1))
    dists = t_vals[:, 1:] - t_vals[:, :-1]
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)
    dists = dists * torch.linalg.norm(ray_directions, dim=-1, keepdim=True)

    alpha = 1.0 - torch.exp(-sigma * dists)
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]
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
    param = next(model_coarse.parameters())
    ray_origins = ray_origins.to(device=param.device, dtype=param.dtype)
    ray_directions = ray_directions.to(device=param.device, dtype=param.dtype)

    xyz_c, t_c = sample_along_rays(ray_origins, ray_directions, near, far, n_coarse, perturb=perturb)
    sigma_c, rgb_c = model_coarse(xyz_c, ray_directions)
    rgb_map_c, depth_map_c, acc_map_c, weights_c = volume_render(
        sigma_c, rgb_c, t_c, ray_directions, white_bkgd=white_bkgd
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
    pdf_weights = weights_c[:, 1:-1].detach() + 1e-5
    t_fine = sample_pdf(t_mids, pdf_weights, n_importance=n_fine, deterministic=not perturb)
    t_all = torch.sort(torch.cat([t_c, t_fine], dim=-1), dim=-1).values
    xyz_f = ray_origins[:, None, :] + ray_directions[:, None, :] * t_all[..., None]

    sigma_f, rgb_f = model_fine(xyz_f, ray_directions)
    rgb_map_f, depth_map_f, acc_map_f, weights_f = volume_render(
        sigma_f, rgb_f, t_all, ray_directions, white_bkgd=white_bkgd
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
    outputs: dict[str, list[torch.Tensor]] = {}
    for start in range(0, ray_origins.shape[0], chunk_size):
        end = min(start + chunk_size, ray_origins.shape[0])
        chunk_out = render_fn(ray_origins[start:end], ray_directions[start:end])
        for key, val in chunk_out.items():
            outputs.setdefault(key, []).append(val)
    return {key: torch.cat(vals, dim=0) for key, vals in outputs.items()}

# %%
def build_part2_data(data_path: str, device: str) -> dict[str, Any]:
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, k = load_data(data_path)

    train_images_t = torch.as_tensor(images_train, dtype=torch.float32, device=device)
    train_c2ws_t = torch.as_tensor(c2ws_train, dtype=torch.float32, device=device)
    val_images_t = torch.as_tensor(images_val, dtype=torch.float32, device=device)
    val_c2ws_t = torch.as_tensor(c2ws_val, dtype=torch.float32, device=device)
    test_c2ws_t = torch.as_tensor(c2ws_test, dtype=torch.float32, device=device)
    k_t = torch.as_tensor(k, dtype=torch.float32, device=device)

    train_dataset = RaysData(train_images_t, k_t, train_c2ws_t, device=device)
    return {
        "k": k_t,
        "train_dataset": train_dataset,
        "val_images": val_images_t,
        "val_c2ws": val_c2ws_t,
        "test_c2ws": test_c2ws_t,
    }


def calibrate_near_far_from_cameras(c2ws: torch.Tensor) -> tuple[float, float]:
    centers = c2ws[:, :3, 3]
    dists = torch.linalg.norm(centers, dim=-1)
    q10 = torch.quantile(dists, 0.10).item()
    q90 = torch.quantile(dists, 0.90).item()
    near = max(0.1, 0.5 * q10)
    far = max(near + 1.0, 1.5 * q90)
    return near, far


def mse_to_psnr(mse_val: torch.Tensor | float) -> float:
    mse = torch.tensor(mse_val, dtype=torch.float32) if not isinstance(mse_val, torch.Tensor) else mse_val.float()
    mse = torch.clamp(mse, min=1e-12)
    return float(-10.0 * torch.log10(mse).item())


@torch.no_grad()
def render_full_validation_image(
    image: torch.Tensor,
    c2w: torch.Tensor,
    k: torch.Tensor,
    model_coarse: NeRFMLP,
    model_fine: NeRFMLP,
    near: float,
    far: float,
    n_coarse: int,
    n_fine: int,
    chunk_size: int,
    device: str,
) -> tuple[torch.Tensor, float]:
    h, w = image.shape[:2]
    rays = image_to_rays(image, c2w, k, device=device)
    rays_o = rays[..., :3].reshape(-1, 3).float()
    rays_d = rays[..., 3:].reshape(-1, 3).float()

    def _render_fn(chunk_o: torch.Tensor, chunk_d: torch.Tensor) -> dict[str, torch.Tensor]:
        return render_rays_hierarchical(
            chunk_o,
            chunk_d,
            model_coarse=model_coarse,
            model_fine=model_fine,
            n_coarse=n_coarse,
            n_fine=n_fine,
            near=near,
            far=far,
            perturb=False,
            white_bkgd=False,
        )

    out = render_image_by_chunks(rays_o, rays_d, _render_fn, chunk_size=chunk_size)
    rgb_pred = out["rgb_fine"].reshape(h, w, 3)
    return rgb_pred, mse_to_psnr(F.mse_loss(rgb_pred, image))


def train_nerf_part2(
    data_path: str,
    output_dir: str,
    device: str = "cuda",
    seed: int = 42,
    hidden_dim: int = 256,
    n_layers: int = 8,
    pos_freqs: int = 10,
    dir_freqs: int = 4,
    n_coarse: int = 32,
    n_fine: int = 32,
    n_steps: int = 5000,
    batch_rays: int = 2048,
    lr: float = 5e-4,
    eval_every: int = 250,
    chunk_size: int = 4096,
    near_override: float | None = None,
    far_override: float | None = None,
    log_every: int | None = None,
    save_progress_renders: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    set_seed(seed)
    out_path = ensure_dir(output_dir)

    data = build_part2_data(data_path, device=device)
    train_dataset: RaysData = data["train_dataset"]
    val_images: torch.Tensor = data["val_images"]
    val_c2ws: torch.Tensor = data["val_c2ws"]
    k: torch.Tensor = data["k"]

    near_auto, far_auto = calibrate_near_far_from_cameras(val_c2ws)
    near = near_override if near_override is not None else near_auto
    far = far_override if far_override is not None else far_auto
    log_every = max(1, n_steps // 20) if log_every is None else log_every

    model_coarse = NeRFMLP(pos_freqs=pos_freqs, dir_freqs=dir_freqs, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    model_fine = NeRFMLP(pos_freqs=pos_freqs, dir_freqs=dir_freqs, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(list(model_coarse.parameters()) + list(model_fine.parameters()), lr=lr)

    loss_hist: list[float] = []
    eval_steps: list[int] = []
    val_psnr_hist: list[float] = []
    best_psnr = -1.0
    best_step = -1
    run_start = time.perf_counter()

    if verbose:
        print(
            f"[train_nerf_part2] start device={device} steps={n_steps} "
            f"batch_rays={batch_rays} coarse={n_coarse} fine={n_fine} "
            f"near={near:.3f} far={far:.3f} eval_every={eval_every}"
        )

    for step in range(1, n_steps + 1):
        iter_start = time.perf_counter()
        rays_o, rays_d, gt_rgb = train_dataset.sample_rays(batch_rays)
        out = render_rays_hierarchical(
            rays_o.float(),
            rays_d.float(),
            model_coarse=model_coarse,
            model_fine=model_fine,
            n_coarse=n_coarse,
            n_fine=n_fine,
            near=near,
            far=far,
            perturb=True,
            white_bkgd=False,
        )

        loss_coarse = F.mse_loss(out["rgb_coarse"], gt_rgb.float())
        loss_fine = F.mse_loss(out["rgb_fine"], gt_rgb.float())
        loss = 0.1 * loss_coarse + loss_fine

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_hist.append(float(loss.item()))

        if verbose and (step == 1 or step % log_every == 0 or step == n_steps):
            iter_seconds = time.perf_counter() - iter_start
            rays_per_sec = batch_rays / max(iter_seconds, 1e-9)
            print(
                f"[train_nerf_part2] step {step}/{n_steps} "
                f"loss={loss.item():.6f} coarse={loss_coarse.item():.6f} "
                f"fine={loss_fine.item():.6f} iter={iter_seconds:.3f}s "
                f"rays_per_sec={rays_per_sec:,.0f}"
            )

        if step % eval_every == 0 or step == n_steps:
            eval_start = time.perf_counter()
            val_rgb_pred, val_psnr = render_full_validation_image(
                image=val_images[0],
                c2w=val_c2ws[0],
                k=k,
                model_coarse=model_coarse,
                model_fine=model_fine,
                near=near,
                far=far,
                n_coarse=n_coarse,
                n_fine=n_fine,
                chunk_size=chunk_size,
                device=device,
            )
            eval_seconds = time.perf_counter() - eval_start

            if save_progress_renders:
                progress_dir = ensure_dir(out_path / "progress_renders")
                save_rgb_png(val_rgb_pred.cpu().numpy(), progress_dir / f"step_{step:04d}.png")

            eval_steps.append(step)
            val_psnr_hist.append(val_psnr)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_step = step
                torch.save(
                    {
                        "model_coarse": model_coarse.state_dict(),
                        "model_fine": model_fine.state_dict(),
                        "step": step,
                        "best_psnr": best_psnr,
                    },
                    out_path / "checkpoint_best.pt",
                )
                if verbose:
                    print(
                        f"[train_nerf_part2] eval step {step}/{n_steps} "
                        f"val_psnr={val_psnr:.3f}dB best={best_psnr:.3f}dB "
                        f"(new best, saved checkpoint) eval={eval_seconds:.2f}s"
                    )
            elif verbose:
                print(
                    f"[train_nerf_part2] eval step {step}/{n_steps} "
                    f"val_psnr={val_psnr:.3f}dB best={best_psnr:.3f}dB "
                    f"eval={eval_seconds:.2f}s"
                )

    total_seconds = time.perf_counter() - run_start
    metrics = {
        "best_psnr": best_psnr,
        "best_step": best_step,
        "final_loss": loss_hist[-1] if loss_hist else None,
        "near": near,
        "far": far,
        "n_steps": n_steps,
        "batch_rays": batch_rays,
        "n_coarse": n_coarse,
        "n_fine": n_fine,
        "val_psnr_hist": val_psnr_hist,
        "eval_steps": eval_steps,
        "total_seconds": total_seconds,
        "avg_seconds_per_step": total_seconds / max(n_steps, 1),
        "log_every": log_every,
    }
    save_metrics_json(metrics, out_path / "report_metrics.json")

    if verbose:
        print(
            f"[train_nerf_part2] done total={total_seconds:.2f}s "
            f"avg_step={total_seconds / max(n_steps, 1):.4f}s "
            f"best_psnr={best_psnr:.3f}dB step={best_step}"
        )

    return {
        "metrics": metrics,
        "loss_hist": loss_hist,
        "eval_steps": eval_steps,
        "val_psnr_hist": val_psnr_hist,
        "models": {"coarse": model_coarse, "fine": model_fine},
        "data": data,
    }


@torch.no_grad()
def _iter_test_trajectory_renders(
    model_coarse: NeRFMLP,
    model_fine: NeRFMLP,
    k: torch.Tensor,
    test_c2ws: torch.Tensor,
    image_hw: tuple[int, int],
    near: float,
    far: float,
    n_coarse: int,
    n_fine: int,
    chunk_size: int,
    device: str,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    h, w = image_hw
    dummy_image = torch.zeros((h, w, 3), device=device)

    for idx in range(test_c2ws.shape[0]):
        print(f"Rendering test view {idx + 1}/{test_c2ws.shape[0]}...")
        rays = image_to_rays(dummy_image, test_c2ws[idx], k, device=device)
        rays_o = rays[..., :3].reshape(-1, 3).float()
        rays_d = rays[..., 3:].reshape(-1, 3).float()

        def _render_fn(chunk_o: torch.Tensor, chunk_d: torch.Tensor) -> dict[str, torch.Tensor]:
            return render_rays_hierarchical(
                chunk_o,
                chunk_d,
                model_coarse=model_coarse,
                model_fine=model_fine,
                n_coarse=n_coarse,
                n_fine=n_fine,
                near=near,
                far=far,
                perturb=False,
                white_bkgd=False,
            )

        out = render_image_by_chunks(rays_o, rays_d, _render_fn, chunk_size=chunk_size)
        rgb = out["rgb_fine"].reshape(h, w, 3).detach().cpu().numpy()
        depth = out["depth_fine"].reshape(h, w).detach().cpu().numpy()
        yield idx, rgb, depth


def _save_gif_from_pngs(png_paths: list[Path], output_path: str | Path, duration: float) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [imageio.imread(path) for path in png_paths]
    imageio.mimsave(output_path, frames, duration=duration)


@torch.no_grad()
def render_test_trajectory_rgb(
    model_coarse: NeRFMLP,
    model_fine: NeRFMLP,
    k: torch.Tensor,
    test_c2ws: torch.Tensor,
    image_hw: tuple[int, int],
    output_dir: str,
    near: float,
    far: float,
    n_coarse: int,
    n_fine: int,
    chunk_size: int,
    device: str,
    gif_duration: float = 0.1,
) -> None:
    out_path = Path(output_dir)
    rgb_npy_dir = ensure_dir(out_path / "test_rgb_npy")
    rgb_png_dir = ensure_dir(out_path / "test_rgb_png")

    png_paths: list[Path] = []
    for idx, rgb, _depth in _iter_test_trajectory_renders(
        model_coarse, model_fine, k, test_c2ws, image_hw, near, far, n_coarse, n_fine, chunk_size, device
    ):
        np.save(rgb_npy_dir / f"view_{idx:02d}.npy", rgb)
        png_path = rgb_png_dir / f"view_{idx:02d}.png"
        save_rgb_png(rgb, png_path)
        png_paths.append(png_path)

    _save_gif_from_pngs(png_paths, out_path / "test_rgb.gif", duration=gif_duration)


@torch.no_grad()
def render_test_trajectory_depth(
    model_coarse: NeRFMLP,
    model_fine: NeRFMLP,
    k: torch.Tensor,
    test_c2ws: torch.Tensor,
    image_hw: tuple[int, int],
    output_dir: str,
    near: float,
    far: float,
    n_coarse: int,
    n_fine: int,
    chunk_size: int,
    device: str,
    gif_duration: float = 0.1,
) -> None:
    out_path = Path(output_dir)
    depth_npy_dir = ensure_dir(out_path / "test_depth_npy")
    depth_png_dir = ensure_dir(out_path / "test_depth_png")

    png_paths: list[Path] = []
    for idx, _rgb, depth in _iter_test_trajectory_renders(
        model_coarse, model_fine, k, test_c2ws, image_hw, near, far, n_coarse, n_fine, chunk_size, device
    ):
        np.save(depth_npy_dir / f"view_{idx:02d}.npy", depth)
        png_path = depth_png_dir / f"view_{idx:02d}.png"
        save_depth_png(depth, png_path)
        png_paths.append(png_path)

    _save_gif_from_pngs(png_paths, out_path / "test_depth.gif", duration=gif_duration)

# %% [markdown]
# Config

# %%
CFG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_path": "lego_200x200.npz",
    "output_dir": "images/output/part2_colab",
    "hidden_dim": 256,
    "n_layers": 8,
    "pos_freqs": 10,
    "dir_freqs": 4,
    "n_coarse": 32,
    "n_fine": 32,
    "batch_rays": 2048,
    "n_steps": 5000,
    "eval_every": 250,
    "chunk_size": 4096,
    "lr": 5e-4,
    "near_override": None,
    "far_override": None,
    "gif_duration": 0.1,
    "save_progress_renders": True,
    "verbose": True,
}

CFG

# %% [markdown]
# Train

# %%
results = train_nerf_part2(
    data_path=CFG["data_path"],
    output_dir=CFG["output_dir"],
    device=CFG["device"],
    seed=CFG["seed"],
    hidden_dim=CFG["hidden_dim"],
    n_layers=CFG["n_layers"],
    pos_freqs=CFG["pos_freqs"],
    dir_freqs=CFG["dir_freqs"],
    n_coarse=CFG["n_coarse"],
    n_fine=CFG["n_fine"],
    n_steps=CFG["n_steps"],
    batch_rays=CFG["batch_rays"],
    lr=CFG["lr"],
    eval_every=CFG["eval_every"],
    chunk_size=CFG["chunk_size"],
    near_override=CFG["near_override"],
    far_override=CFG["far_override"],
    save_progress_renders=CFG["save_progress_renders"],
    verbose=CFG["verbose"],
)

plot_training_curves(
    results["loss_hist"],
    results["eval_steps"],
    results["val_psnr_hist"],
    CFG["output_dir"],
)

print("Saved training artifacts to:", Path(CFG["output_dir"]))
results["metrics"]

# %% [markdown]
# Shared Render Setup

# %%
models = results["models"]
data = results["data"]
image_hw = (data["val_images"].shape[1], data["val_images"].shape[2])
render_kwargs = {
    "model_coarse": models["coarse"],
    "model_fine": models["fine"],
    "k": data["k"],
    "test_c2ws": data["test_c2ws"],
    "image_hw": image_hw,
    "output_dir": CFG["output_dir"],
    "near": results["metrics"]["near"],
    "far": results["metrics"]["far"],
    "n_coarse": CFG["n_coarse"],
    "n_fine": CFG["n_fine"],
    "chunk_size": CFG["chunk_size"],
    "device": CFG["device"],
    "gif_duration": CFG["gif_duration"],
}

# %% [markdown]
# Render RGB

# %%
render_test_trajectory_rgb(**render_kwargs)
print("Saved RGB outputs to:", Path(CFG["output_dir"]) / "test_rgb_npy")
print("Saved RGB PNGs to:", Path(CFG["output_dir"]) / "test_rgb_png")
print("Saved RGB GIF to:", Path(CFG["output_dir"]) / "test_rgb.gif")

# %% [markdown]
# Render Depth

# %%
render_test_trajectory_depth(**render_kwargs)
print("Saved depth outputs to:", Path(CFG["output_dir"]) / "test_depth_npy")
print("Saved depth PNGs to:", Path(CFG["output_dir"]) / "test_depth_png")
print("Saved depth GIF to:", Path(CFG["output_dir"]) / "test_depth.gif")

# %% [markdown]
# Preview Saved GIFs

# %%
from IPython.display import Image as IPyImage, display

rgb_gif = Path(CFG["output_dir"]) / "test_rgb.gif"
depth_gif = Path(CFG["output_dir"]) / "test_depth.gif"

if rgb_gif.exists():
    display(IPyImage(data=rgb_gif.read_bytes(), format="gif"))
else:
    print("RGB GIF not found:", rgb_gif)

if depth_gif.exists():
    display(IPyImage(data=depth_gif.read_bytes(), format="gif"))
else:
    print("Depth GIF not found:", depth_gif)
