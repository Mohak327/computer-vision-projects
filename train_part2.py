from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from dataset_3d import RaysData, image_to_rays, load_data
from nerf_model import NeRFMLP
from rendering import render_image_by_chunks, render_rays_hierarchical


def build_part2_data(
    data_path: str,
    device: str,
) -> dict[str, Any]:
    """Load NPZ and create train/val ray datasets."""
    (
        images_train,
        c2ws_train,
        images_val,
        c2ws_val,
        c2ws_test,
        k,
    ) = load_data(data_path)

    train_images_t = torch.as_tensor(images_train, dtype=torch.float32, device=device)
    train_c2ws_t = torch.as_tensor(c2ws_train, dtype=torch.float32, device=device)
    val_images_t = torch.as_tensor(images_val, dtype=torch.float32, device=device)
    val_c2ws_t = torch.as_tensor(c2ws_val, dtype=torch.float32, device=device)
    test_c2ws_t = torch.as_tensor(c2ws_test, dtype=torch.float32, device=device)
    k_t = torch.as_tensor(k, dtype=torch.float32, device=device)

    train_dataset = RaysData(train_images_t, k_t, train_c2ws_t, split="train", device=device)

    return {
        "k": k_t,
        "train_dataset": train_dataset,
        "val_images": val_images_t,
        "val_c2ws": val_c2ws_t,
        "test_c2ws": test_c2ws_t,
    }


def calibrate_near_far_from_cameras(c2ws: torch.Tensor) -> tuple[float, float]:
    """Estimate near/far from camera-center distances.

    This provides a data-driven starting point and can be manually overridden.
    """
    centers = c2ws[:, :3, 3]
    dists = torch.linalg.norm(centers, dim=-1)

    q10 = torch.quantile(dists, 0.10).item()
    q90 = torch.quantile(dists, 0.90).item()
    near = max(0.1, 0.5 * q10)
    far = 1.5 * q90

    if far <= near:
        far = near + 1.0
    return near, far


def mse_to_psnr(mse_val: torch.Tensor | float) -> float:
    """Convert MSE to PSNR in dB."""
    if not isinstance(mse_val, torch.Tensor):
        mse = torch.tensor(mse_val, dtype=torch.float32)
    else:
        mse = mse_val.float()
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
    """Render one full validation image and return PSNR."""
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

    mse = F.mse_loss(rgb_pred, image)
    psnr = mse_to_psnr(mse)
    return rgb_pred, psnr


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
    verbose: bool = True,
) -> dict[str, Any]:
    """Train coarse+fine NeRF and save key artifacts."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = build_part2_data(data_path, device=device)
    train_dataset: RaysData = data["train_dataset"]
    val_images: torch.Tensor = data["val_images"]
    val_c2ws: torch.Tensor = data["val_c2ws"]
    k: torch.Tensor = data["k"]

    near_auto, far_auto = calibrate_near_far_from_cameras(val_c2ws)
    near = near_override if near_override is not None else near_auto
    far = far_override if far_override is not None else far_auto

    run_start = time.perf_counter()
    if log_every is None:
        log_every = max(1, n_steps // 20)

    if verbose:
        print(
            f"[train_nerf_part2] start device={device} steps={n_steps} "
            f"batch_rays={batch_rays} coarse={n_coarse} fine={n_fine} "
            f"near={near:.3f} far={far:.3f} eval_every={eval_every}"
        )

    model_coarse = NeRFMLP(
        pos_freqs=pos_freqs,
        dir_freqs=dir_freqs,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    ).to(device)
    model_fine = NeRFMLP(
        pos_freqs=pos_freqs,
        dir_freqs=dir_freqs,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    ).to(device)

    params = list(model_coarse.parameters()) + list(model_fine.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    loss_hist: list[float] = []
    eval_steps: list[int] = []
    val_psnr_hist: list[float] = []

    best_psnr = -1.0
    best_step = -1

    for step in range(1, n_steps + 1):
        iter_start = time.perf_counter()

        rays_o, rays_d, gt_rgb = train_dataset.sample_rays(batch_rays)
        rays_o = rays_o.float()
        rays_d = rays_d.float()
        gt_rgb = gt_rgb.float()

        out = render_rays_hierarchical(
            rays_o,
            rays_d,
            model_coarse=model_coarse,
            model_fine=model_fine,
            n_coarse=n_coarse,
            n_fine=n_fine,
            near=near,
            far=far,
            perturb=True,
            white_bkgd=False,
        )

        loss_coarse = F.mse_loss(out["rgb_coarse"], gt_rgb)
        loss_fine = F.mse_loss(out["rgb_fine"], gt_rgb)
        loss = 0.1 * loss_coarse + loss_fine

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_hist.append(float(loss.item()))

        iter_seconds = time.perf_counter() - iter_start
        if verbose and (step == 1 or step % log_every == 0 or step == n_steps):
            rays_per_sec = batch_rays / max(iter_seconds, 1e-9)
            print(
                f"[train_nerf_part2] step {step}/{n_steps} "
                f"loss={loss.item():.6f} coarse={loss_coarse.item():.6f} "
                f"fine={loss_fine.item():.6f} iter={iter_seconds:.3f}s "
                f"rays_per_sec={rays_per_sec:,.0f}"
            )

        if step % eval_every == 0 or step == n_steps:
            eval_start = time.perf_counter()
            _, val_psnr = render_full_validation_image(
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
    if verbose:
        print(
            f"[train_nerf_part2] done total={total_seconds:.2f}s "
            f"avg_step={total_seconds / max(n_steps, 1):.4f}s "
            f"best_psnr={best_psnr:.3f}dB step={best_step}"
        )

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

    with open(out_path / "report_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "metrics": metrics,
        "loss_hist": loss_hist,
        "eval_steps": eval_steps,
        "val_psnr_hist": val_psnr_hist,
        "models": {"coarse": model_coarse, "fine": model_fine},
        "data": data,
    }


@torch.no_grad()
def render_test_trajectory(
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
) -> None:
    """Render all test poses and save RGB/depth outputs as numpy arrays.

    PNG writing is intentionally left to notebook utilities to keep this module torch-centric.
    """
    out_path = Path(output_dir)
    rgb_dir = out_path / "test_rgb_npy"
    depth_dir = out_path / "test_depth_npy"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    h, w = image_hw
    dummy_image = torch.zeros((h, w, 3), device=device)

    for idx in range(test_c2ws.shape[0]):
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

        np.save(rgb_dir / f"view_{idx:02d}.npy", rgb)
        np.save(depth_dir / f"view_{idx:02d}.npy", depth)
