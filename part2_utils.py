from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mse_to_psnr(mse: float | torch.Tensor) -> float:
    mse_tensor = mse if isinstance(mse, torch.Tensor) else torch.tensor(mse)
    mse_tensor = torch.clamp(mse_tensor.float(), min=1e-12)
    return float((-10.0 * torch.log10(mse_tensor)).item())


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
