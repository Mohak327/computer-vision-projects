from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from ..evaluation.metrics import compute_body_selectivity, compute_psnr
from ..models.world_model import MotorGatedWorldModel


@torch.no_grad()
def evaluate_world_model(
    model: MotorGatedWorldModel,
    dataloader: DataLoader,
    device: str,
    limit: int = 2,
) -> dict[str, Any]:
    losses = []
    psnrs = []
    selectivity_samples = []
    smoothness = []

    for idx, batch in enumerate(dataloader):
        if idx >= limit:
            break
        frames = batch["frames"].to(device)
        masks = batch["masks"].to(device)
        motor = batch["motor"].to(device)
        outputs = model(frames, motor)
        target = frames[:, 1:]
        loss = torch.mean((outputs.predicted_frames - target) ** 2)
        losses.append(float(loss.item()))
        psnrs.append(compute_psnr(outputs.predicted_frames[0, 0].permute(1, 2, 0).cpu(), target[0, 0].permute(1, 2, 0).cpu()))
        selectivity_samples.append(compute_body_selectivity(outputs.latent_states.cpu(), masks[:, :-1].cpu()))
        if outputs.predicted_frames.shape[1] > 1:
            smoothness.append(float(torch.mean(torch.abs(outputs.predicted_frames[:, 1:] - outputs.predicted_frames[:, :-1])).item()))

    mean_selectivity = {
        key: float(sum(sample[key] for sample in selectivity_samples) / max(len(selectivity_samples), 1))
        for key in selectivity_samples[0].keys()
    } if selectivity_samples else {}
    return {
        "prediction_mse": float(sum(losses) / max(len(losses), 1)),
        "prediction_psnr": float(sum(psnrs) / max(len(psnrs), 1)),
        "temporal_smoothness": float(sum(smoothness) / max(len(smoothness), 1)) if smoothness else 0.0,
        **mean_selectivity,
    }
