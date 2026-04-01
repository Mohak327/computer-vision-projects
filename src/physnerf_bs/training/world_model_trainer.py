from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import DataConfig, EvaluationConfig, WorldModelConfig
from ..data.datasets import VisuomotorSequenceDataset
from ..evaluation.world_model_eval import evaluate_world_model
from ..models.world_model import MotorGatedWorldModel
from ..utils.io import ensure_dir, save_json


class WorldModelTrainer:
    def __init__(
        self,
        dataset_root: str | Path,
        data_config: DataConfig,
        world_model_config: WorldModelConfig,
        eval_config: EvaluationConfig,
        device: str = "cpu",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.data_config = data_config
        self.world_model_config = world_model_config
        self.eval_config = eval_config
        self.device = device

        self.train_dataset = VisuomotorSequenceDataset(
            self.dataset_root,
            split="train",
            clip_length=world_model_config.clip_length,
            preload_images=data_config.preload_images,
        )
        self.val_dataset = VisuomotorSequenceDataset(
            self.dataset_root,
            split="val",
            clip_length=world_model_config.clip_length,
            preload_images=data_config.preload_images,
        )
        sample = self.train_dataset[0]
        motor_dim = sample["motor"].shape[-1]
        input_channels = sample["frames"].shape[1]
        self.model = MotorGatedWorldModel(
            input_channels=input_channels,
            motor_dim=motor_dim,
            hidden_channels=world_model_config.hidden_channels,
            gated_channels=world_model_config.gated_channels,
            kernel_size=world_model_config.kernel_size,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=world_model_config.lr)

    def train(self, output_dir: str | Path) -> dict[str, Any]:
        output_dir = ensure_dir(output_dir)
        train_loader = DataLoader(self.train_dataset, batch_size=self.world_model_config.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset if len(self.val_dataset) else self.train_dataset, batch_size=1, shuffle=False)
        history: list[dict[str, float]] = []
        best_loss = float("inf")

        for step in range(1, self.world_model_config.steps + 1):
            batch = next(iter(train_loader))
            frames = batch["frames"].to(self.device)
            motor = batch["motor"].to(self.device)
            outputs = self.model(frames, motor)
            target = frames[:, 1:]
            loss = F.mse_loss(outputs.predicted_frames, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            history.append({"step": float(step), "loss": float(loss.item())})

            if loss.item() < best_loss:
                best_loss = float(loss.item())
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "step": step,
                        "loss": best_loss,
                    },
                    output_dir / "checkpoint_best.pt",
                )

        metrics = evaluate_world_model(self.model, val_loader, self.device)
        metrics["best_loss"] = best_loss
        metrics["history_tail"] = history[-5:]
        save_json(metrics, output_dir / "metrics.json")
        return {"metrics": metrics, "model": self.model}
