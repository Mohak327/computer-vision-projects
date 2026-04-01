from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..config import DataConfig, EvaluationConfig, NerfConfig, PhysicsConfig
from ..data.datasets import FrameDataset, PoseConditionedRayDataset, _pixels_to_rays
from ..evaluation.nerf_eval import evaluate_nerf_model
from ..models.nerf import PoseConditionedNeRF, render_image_grid, render_rays_hierarchical
from ..physics.hamiltonian import HybridHamiltonianLoss
from ..utils.io import ensure_dir, save_json, save_rgb_image


class NerfTrainer:
    def __init__(
        self,
        dataset_root: str | Path,
        data_config: DataConfig,
        nerf_config: NerfConfig,
        physics_config: PhysicsConfig,
        eval_config: EvaluationConfig,
        device: str = "cpu",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.data_config = data_config
        self.nerf_config = nerf_config
        self.physics_config = physics_config
        self.eval_config = eval_config
        self.device = device

        train_frames = FrameDataset(self.dataset_root, split="train", preload_images=data_config.preload_images)
        if not train_frames.records:
            raise ValueError("Training split is empty.")
        self.train_dataset = PoseConditionedRayDataset(self.dataset_root, split="train", data_config=data_config, device=device)
        self.val_dataset = FrameDataset(self.dataset_root, split="val", preload_images=data_config.preload_images)
        self.test_dataset = FrameDataset(self.dataset_root, split="test", preload_images=data_config.preload_images)
        pose_dim = len(train_frames.records[0].qpos)

        self.model_coarse = PoseConditionedNeRF(
            pose_dim=pose_dim,
            mode=nerf_config.mode,
            pos_freqs=nerf_config.pos_freqs,
            dir_freqs=nerf_config.dir_freqs,
            pose_freqs=nerf_config.pose_freqs,
            hidden_dim=nerf_config.hidden_dim,
            n_layers=nerf_config.n_layers,
            skip_layer=nerf_config.skip_layer,
        ).to(device)
        self.model_fine = PoseConditionedNeRF(
            pose_dim=pose_dim,
            mode=nerf_config.mode,
            pos_freqs=nerf_config.pos_freqs,
            dir_freqs=nerf_config.dir_freqs,
            pose_freqs=nerf_config.pose_freqs,
            hidden_dim=nerf_config.hidden_dim,
            n_layers=nerf_config.n_layers,
            skip_layer=nerf_config.skip_layer,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.model_coarse.parameters()) + list(self.model_fine.parameters()),
            lr=nerf_config.lr,
        )
        self.physics_loss = HybridHamiltonianLoss(physics_config)

    @torch.no_grad()
    def render_validation_frame(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        rgb = item["rgb"].to(self.device)
        h, w = rgb.shape[:2]
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        uv = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).float().to(self.device)
        ray_o, ray_d = _pixels_to_rays(item["intrinsics"].to(self.device), item["extrinsics"].to(self.device), uv)
        pose = None if self.nerf_config.mode == "static" else item["qpos"].to(self.device)
        pose_batch = None if pose is None else pose.unsqueeze(0).expand(ray_o.shape[0], -1)

        def render_fn(chunk_o: torch.Tensor, chunk_d: torch.Tensor, chunk_pose: torch.Tensor | None):
            return render_rays_hierarchical(
                chunk_o,
                chunk_d,
                pose=chunk_pose,
                model_coarse=self.model_coarse,
                model_fine=self.model_fine,
                n_coarse=self.nerf_config.n_coarse,
                n_fine=self.nerf_config.n_fine,
                near=self.nerf_config.near,
                far=self.nerf_config.far,
                perturb=False,
                white_background=self.nerf_config.white_background,
            )

        return render_image_grid(ray_o, ray_d, pose_batch, render_fn, self.nerf_config.chunk_size, (h, w))

    def train(self, output_dir: str | Path) -> dict[str, Any]:
        output_dir = ensure_dir(output_dir)
        history: list[dict[str, float]] = []
        best_psnr = float("-inf")
        best_step = 0

        for step in range(1, self.nerf_config.steps + 1):
            batch = self.train_dataset.sample_rays(self.data_config.ray_batch_size)
            pose = None if self.nerf_config.mode == "static" else batch["pose"]
            render = render_rays_hierarchical(
                batch["ray_origins"],
                batch["ray_directions"],
                pose=pose,
                model_coarse=self.model_coarse,
                model_fine=self.model_fine,
                n_coarse=self.nerf_config.n_coarse,
                n_fine=self.nerf_config.n_fine,
                near=self.nerf_config.near,
                far=self.nerf_config.far,
                perturb=True,
                white_background=self.nerf_config.white_background,
            )
            coarse_loss = F.mse_loss(render.rgb_coarse, batch["target_rgb"])
            fine_loss = F.mse_loss(render.rgb_fine, batch["target_rgb"])
            total_loss = coarse_loss + fine_loss
            physics_stats = None
            if self.physics_config.enable:
                physics_stats = self.physics_loss(
                    predicted_depth=render.depth_fine,
                    predicted_acc=render.acc_fine,
                    target_depth=batch["target_depth"],
                    target_mask=batch["target_mask"],
                    teacher_total_energy=batch["teacher_total_energy"],
                )
                total_loss = total_loss + physics_stats.total

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self.optimizer.step()

            record = {"step": float(step), "loss": float(total_loss.item()), "fine_loss": float(fine_loss.item())}
            if physics_stats is not None:
                record["physics_loss"] = float(physics_stats.total.item())
            history.append(record)

            if step % self.nerf_config.eval_every == 0 or step == self.nerf_config.steps:
                item = self.val_dataset[0] if len(self.val_dataset) else self.test_dataset[0]
                render_val = self.render_validation_frame(item)
                val_psnr = float((-10.0 * torch.log10(F.mse_loss(render_val["rgb"], item["rgb"].to(self.device)).clamp_min(1e-12))).item())
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    best_step = step
                    torch.save(
                        {
                            "coarse": self.model_coarse.state_dict(),
                            "fine": self.model_fine.state_dict(),
                            "step": step,
                            "psnr": best_psnr,
                            "config": asdict(self.nerf_config),
                        },
                        output_dir / "checkpoint_best.pt",
                    )
                save_rgb_image(render_val["rgb"].detach().cpu().numpy(), output_dir / f"val_render_step_{step:04d}.png")

        eval_metrics = evaluate_nerf_model(
            self.model_coarse,
            self.model_fine,
            self.test_dataset if len(self.test_dataset) else self.val_dataset,
            self.nerf_config,
            self.physics_config,
            self.eval_config,
            device=self.device,
        )
        metrics = {
            "best_psnr": best_psnr,
            "best_step": best_step,
            "history_tail": history[-5:],
            "evaluation": eval_metrics,
        }
        save_json(metrics, output_dir / "metrics.json")
        return {
            "metrics": metrics,
            "model_coarse": self.model_coarse,
            "model_fine": self.model_fine,
        }
