from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DataConfig
from ..schemas import FrameRecord, SequenceRecord
from ..utils.io import load_depth_array, load_json, load_jsonl, load_mask_image, load_rgb_image


@dataclass(slots=True)
class DatasetBundle:
    root: Path
    frames: list[FrameRecord]
    sequences: list[SequenceRecord]


def load_dataset_bundle(root: str | Path) -> DatasetBundle:
    root = Path(root)
    frames = [FrameRecord.from_dict(item) for item in load_jsonl(root / "frames.jsonl")]
    sequences = [SequenceRecord.from_dict(item) for item in load_json(root / "sequences.json")]
    return DatasetBundle(root=root, frames=frames, sequences=sequences)


class FrameDataset(Dataset):
    def __init__(self, root: str | Path, split: str | None = None, preload_images: bool = False) -> None:
        bundle = load_dataset_bundle(root)
        self.root = bundle.root
        self.records = [record for record in bundle.frames if split is None or record.split == split]
        self.preload_images = preload_images
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        if preload_images:
            for record in self.records:
                self._cache[record.frame_id] = self._load_arrays(record)

    def _load_arrays(self, record: FrameRecord) -> dict[str, np.ndarray]:
        return {
            "rgb": load_rgb_image(self.root / record.rgb_path),
            "depth": load_depth_array(self.root / record.depth_path),
            "segmentation": load_mask_image(self.root / record.segmentation_path).astype(np.float32),
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        arrays = self._cache.get(record.frame_id) if self.preload_images else None
        if arrays is None:
            arrays = self._load_arrays(record)
        return {
            "record": record,
            "rgb": torch.from_numpy(arrays["rgb"]).float(),
            "depth": torch.from_numpy(arrays["depth"]).float(),
            "segmentation": torch.from_numpy(arrays["segmentation"]).float(),
            "qpos": torch.tensor(record.qpos, dtype=torch.float32),
            "qvel": torch.tensor(record.qvel, dtype=torch.float32),
            "control": torch.tensor(record.control, dtype=torch.float32),
            "torque": torch.tensor(record.torque, dtype=torch.float32),
            "intrinsics": torch.tensor(record.camera_intrinsics, dtype=torch.float32),
            "extrinsics": torch.tensor(record.camera_extrinsics, dtype=torch.float32),
            "physics_total_energy": torch.tensor(record.physics_targets.total_energy, dtype=torch.float32),
        }


def _pixels_to_rays(k: torch.Tensor, c2w: torch.Tensor, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = uv.shape[0]
    ones = torch.ones((n, 1), device=uv.device, dtype=uv.dtype)
    pixels_h = torch.cat([uv, ones], dim=-1)
    k_inv = torch.linalg.inv(k)
    dirs_camera = (k_inv @ pixels_h.T).T
    rotation = c2w[:3, :3]
    dirs_world = (rotation @ dirs_camera.T).T
    dirs_world = dirs_world / torch.linalg.norm(dirs_world, dim=-1, keepdim=True).clamp_min(1e-8)
    origins = c2w[:3, 3].unsqueeze(0).expand_as(dirs_world)
    return origins, dirs_world


class PoseConditionedRayDataset:
    def __init__(self, root: str | Path, split: str, data_config: DataConfig, device: str = "cpu") -> None:
        self.frame_dataset = FrameDataset(root, split=split, preload_images=data_config.preload_images)
        self.device = device
        self.rng = np.random.default_rng(0)

    def sample_rays(self, batch_size: int) -> dict[str, torch.Tensor]:
        frame_indices = self.rng.integers(0, len(self.frame_dataset), size=batch_size)
        rays_o, rays_d, rgbs, depths, masks, poses, energies = [], [], [], [], [], [], []
        for frame_idx in frame_indices:
            item = self.frame_dataset[int(frame_idx)]
            rgb = item["rgb"]
            depth = item["depth"]
            mask = item["segmentation"]
            h, w = rgb.shape[:2]
            y = int(self.rng.integers(0, h))
            x = int(self.rng.integers(0, w))
            uv = torch.tensor([[float(x), float(y)]], dtype=torch.float32)
            ray_o, ray_d = _pixels_to_rays(item["intrinsics"], item["extrinsics"], uv)
            rays_o.append(ray_o[0])
            rays_d.append(ray_d[0])
            rgbs.append(rgb[y, x])
            depths.append(depth[y, x])
            masks.append(mask[y, x])
            poses.append(item["qpos"])
            energies.append(item["physics_total_energy"])
        return {
            "ray_origins": torch.stack(rays_o).to(self.device),
            "ray_directions": torch.stack(rays_d).to(self.device),
            "target_rgb": torch.stack(rgbs).to(self.device),
            "target_depth": torch.stack(depths).to(self.device),
            "target_mask": torch.stack(masks).to(self.device),
            "pose": torch.stack(poses).to(self.device),
            "teacher_total_energy": torch.stack(energies).to(self.device),
        }


class VisuomotorSequenceDataset(Dataset):
    def __init__(self, root: str | Path, split: str, clip_length: int, preload_images: bool = False) -> None:
        bundle = load_dataset_bundle(root)
        self.root = bundle.root
        self.clip_length = clip_length
        self.frame_map = {record.frame_id: record for record in bundle.frames}
        self.sequences = [sequence for sequence in bundle.sequences if sequence.split == split]
        self.preload_images = preload_images
        self.image_cache: dict[str, np.ndarray] = {}
        self.mask_cache: dict[str, np.ndarray] = {}
        if preload_images:
            for frame_id, record in self.frame_map.items():
                self.image_cache[frame_id] = load_rgb_image(self.root / record.rgb_path)
                self.mask_cache[frame_id] = load_mask_image(self.root / record.segmentation_path).astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence = self.sequences[index]
        frame_ids = sequence.frame_ids[: self.clip_length]
        frames = []
        masks = []
        qpos = []
        motor = []
        for step, frame_id in enumerate(frame_ids):
            record = self.frame_map[frame_id]
            if frame_id in self.image_cache:
                frames.append(self.image_cache[frame_id])
                masks.append(self.mask_cache[frame_id])
            else:
                frames.append(load_rgb_image(self.root / record.rgb_path))
                masks.append(load_mask_image(self.root / record.segmentation_path).astype(np.float32))
            qpos.append(np.asarray(record.qpos, dtype=np.float32))
            motor.append(np.asarray(sequence.motor_stream[step], dtype=np.float32))
        return {
            "frames": torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float(),
            "masks": torch.from_numpy(np.stack(masks)).unsqueeze(1).float(),
            "qpos": torch.from_numpy(np.stack(qpos)).float(),
            "motor": torch.from_numpy(np.stack(motor)).float(),
        }
