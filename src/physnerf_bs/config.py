from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 7
    device: str = "cpu"
    dtype: str = "float32"
    output_root: str = "outputs/default"


@dataclass(slots=True)
class SimulationConfig:
    backend: str = "dm_control"
    fallback_backend: str = "synthetic"
    image_height: int = 64
    image_width: int = 64
    n_cameras: int = 8
    camera_radius: float = 3.0
    camera_elevation_deg: float = 20.0
    frames_per_motion: int = 48
    motions: list[str] = field(default_factory=lambda: ["walk", "arm_raise", "crouch"])
    dof: int = 17
    body_mass: float = 75.0
    noise_std: float = 0.01
    use_segmentation: bool = True
    use_depth: bool = True


@dataclass(slots=True)
class DataConfig:
    root: str = "outputs/default/dataset"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    preload_images: bool = False
    clip_length: int = 8
    ray_batch_size: int = 1024


@dataclass(slots=True)
class NerfConfig:
    mode: str = "pose"
    pos_freqs: int = 10
    dir_freqs: int = 4
    pose_freqs: int = 4
    hidden_dim: int = 128
    n_layers: int = 6
    skip_layer: int = 3
    n_coarse: int = 24
    n_fine: int = 24
    near: float = 0.5
    far: float = 4.5
    lr: float = 5e-4
    steps: int = 40
    eval_every: int = 10
    chunk_size: int = 2048
    white_background: bool = True
    checkpoint_every: int = 20


@dataclass(slots=True)
class PhysicsConfig:
    enable: bool = True
    weight: float = 0.1
    depth_weight: float = 0.5
    silhouette_weight: float = 0.25
    energy_weight: float = 1.0
    proxy_gravity_scale: float = 9.81


@dataclass(slots=True)
class WorldModelConfig:
    hidden_channels: int = 24
    gated_channels: int = 8
    kernel_size: int = 5
    steps: int = 30
    clip_length: int = 8
    batch_size: int = 2
    lr: float = 1e-3
    checkpoint_every: int = 10


@dataclass(slots=True)
class EvaluationConfig:
    lpips_net: str = "alex"
    batch_size: int = 2
    occlusion_threshold: float = 0.25


@dataclass(slots=True)
class ExperimentConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    nerf: NerfConfig = field(default_factory=NerfConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
    kwargs = {name: data.get(name, field_def.default) for name, field_def in fields.items()}
    return cls(**kwargs)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def load_experiment_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    config = ExperimentConfig()
    merged = config_to_dict(config)
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            from_file = yaml.safe_load(handle) or {}
        merged = _merge_dict(merged, from_file)
    if overrides:
        merged = _merge_dict(merged, overrides)
    return ExperimentConfig(
        runtime=_from_dict(RuntimeConfig, merged["runtime"]),
        simulation=_from_dict(SimulationConfig, merged["simulation"]),
        data=_from_dict(DataConfig, merged["data"]),
        nerf=_from_dict(NerfConfig, merged["nerf"]),
        physics=_from_dict(PhysicsConfig, merged["physics"]),
        world_model=_from_dict(WorldModelConfig, merged["world_model"]),
        evaluation=_from_dict(EvaluationConfig, merged["evaluation"]),
    )


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)
