from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, save_experiment_config
from ..data.writer import write_simulation_dataset
from ..sim.simulator import HumanoidSimulationService
from ..utils.io import ensure_dir, save_json
from ..utils.random import set_seed
from .nerf_trainer import NerfTrainer
from .world_model_trainer import WorldModelTrainer


def _resolve_config_paths(config: ExperimentConfig) -> ExperimentConfig:
    output_root = Path(config.runtime.output_root)
    data_root = Path(config.data.root)
    if not data_root.is_absolute():
        data_root = output_root / data_root.name if data_root.name else output_root / "dataset"
    return replace(
        config,
        data=replace(config.data, root=str(data_root)),
    )


def run_full_pipeline(config: ExperimentConfig) -> dict[str, Any]:
    config = _resolve_config_paths(config)
    set_seed(config.runtime.seed)

    output_root = ensure_dir(config.runtime.output_root)
    save_experiment_config(config, output_root / "resolved_config.yaml")

    simulation = HumanoidSimulationService(config.simulation, seed=config.runtime.seed)
    bundles = simulation.generate_dataset()
    dataset_result = write_simulation_dataset(bundles, config.data, n_cameras=config.simulation.n_cameras)

    runs: dict[str, Any] = {
        "dataset": {
            "root": str(dataset_result.root),
            "num_frames": dataset_result.num_frames,
            "num_sequences": dataset_result.num_sequences,
        }
    }

    nerf_variants = [
        ("static_nerf", replace(config.nerf, mode="static"), replace(config.physics, enable=False)),
        ("pose_nerf", replace(config.nerf, mode="pose"), replace(config.physics, enable=False)),
        ("phys_nerf", replace(config.nerf, mode="pose"), replace(config.physics, enable=True)),
    ]
    for name, nerf_cfg, physics_cfg in nerf_variants:
        trainer = NerfTrainer(
            dataset_root=config.data.root,
            data_config=config.data,
            nerf_config=nerf_cfg,
            physics_config=physics_cfg,
            eval_config=config.evaluation,
            device=config.runtime.device,
        )
        runs[name] = trainer.train(output_root / name)

    world_trainer = WorldModelTrainer(
        dataset_root=config.data.root,
        data_config=config.data,
        world_model_config=config.world_model,
        eval_config=config.evaluation,
        device=config.runtime.device,
    )
    runs["world_model"] = world_trainer.train(output_root / "world_model")

    summary = {
        key: value["metrics"] if isinstance(value, dict) and "metrics" in value else value
        for key, value in runs.items()
    }
    save_json(summary, output_root / "pipeline_summary.json")
    return runs
