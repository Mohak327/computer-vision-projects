from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_experiment_config
from .data.writer import write_simulation_dataset
from .evaluation.nerf_eval import evaluate_nerf_model
from .evaluation.world_model_eval import evaluate_world_model
from .sim.simulator import HumanoidSimulationService
from .training.nerf_trainer import NerfTrainer
from .training.orchestrator import run_full_pipeline
from .training.world_model_trainer import WorldModelTrainer
from .utils.io import save_json
from .utils.random import set_seed


def _parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PhysNeRF-BS command line tools")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser


def generate_dataset_main() -> None:
    parser = _parse_args()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    set_seed(config.runtime.seed)
    simulation = HumanoidSimulationService(config.simulation, seed=config.runtime.seed)
    bundles = simulation.generate_dataset()
    result = write_simulation_dataset(bundles, config.data, n_cameras=config.simulation.n_cameras)
    print(f"Wrote dataset to {result.root} with {result.num_frames} frames.")


def train_nerf_main() -> None:
    parser = _parse_args()
    parser.add_argument("--mode", type=str, default=None)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.mode is not None:
        if args.mode == "pose_physics":
            config.nerf.mode = "pose"
            config.physics.enable = True
        else:
            config.nerf.mode = args.mode
            config.physics.enable = False
    trainer = NerfTrainer(
        dataset_root=config.data.root,
        data_config=config.data,
        nerf_config=config.nerf,
        physics_config=config.physics,
        eval_config=config.evaluation,
        device=config.runtime.device,
    )
    output_dir = Path(config.runtime.output_root) / f"nerf_{config.nerf.mode}"
    metrics = trainer.train(output_dir)
    print(metrics["metrics"])


def train_world_model_main() -> None:
    parser = _parse_args()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    trainer = WorldModelTrainer(
        dataset_root=config.data.root,
        data_config=config.data,
        world_model_config=config.world_model,
        eval_config=config.evaluation,
        device=config.runtime.device,
    )
    output_dir = Path(config.runtime.output_root) / "world_model"
    metrics = trainer.train(output_dir)
    print(metrics["metrics"])


def evaluate_nerf_main() -> None:
    parser = _parse_args()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    trainer = NerfTrainer(
        dataset_root=config.data.root,
        data_config=config.data,
        nerf_config=config.nerf,
        physics_config=config.physics,
        eval_config=config.evaluation,
        device=config.runtime.device,
    )
    trainer.train(Path(config.runtime.output_root) / "_tmp_eval_nerf")
    dataset = trainer.test_dataset if len(trainer.test_dataset) else trainer.val_dataset
    metrics = evaluate_nerf_model(
        trainer.model_coarse,
        trainer.model_fine,
        dataset,
        config.nerf,
        config.physics,
        config.evaluation,
        config.runtime.device,
    )
    save_json(metrics, Path(config.runtime.output_root) / "nerf_eval_metrics.json")
    print(metrics)


def evaluate_world_model_main() -> None:
    parser = _parse_args()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    trainer = WorldModelTrainer(
        dataset_root=config.data.root,
        data_config=config.data,
        world_model_config=config.world_model,
        eval_config=config.evaluation,
        device=config.runtime.device,
    )
    train_output = trainer.train(Path(config.runtime.output_root) / "_tmp_eval_world_model")
    from torch.utils.data import DataLoader

    loader = DataLoader(trainer.val_dataset if len(trainer.val_dataset) else trainer.train_dataset, batch_size=1, shuffle=False)
    metrics = evaluate_world_model(train_output["model"], loader, config.runtime.device)
    save_json(metrics, Path(config.runtime.output_root) / "world_model_eval_metrics.json")
    print(metrics)


def run_ablation_suite_main() -> None:
    parser = _parse_args()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    results = run_full_pipeline(config)
    summary = {key: value["metrics"] if isinstance(value, dict) and "metrics" in value else value for key, value in results.items()}
    print(summary)
