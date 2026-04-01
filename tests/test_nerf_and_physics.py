from pathlib import Path
import shutil

from physnerf_bs.config import load_experiment_config
from physnerf_bs.data.writer import write_simulation_dataset
from physnerf_bs.sim.simulator import HumanoidSimulationService
from physnerf_bs.training.nerf_trainer import NerfTrainer


def make_smoke_config(root: Path):
    return load_experiment_config(
        "configs/smoke.yaml",
        overrides={
            "runtime": {"output_root": str(root), "device": "cpu"},
            "data": {"root": str(root / "dataset")},
            "nerf": {"steps": 2, "eval_every": 1},
        },
    )


def test_pose_nerf_training_smoke() -> None:
    root = Path("outputs/test_runs/test_nerf")
    shutil.rmtree(root, ignore_errors=True)
    config = make_smoke_config(root)
    service = HumanoidSimulationService(config.simulation, seed=config.runtime.seed)
    bundles = service.generate_dataset()
    write_simulation_dataset(bundles, config.data, n_cameras=config.simulation.n_cameras)

    trainer = NerfTrainer(
        dataset_root=config.data.root,
        data_config=config.data,
        nerf_config=config.nerf,
        physics_config=config.physics,
        eval_config=config.evaluation,
        device=config.runtime.device,
    )
    result = trainer.train(Path(config.runtime.output_root) / "nerf")
    assert "evaluation" in result["metrics"]
    assert (Path(config.runtime.output_root) / "nerf" / "metrics.json").exists()
