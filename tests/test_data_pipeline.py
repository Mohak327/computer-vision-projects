from pathlib import Path
import shutil

from physnerf_bs.config import load_experiment_config
from physnerf_bs.data.datasets import FrameDataset, VisuomotorSequenceDataset, load_dataset_bundle
from physnerf_bs.data.writer import write_simulation_dataset
from physnerf_bs.sim.simulator import HumanoidSimulationService


def make_smoke_config(root: Path):
    return load_experiment_config(
        "configs/smoke.yaml",
        overrides={
            "runtime": {"output_root": str(root), "device": "cpu"},
            "data": {"root": str(root / "dataset")},
        },
    )


def test_dataset_generation_smoke() -> None:
    root = Path("outputs/test_runs/test_dataset")
    shutil.rmtree(root, ignore_errors=True)
    config = make_smoke_config(root)
    service = HumanoidSimulationService(config.simulation, seed=config.runtime.seed)
    bundles = service.generate_dataset()
    result = write_simulation_dataset(bundles, config.data, n_cameras=config.simulation.n_cameras)

    assert result.frames_manifest.exists()
    assert result.sequences_manifest.exists()
    bundle = load_dataset_bundle(result.root)
    assert bundle.frames
    assert bundle.sequences

    frame_dataset = FrameDataset(result.root, split="train", preload_images=False)
    seq_dataset = VisuomotorSequenceDataset(result.root, split="train", clip_length=config.data.clip_length)
    sample = frame_dataset[0]
    clip = seq_dataset[0]
    assert sample["rgb"].shape[-1] == 3
    assert clip["frames"].shape[0] == config.data.clip_length
