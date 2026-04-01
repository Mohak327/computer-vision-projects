from pathlib import Path
import shutil

from physnerf_bs.config import load_experiment_config
from physnerf_bs.training.orchestrator import run_full_pipeline


def test_end_to_end_smoke_pipeline() -> None:
    root = Path("outputs/test_runs/test_pipeline")
    shutil.rmtree(root, ignore_errors=True)
    config = load_experiment_config(
        "configs/smoke.yaml",
        overrides={
            "runtime": {"output_root": str(root), "device": "cpu"},
            "data": {"root": str(root / "dataset")},
            "world_model": {"steps": 2},
            "nerf": {"steps": 2, "eval_every": 1},
        },
    )
    results = run_full_pipeline(config)
    summary_path = Path(config.runtime.output_root) / "pipeline_summary.json"
    assert summary_path.exists()
    assert "world_model" in results
    assert "phys_nerf" in results
