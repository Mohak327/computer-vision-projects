# Running PhysNeRF-BS

This guide is written for PowerShell on Windows, using the current repo layout.

## 1. Open the project

```powershell
cd C:\Users\mohak\Documents\PROJECTS\rgb-img-merging
```

## 2. Pick the Python interpreter

If you installed Python 3.12 on Windows, do not assume `python` now points to it.
On this machine, `python` was still resolving to Python 3.13 after 3.12 was installed.

Check the exact interpreter you want:

```powershell
& "C:\Users\mohak\AppData\Local\Programs\Python\Python312\python.exe" --version
```

Expected result:

```text
Python 3.12.x
```

If plain `python --version` still shows `3.13.x`, use the full Python 3.12 path in every setup command below.

## 3. Create and activate a virtual environment

Preferred command:

```powershell
& "C:\Users\mohak\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If that works, the activated environment should now use Python 3.12:

```powershell
python --version
python -m pip --version
```

If `venv` creation fails on this machine because of temp-directory permissions, use Python 3.12 directly without a venv:

```powershell
& "C:\Users\mohak\AppData\Local\Programs\Python\Python312\python.exe" -m pip install --upgrade pip
```

## 4. Install the package and dependencies

The simplest one-shot install is:

```powershell
python -m pip install -r requirements.txt
```

This installs:
- the package itself
- dev tools like `pytest`
- MuJoCo / `dm_control` dependencies
- metric extras like `lpips`

If you want lighter installs, you can still use extras directly:

For smoke tests and the synthetic backend only:

```powershell
python -m pip install -e ".[dev]"
```

For the full MuJoCo / `dm_control` path without the metrics extra:

```powershell
python -m pip install -e ".[dev,sim]"
```

Important:
- Use `python -m pip`, not bare `pip`, so the install goes to the same interpreter you just checked.
- If you are not using a venv, replace `python` with the full 3.12 path.

Example without venv:

```powershell
& "C:\Users\mohak\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt
```

## 5. Confirm the install works

```powershell
python -c "import physnerf_bs; print('physnerf_bs import OK')"
```

## 6. Run the smoke tests

These use the synthetic backend and the small config in `configs/smoke.yaml`.

```powershell
pytest -q
```

Expected result:

```text
3 passed
```

## 7. Generate a small dataset manually

```powershell
generate_dataset --config configs/smoke.yaml
```

This writes data under:

```text
outputs/smoke/dataset
```

Key files to inspect:
- `outputs/smoke/dataset/frames.jsonl`
- `outputs/smoke/dataset/sequences.json`
- `outputs/smoke/dataset/dataset_summary.json`

## 8. Train only the NeRF model

Pose-conditioned NeRF with smoke settings:

```powershell
train_nerf --config configs/smoke.yaml
```

Outputs go under:

```text
outputs/smoke/nerf_pose
```

If you want the static baseline instead:

```powershell
train_nerf --config configs/smoke.yaml --mode static
```

If you want pose-conditioned NeRF with physics enabled:

```powershell
train_nerf --config configs/smoke.yaml --mode pose_physics
```

Important output files:
- `metrics.json`
- `checkpoint_best.pt`
- `val_render_step_0001.png` and later renders

## 9. Train only the world model

```powershell
train_world_model --config configs/smoke.yaml
```

Outputs go under:

```text
outputs/smoke/world_model
```

Important output files:
- `metrics.json`
- `checkpoint_best.pt`

## 10. Run the full end-to-end smoke pipeline

This is the easiest end-to-end command to start testing everything:

```powershell
run_ablation_suite --config configs/smoke.yaml
```

This runs:
- dataset generation
- `static_nerf`
- `pose_nerf`
- `phys_nerf`
- `world_model`

Main summary file:

```text
outputs/smoke/pipeline_summary.json
```

## 11. Run evaluation commands

NeRF evaluation:

```powershell
evaluate_nerf --config configs/smoke.yaml
```

World-model evaluation:

```powershell
evaluate_world_model --config configs/smoke.yaml
```

Saved outputs:
- `outputs/smoke/nerf_eval_metrics.json`
- `outputs/smoke/world_model_eval_metrics.json`

## 12. Switch from smoke mode to fuller runs

Use the default config:

```powershell
run_ablation_suite --config configs/default.yaml
```

Before doing that, make sure:
- you installed `.[dev,sim]`
- `dm_control` and `mujoco` import correctly
- you have enough runtime for larger dataset generation and training

## 13. Most useful files to inspect while debugging

- `configs/smoke.yaml`
- `configs/default.yaml`
- `src/physnerf_bs/cli.py`
- `src/physnerf_bs/training/orchestrator.py`
- `src/physnerf_bs/training/nerf_trainer.py`
- `src/physnerf_bs/training/world_model_trainer.py`

## 14. Recommended first test sequence

If you want the cleanest first pass, run these in order:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
generate_dataset --config configs/smoke.yaml
train_nerf --config configs/smoke.yaml --mode pose_physics
train_world_model --config configs/smoke.yaml
run_ablation_suite --config configs/smoke.yaml
```

## 15. If a command fails

- If `python --version` shows `3.13.x`, you are still using the wrong interpreter. Use:

```powershell
& "C:\Users\mohak\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt
```

- If `physnerf_bs` cannot be imported, rerun:

```powershell
python -m pip install -r requirements.txt
```

- If MuJoCo / `dm_control` fails, use `configs/smoke.yaml` first, since it forces the synthetic backend.

- If you want a fully fresh smoke output directory, delete:

```text
outputs/smoke
```

then rerun the commands.
