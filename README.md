# Neural Fields and NeRF Reconstruction

This repository contains both parts of the project:

- Part 1: fit a 2D coordinate-based neural field to a single RGB image
- Part 2: train a coarse-to-fine NeRF for 3D reconstruction and novel-view rendering

The repo includes notebooks, modular Python helpers, saved output artifacts, and HTML reports for both parts.

---

## Project Structure

```text
rgb-img-merging/
|-- README.md
|-- main_part1.ipynb
|-- main_part2.ipynb
|-- dataset_3d.py
|-- nerf_model.py
|-- rendering.py
|-- train_part2.py
|-- part2_utils.py
|-- visualize_viser.py
|-- lego_200x200.npz
|-- part1_results_report.html
|-- part1_results_diff_report.html
|-- part2_results_report.html
|-- combined_results_report.html
`-- images/
    |-- input/
    `-- output/
        |-- part1_neural_field/
        `-- part2_3d_reconstruction/
```

---

## Part 1 Summary

Part 1 fits an MLP with sinusoidal positional encoding directly to image coordinates so the network can reconstruct a 2D RGB image.

### Part 1 pipeline

1. Load the source image from `images/input/part1/img_1.jpg`.
2. Normalize RGB values to `[0, 1]`.
3. Build a full normalized coordinate grid `(x, y)`.
4. Apply sinusoidal positional encoding with frequency level `L`.
5. Train an MLP to map encoded coordinates to RGB values.
6. Randomly sample pixels each iteration for efficient optimization.
7. Optimize with MSE loss and Adam.
8. Track PSNR during training.
9. Save progression renders, final reconstructions, and PSNR curves.
10. Run a 2x2 hyperparameter comparison across two `L` values and two widths.

### Part 1 outputs

Saved in `images/output/part1_neural_field/`:

- `baseline_final.png`
- `baseline_progression.png`
- `baseline_psnr_curve.png`
- `hq_final.png`
- `hq_progression.png`
- `hq_psnr_curve.png`
- `grid_2x2_results.png`
- `report_metrics.json`

### Part 1 reported results

- Baseline PSNR: `25.43 dB`
- High-quality PSNR: `27.37 dB`
- Improvement over baseline: `+1.95 dB`
- Best grid config: `L=10`, `width=256`, `PSNR=25.70 dB`

---

## Part 2 Summary

Part 2 trains a NeRF on posed multi-view images from `lego_200x200.npz`, then renders RGB and depth novel-view trajectories.

### Part 2 pipeline

1. Load training, validation, and test camera poses from `lego_200x200.npz`.
2. Build camera intrinsics from the focal length.
3. Convert pixels to rays using the camera intrinsics and camera-to-world matrices.
4. Flatten all training rays into a ray dataset for random batch sampling.
5. Use a NeRF MLP with positional encoding for 3D points and view directions.
6. Sample coarse points along rays between near and far bounds.
7. Volume render coarse RGB and depth outputs.
8. Use hierarchical importance sampling to draw fine samples from the coarse PDF.
9. Volume render fine outputs.
10. Train with coarse + fine RGB reconstruction loss.
11. Evaluate periodically on a full validation image.
12. Save validation progress renders, PSNR curves, best checkpoint, metrics JSON, and test trajectory outputs.
13. Export separate RGB and depth outputs as `.npy`, `.png`, and `.gif`.
14. Visualize scene setup with Viser screenshots.

### Part 2 training settings used

From `images/output/part2_3d_reconstruction/report_metrics.json`:

- Steps: `5000`
- Batch rays: `2048`
- Coarse samples: `32`
- Fine samples: `32`
- Best validation PSNR: `23.95 dB`
- Best step: `4000`
- Final loss: `0.00514`
- Near / far: `2.016 / 6.047`
- Average seconds per step: `0.252`

### Part 2 outputs

Saved in `images/output/part2_3d_reconstruction/`:

- `loss_curve.png`
- `psnr_curve.png`
- `report_metrics.json`
- `checkpoint_best.pt`
- `test_rgb.gif`
- `test_depth.gif`
- `progress_renders/`
- `test_rgb_npy/`
- `test_rgb_png/`
- `test_depth_npy/`
- `test_depth_png/`
- `viser/`

---

## How To Run

### Part 1

1. Open `main_part1.ipynb`.
2. Run the notebook cells in order.
3. Saved outputs will appear in `images/output/part1_neural_field/`.

### Part 2

1. Open `main_part2.ipynb`.
2. Run training cells first.
3. Run the RGB render cell if you want RGB outputs.
4. Run the depth render cell if you want depth outputs.
5. Saved outputs will appear in `images/output/part2_3d_reconstruction/`.

---

## Reports

HTML reports included in the repo:

- `part1_results_report.html`
- `part2_results_report.html`
- `combined_results_report.html`

These summarize the saved visual outputs and metrics for both parts.

---

## Environment Setup

### Prerequisites

- Python 3.10+
- `pip`
- Jupyter or VS Code notebook support

### Recommended packages

- `numpy`
- `matplotlib`
- `pillow`
- `torch`
- `torchvision`
- `tqdm`
- `ipykernel`
- `imageio`
- `viser` for the 3D visualization workflow

### Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy matplotlib pillow torch torchvision tqdm ipykernel imageio viser
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy matplotlib pillow torch torchvision tqdm ipykernel imageio viser
```

---

## Notes

- Part 1 is a 2D neural field reconstruction task.
- Part 2 is a full 3D NeRF reconstruction task with hierarchical sampling.
- Part 2 training is the heavier stage and benefits a lot from running on GPU.
- The saved HTML reports are the easiest way to review final outputs quickly.
