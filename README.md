# Part 1: Fit a Neural Field to a 2D Image

This project fits a coordinate-based neural field (MLP + sinusoidal positional encoding) to an RGB image using PyTorch.

The full implementation is in `main.ipynb` and includes:
- random pixel sampling dataloader,
- MSE loss + Adam optimizer,
- PSNR tracking,
- training progression visualizations,
- 2x2 hyperparameter comparison,
- a high-quality reconstruction run.

---

## Project Structure

```
rgb-img-merging/
├── .git/
├── .gitignore
├── .venv/
├── README.md
├── main.ipynb
├── __pycache__/
└── images/
	├── input/
	│   ├── .DS_Store
	│   └── part1/
	│       └── img_1.jpg
	└── output/
		└── part1_neural_field/
			├── baseline_final.png
			├── baseline_progression.png
			├── baseline_psnr_curve.png
			├── hq_final.png
			├── hq_progression.png
			├── hq_psnr_curve.png
			└── grid_2x2_results.png
```

For Part 1, the starting image is located at `images/input/part1/img_1.jpg`.

---

## Pipeline Overview

### 1. Load and Normalize Image
- Reads `images/input/part1/img_1.jpg`.
- Normalizes RGB values to `[0, 1]`.
- Builds full normalized coordinate grid `(x, y)` in `[0, 1]`.

### 2. Positional Encoding (PE)
- Applies sinusoidal PE with max frequency level `L`.
- Uses concatenation: original coordinates + `sin/cos` frequency bands.

### 3. Neural Field MLP
- Input: encoded 2D coordinates.
- Hidden layers: ReLU activations.
- Output layer: 3 channels with Sigmoid to keep RGB in `[0, 1]`.

### 4. Random Pixel Sampling Dataloader
- Each iteration samples `N` random pixels.
- Returns `N x 2` coordinates and corresponding `N x 3` RGB targets.

### 5. Optimization + Metrics
- Loss: MSE (`torch.nn.functional.mse_loss`).
- Optimizer: Adam.
- Metric: PSNR (`-10 * log10(MSE)`).
- Includes LR decay and best-checkpoint restore for higher reconstruction quality.

### 6. Deliverables Implemented in Notebook
- Model architecture report (layers, width, LR, etc.).
- Training progression images at multiple steps.
- 2x2 hyperparameter grid over two `L` values and two width values.
- PSNR curve for a selected run.

---

## Development Setup

### Prerequisites
- Python 3.10+ (3.11 recommended)
- `pip`
- Jupyter support (VS Code notebook or JupyterLab)

Required Python packages:
- `numpy`
- `matplotlib`
- `pillow`
- `torch`
- `ipykernel`

### Installation

```bash
# from project root
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy matplotlib pillow torch torchvision tqdm ipykernel
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy matplotlib pillow torch torchvision tqdm ipykernel
```

---

## Usage

### Run the Notebook
1. Open `main.ipynb`.
2. Run cells in order from top to bottom.
3. Baseline, HQ, and grid outputs are saved to `images/output/part1_neural_field/`.

### Quick/Fast Development Mode
- In the baseline run cell, keep `FAST_DEV_RUN = True` for short tests.
- Set `FAST_DEV_RUN = False` for full-quality training.

---

## Current Results
- Baseline PSNR: `25.43 dB`
- High-quality PSNR: `28.38 dB`
- Improvement over baseline: `+2.95 dB`

---

## Notes
- Training is CPU-heavy at full settings; full HQ run can take around 10-20 minutes depending on hardware.
- If imports are unresolved, verify the selected interpreter is `.venv`.
