# HW3: Two-View Structure from Motion (SIFT + Essential Matrix + Triangulation)

This project implements a complete two-view SfM pipeline in `main.ipynb` and exports an interactive 3D scene for Viser.

---

## Project Structure

```text
rgb-img-merging/
├─ main.ipynb                 # End-to-end HW3 pipeline (steps 0-5 + exports + interactive viewer call)
├─ intrinsics.py              # Camera intrinsic matrix K computation from camera/sensor specs
├─ features.py                # SIFT feature extraction + BFMatcher NNDR matching
├─ ransac.py                  # Essential matrix estimation with RANSAC + Sampson inlier scoring
├─ triangulation.py           # E decomposition, pose recovery, triangulation, reprojection filtering
├─ visualize_viser.py         # Interactive 3D viewer for exported scene (.npz)
├─ images/
│  ├─ input/                  # Input image pair (img1.jpeg, img2.jpeg)
│  └─ output/
│     └─ hw3/                 # Generated plots, 3-view screenshots, and scene npz
├─ hw3.pdf                    # Assignment/reference document
└─ README.md
```

---

## Pipeline Architecture

### Step 0: Load Input Images
- Reads RGB + grayscale versions of the two input images.
- Creates output directory: `images/output/hw3`.

### Step 1: Camera Intrinsics (`intrinsics.py`)
- `compute_K(...)` converts focal length + sensor size + image size into intrinsic matrix `K`.

### Step 2: SIFT Features (`features.py`)
- `get_sift_features(...)` extracts keypoints/descriptors.
- Applies edge discard filtering and returns keypoints in `(row, col)` format.

### Step 3: Feature Matching + NNDR (`features.py`)
- `match_features(...)` performs BFMatcher (`L2`) 2-NN matching.
- Applies Lowe ratio test (NNDR).
- Notebook visualizes:
	- NNDR histogram,
	- pre-RANSAC correspondence overlay,
	- top-5 descriptor matches by lowest NNDR.

### Step 4: Essential Matrix RANSAC (`ransac.py`)
- Estimates `E` using normalized 8-point algorithm under RANSAC.
- Uses Sampson distance for inlier selection.
- Recovers relative pose `(R, t)` from refined `E`.

### Step 5: Triangulation (`triangulation.py`)
- Triangulates inlier correspondences.
- Filters with depth + reprojection-error constraints.
- Produces sparse 3D cloud and visualization plots.

### Scene Export + Interactive Viewer (`visualize_viser.py`)
- Exports `step5_scene_data.npz` from notebook.
- Loads `.npz` into Viser with toggles for point cloud, cameras, camera images, and baseline.

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
- `opencv-python`
- `viser`
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
pip install numpy matplotlib pillow opencv-python viser ipykernel
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy matplotlib pillow opencv-python viser ipykernel
```

---

## Usage

### Run the Notebook Pipeline
1. Open `main.ipynb`.
2. Run cells in order from top to bottom.
3. Outputs are written to `images/output/hw3/`.

### Launch Interactive Viser (from Notebook)
- The notebook imports `visualize_scene` from `visualize_viser.py` and starts server with:
	- `server = visualize_scene(scene_npz, port=8081, block=False)`

### Launch Interactive Viser (CLI)

```bash
python visualize_viser.py images/output/hw3/step5_scene_data.npz
```

Open the printed local URL in your browser.

---

## Implementation Checklist
- [x] Step 0: Load and display input images
- [x] Step 1: Compute intrinsic matrix `K`
- [x] Step 2: Detect SIFT features in both views
- [x] Step 3: Match descriptors with NNDR + visualize top-5 NNDR matches
- [x] Step 4: Estimate `E` with RANSAC and recover `(R, t)`
- [x] Step 5: Triangulate inlier correspondences and filter 3D points
- [x] Save 3 triangulation views with camera frustums
- [x] Export `.npz` scene and view in Viser

---

## Notes
- Coordinate convention in matching/triangulation code is primarily `(row, col)` in notebook-level arrays.
- If imports such as `cv2`, `numpy`, or `viser` appear unresolved in the editor, verify the selected Python interpreter points to `.venv`.
- `panorama_results.html` is currently not part of the HW3 SfM execution flow in `main.ipynb`.
