# RGB Image Merging - HW2 Specification

---

## Project Structure
```
rgb-img-merging/
├── main.ipynb               # Main Jupyter notebook - Complete pipeline (USE THIS)
├── harris_detector.py       # Step 1: Harris Corner Detection
├── nms_suppressor.py        # Step 2: Non-Maximal Suppression
├── feature_descriptor.py    # Step 3: Feature Descriptor Extraction
├── feature_matcher.py       # Step 4: Feature Matching
├── advanced_stitcher.py     # Step 5: RANSAC Homography & Panorama Creation
├── generate_html.py         # HTML output generator
├── BELLS_AND_WHISTLES.md    # Advanced features documentation
├── images/                  # Input images directory
│   └── tajm/                # Sample Taj Mahal image set
├── output/                  # Processed results and visualizations
├── __pycache__/             # Python cache files
└── README.md                # This file
```

## Pipeline Architecture

The project is organized into 5 modular classes, one for each step:

### 1. HarrisDetector (`harris_detector.py`)
- Detects Harris corners in grayscale images
- Filters corners near image edges
- Returns corner coordinates and response map

### 2. NMSSuppressor (`nms_suppressor.py`)
- Applies Non-Maximal Suppression to corner points
- Reduces redundant nearby corners
- Selects strongest corners within local neighborhoods

### 3. FeatureDescriptor (`feature_descriptor.py`)
- Extracts feature descriptors around detected corners
- Creates normalized MOPS-like patch descriptors from color images
- Handles boundary cases

### 4. FeatureMatcher (`feature_matcher.py`)
- Matches descriptors between image pairs
- Uses SSD distance and Lowe's ratio test
- Visualizes correspondences with proper aspect ratios

### 5. AdvancedStitcher (`advanced_stitcher.py`)
- Estimates homography transformation using RANSAC
- Creates panoramas with alpha blending
- Supports multi-image stitching (3+ images)
- Includes bells & whistles: ANMS, rotation-invariant descriptors

## Development Setup

### Prerequisites
- Python 3.x
- NumPy
- PIL/Pillow
- SciPy
- scikit-image
- Matplotlib

### Installation
```bash
pip install numpy pillow scipy scikit-image matplotlib
```

### Usage

**Using Jupyter Notebook (Primary Method)**
```bash
jupyter notebook main.ipynb
```
Run cells interactively to see results for each step:
1. Load images from `images/tajm/` folder
2. Step 1: Harris Corner Detection
3. Step 2: Non-Maximal Suppression
4. Step 3: Feature Descriptor Extraction
5. Step 4: Feature Matching (with visualizations)
6. Step 5: RANSAC Homography & Panorama Creation

All results are automatically saved to `output/` folder.

## Implementation Checklist
- [x] Step 0: Capture panoramic photo set (4 images of Taj Mahal)
- [x] Step 1: Implement Harris corner detection
- [x] Step 1: Visualize detected corners
- [x] Step 2: Implement Non-Maximal Suppression
- [x] Step 2: Visualize NMS results
- [x] Step 3: Extract feature descriptors (color MOPS-like)
- [x] Step 3: Visualize sample descriptors
- [x] Step 4: Match features with Lowe's ratio test
- [x] Step 4: Show NNR histogram and matches for all pairs
- [x] Step 5: RANSAC homography estimation
- [x] Step 5: Multi-image panorama stitching with alpha blending
- [x] Generate visualizations for all steps

## Notes
- Place your images in `images/` directory (organized in subfolders like `images/tajm/`)
- All processed outputs are automatically saved to `output/` directory with descriptive names
- The notebook generates visualizations for each step of the pipeline
- Feature matches are visualized with white borders separating the two images
- Multi-image stitching uses a right-to-left approach for optimal results
- See `BELLS_AND_WHISTLES.md` for advanced features and implementation details
