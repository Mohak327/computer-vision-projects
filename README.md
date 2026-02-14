# RGB Image Merging - HW2 Specification

## Project Overview
Computer vision project implementing a complete 5-step panoramic image stitching pipeline:
1. Harris Corner Detection
2. Non-Maximal Suppression (NMS)
3. Feature Descriptor Extraction (MOPS-like)
4. Feature Matching (Lowe's ratio test)
5. RANSAC Homography & Multi-Image Panorama Creation

The pipeline supports stitching 2+ images into seamless panoramas with proper alignment and alpha blending.

## Assignment Structure

### Step 0: Taking Photos (0 points, Required)
**Objective:** Capture 2 photos for panoramic stitching

**Requirements:**
- Take 2 photos as you would for a panorama
- Keep camera level - only rotate, do NOT translate
- Lock exposure between shots
- Lock focus between shots
- Ensure consistent lighting conditions

**Deliverables:**
- 2 source images ready for processing

---

### Step 1: Harris Corner Detection (5 points)
**Objective:** Implement Harris Interest Point Detector

**Technical Requirements:**
- Use Harris Interest Point Detector (Section 2)
- Single scale implementation (no multi-scale)
- Sub-pixel accuracy not required
- Can use provided sample code: `harris.py`

**Implementation Notes:**
- Start with basic Harris corner detection
- Expected output: ~5000+ corner points per image

**Deliverables:**
1. Show 2 original images side-by-side
2. Show detected corners overlaid on both images side-by-side

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
