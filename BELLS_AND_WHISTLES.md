# Bells & Whistles: Advanced Image Stitching Features

## Overview

The `AdvancedStitcher` class implements five advanced features (bells & whistles) for panoramic image stitching, providing improvements over the basic pipeline.

## Features

### 1. Adaptive Non-Maximal Suppression (ANMS)

**What it does:** Instead of using a fixed window size for NMS, ANMS selects corners based on their *suppression radius* - the minimum distance to another corner with significantly higher response.

**Benefits:**
- Better spatial distribution of features across the image
- More uniform coverage compared to fixed-window NMS
- Adapts to local feature density

**Method:** `adaptive_non_maximal_suppression(corners, response_map, num_corners=500)`

**Parameters:**
- `corners`: Detected corner coordinates (2 x n array)
- `response_map`: Harris response strength map
- `num_corners`: Target number of corners to keep

### 2. RANSAC Homography Estimation

**What it does:** Estimates a full homography transformation (8 DOF) instead of affine transformation (6 DOF) for better geometric accuracy.

**Benefits:**
- Handles perspective distortion
- Better for wide-angle panoramas
- More geometrically accurate alignment
- Robust to outliers through RANSAC

**Methods:**
- `estimate_homography(src_pts, dst_pts)`: Compute homography from 4+ point correspondences
- `ransac_homography(src_coords, dst_coords)`: Robust estimation with RANSAC
- `apply_homography(H, points)`: Transform points using homography

**Key Difference:** Homography can model perspective effects (parallel lines converging), while affine cannot.

### 3. Panorama Creation with Alpha Blending

**What it does:** Creates a complete panoramic image by warping one image to align with another and blending them seamlessly.

**Features:**
- Image warping using homography
- Distance-based alpha blending in overlap regions
- Feathering for smooth transitions
- Automatic canvas size calculation

**Method:** `create_panorama(img1, img2, H)`

**Parameters:**
- `img1`: Reference image (stays fixed)
- `img2`: Image to be warped and blended
- `H`: Homography transformation from img2 to img1

**Benefits:**
- No visible seams
- Smooth color transitions
- Handles varying lighting conditions

### 4. Rotation Invariant Descriptors

**What it does:** Extracts feature descriptors that are invariant to image rotation by normalizing to a canonical orientation.

**Process:**
1. Compute gradient orientation histogram in patch
2. Find dominant orientation
3. Rotate patch to canonical orientation
4. Extract descriptor from rotated patch

**Benefits:**
- Robust to camera rotation
- Better matching under viewpoint changes
- Useful for unordered image sets

**Method:** `extract_rotation_invariant_descriptor(img, coords, patch_size=40, descriptor_size=8)`

**Returns:**
- Descriptors (n x d array)
- Valid coordinates
- Dominant orientations for each feature

### 5. Multi-Image Stitching

**What it does:** Extends stitching to handle 3 or more images, creating wide panoramas.

**Process:**
1. Detect and match features in all consecutive image pairs
2. Estimate homographies for each pair
3. Iteratively warp and blend images
4. Build panorama from left to right (or any direction)

**Method:** `stitch_multiple_images(images, matches_list)`

**Parameters:**
- `images`: List of images to stitch
- `matches_list`: List of (src_coords, dst_coords) tuples for consecutive pairs

**Benefits:**
- Create wide-angle panoramas (> 180°)
- Flexible image ordering
- Automatic accumulation of transformations

## Usage Example

```python
from advanced_stitcher import AdvancedStitcher

# Initialize
stitcher = AdvancedStitcher(ransac_threshold=5.0, ransac_iterations=1000)

# 1. Apply ANMS
coords_anms = stitcher.adaptive_non_maximal_suppression(corners, harris_response, num_corners=250)

# 2. Estimate homography
H, inliers = stitcher.ransac_homography(match_coords1, match_coords2)

# 3. Create panorama
panorama = stitcher.create_panorama(img1, img2, H)

# 4. Extract rotation-invariant descriptors
descriptors, coords, orientations = stitcher.extract_rotation_invariant_descriptor(img, corners)

# 5. Stitch multiple images
multi_panorama = stitcher.stitch_multiple_images([img1, img2, img3], matches_list)
```

## Implementation Details

### ANMS Algorithm
- Uses suppression radius concept from Brown et al. (2005)
- Radius calculation: minimum distance to a corner with response > current_response / c_robust
- c_robust = 0.9 (tunable parameter)
- Selects top N corners with largest radii

### Homography Estimation
- Uses Direct Linear Transform (DLT) algorithm
- Builds 2n x 9 constraint matrix from n point correspondences
- Solves using SVD (last row of V^T)
- RANSAC for robust estimation with outlier rejection

### Alpha Blending
- Distance transform for each image mask
- Blend weight = distance from edge of mask
- Normalized weights in overlap region
- Per-channel blending for color images

### Rotation Invariance
- 36-bin orientation histogram
- Gaussian-weighted gradients
- Dominant orientation from histogram peak
- Bilinear interpolation for rotation

## Performance Tips

1. **ANMS**: Use num_corners=250-500 for good coverage without excessive computation
2. **RANSAC**: Higher iterations (1000-5000) for better accuracy, adjust threshold based on image resolution
3. **Panorama**: Works best with 30-50% overlap between images
4. **Rotation**: Adds ~2x computation overhead, use only if needed
5. **Multi-image**: Process in order of overlap for best results

## References

- Brown, M., & Lowe, D. G. (2005). "Automatic Panoramic Image Stitching using Invariant Features"
- Hartley, R., & Zisserman, A. (2004). "Multiple View Geometry in Computer Vision"
- Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints" (SIFT paper)

## Files

- `advanced_stitcher.py`: Main implementation
- `main.ipynb`: Demo notebook with all features
- `output/bell*_*.png`: Visualization outputs
- `output/panorama_final.png`: Final panorama result
