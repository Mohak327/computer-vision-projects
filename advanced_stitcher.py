"""
Bells & Whistles: Advanced Image Stitching Features
Implements ANMS, RANSAC Homography, Panorama Creation, Rotation Invariance, and Multi-image Support
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.distance import cdist
import cv2


class AdvancedStitcher:
    """
    Advanced image stitching with bells and whistles features.
    """
    
    def __init__(self, ransac_threshold=5.0, ransac_iterations=1000):
        """
        Initialize Advanced Stitcher.
        
        Args:
            ransac_threshold: Distance threshold for RANSAC inliers
            ransac_iterations: Number of RANSAC iterations
        """
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
    
    # ============================================================
    # BELL 1: Adaptive Non-Maximal Suppression (ANMS)
    # ============================================================
    
    def adaptive_non_maximal_suppression(self, corners, response_map, num_corners=500):
        """
        Implement Adaptive Non-Maximal Suppression (ANMS).
        
        Instead of using a fixed window size, ANMS selects corners based on
        their suppression radius - the minimum distance to another corner
        with significantly higher response.
        
        Args:
            corners: 2 x n array of corner coordinates (y, x)
            response_map: Harris response strength map
            num_corners: Target number of corners to keep
            
        Returns:
            selected_corners: 2 x m array of selected corners (m <= num_corners)
        """
        if corners.shape[1] <= num_corners:
            return corners
        
        n = corners.shape[1]
        radii = np.inf * np.ones(n)
        
        # Get response values for all corners
        responses = np.array([response_map[int(corners[0, i]), int(corners[1, i])] 
                              for i in range(n)])
        
        # For each corner, find minimum distance to another corner with
        # significantly higher response (c_robust = 0.9)
        c_robust = 0.9
        
        for i in range(n):
            for j in range(n):
                if responses[j] > responses[i] / c_robust:
                    # Calculate Euclidean distance
                    dist = np.sqrt((corners[0, i] - corners[0, j])**2 + 
                                   (corners[1, i] - corners[1, j])**2)
                    if dist < radii[i]:
                        radii[i] = dist
        
        # Sort corners by suppression radius (descending) and select top num_corners
        sorted_indices = np.argsort(-radii)[:num_corners]
        selected_corners = corners[:, sorted_indices]
        
        return selected_corners
    
    # ============================================================
    # BELL 2: RANSAC Homography Estimation
    # ============================================================
    
    def estimate_homography(self, src_pts, dst_pts):
        """
        Estimate homography matrix from point correspondences.
        
        Args:
            src_pts: 2 x 4 array of source points
            dst_pts: 2 x 4 array of destination points
            
        Returns:
            H: 3 x 3 homography matrix
        """
        # Need at least 4 points for homography
        assert src_pts.shape[1] >= 4 and dst_pts.shape[1] >= 4
        
        # Build the constraint matrix A for Ah = 0
        A = []
        for i in range(4):
            x, y = src_pts[1, i], src_pts[0, i]  # x, y (note: coords are [y, x])
            u, v = dst_pts[1, i], dst_pts[0, i]  # u, v
            
            A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        
        A = np.array(A)
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        
        # Normalize so that H[2,2] = 1
        H = H / H[2, 2]
        
        return H
    
    def apply_homography(self, H, points):
        """
        Apply homography transformation to points.
        
        Args:
            H: 3 x 3 homography matrix
            points: 2 x n array of points (y, x)
            
        Returns:
            transformed_points: 2 x n array of transformed points
        """
        n = points.shape[1]
        
        # Convert to homogeneous coordinates (x, y, 1) - note swap y,x to x,y
        homogeneous = np.vstack([points[1, :], points[0, :], np.ones(n)])
        
        # Apply homography
        transformed = H @ homogeneous
        
        # Convert back to Cartesian coordinates and swap back to (y, x)
        transformed_points = np.vstack([
            transformed[1, :] / transformed[2, :],
            transformed[0, :] / transformed[2, :]
        ])
        
        return transformed_points
    
    def ransac_homography(self, src_coords, dst_coords):
        """
        Estimate homography using RANSAC.
        
        Args:
            src_coords: 2 x n array of source coordinates (y, x)
            dst_coords: 2 x n array of destination coordinates (y, x)
            
        Returns:
            best_H: Best homography matrix
            inliers: Boolean array indicating inliers
        """
        n = src_coords.shape[1]
        best_inliers = np.zeros(n, dtype=bool)
        best_H = np.eye(3)
        max_inliers = 0
        
        for _ in range(self.ransac_iterations):
            # Randomly select 4 correspondences
            indices = np.random.choice(n, 4, replace=False)
            src_sample = src_coords[:, indices]
            dst_sample = dst_coords[:, indices]
            
            try:
                # Estimate homography from sample
                H = self.estimate_homography(src_sample, dst_sample)
                
                # Transform all source points
                transformed = self.apply_homography(H, src_coords)
                
                # Calculate distances to destination points
                distances = np.sqrt(np.sum((transformed - dst_coords)**2, axis=0))
                
                # Count inliers
                inliers = distances < self.ransac_threshold
                num_inliers = np.sum(inliers)
                
                # Update best model if this is better
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_inliers = inliers
                    best_H = H
            except:
                continue
        
        # Re-estimate homography using all inliers
        if max_inliers >= 4:
            try:
                best_H = self.estimate_homography(
                    src_coords[:, best_inliers], 
                    dst_coords[:, best_inliers]
                )
            except:
                pass
        
        return best_H, best_inliers
    
    # ============================================================
    # BELL 3: Panorama Creation
    # ============================================================
    
    def warp_image(self, img, H, output_shape):
        """
        Warp image using homography.
        
        Args:
            img: Input image (H x W x C or H x W)
            H: Homography matrix
            output_shape: (height, width) of output image
            
        Returns:
            warped: Warped image
        """
        h, w = output_shape
        
        # Create coordinate grid for output image
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.vstack([x_coords.ravel(), y_coords.ravel(), np.ones(h * w)])
        
        # Apply inverse homography to find source coordinates
        H_inv = np.linalg.inv(H)
        src_coords = H_inv @ coords
        src_coords = src_coords / src_coords[2, :]
        
        # Reshape to grid
        src_x = src_coords[0, :].reshape(h, w)
        src_y = src_coords[1, :].reshape(h, w)
        
        # Interpolate
        if len(img.shape) == 3:  # Color image
            warped = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
            for c in range(img.shape[2]):
                warped[:, :, c] = map_coordinates(
                    img[:, :, c], [src_y, src_x], 
                    order=1, mode='constant', cval=0
                )
        else:  # Grayscale
            warped = map_coordinates(
                img, [src_y, src_x], 
                order=1, mode='constant', cval=0
            )
        
        return warped
    
    def create_panorama(self, img1, img2, H):
        """
        Create panorama by stitching two images with alpha blending.
        
        Args:
            img1: First image (left/reference image)
            img2: Second image (right image to be warped)
            H: Homography from img2 to img1's coordinate system
            
        Returns:
            panorama: Stitched panoramic image
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Find corners of img2 in img1's coordinate system
        corners_img2 = np.array([
            [0, 0, h2-1, h2-1],
            [0, w2-1, 0, w2-1]
        ], dtype=float)
        
        transformed_corners = self.apply_homography(H, corners_img2)
        
        # Determine output size - find bounding box of both images
        all_x = np.concatenate([
            [0, w1],
            transformed_corners[1, :]
        ])
        all_y = np.concatenate([
            [0, h1],
            transformed_corners[0, :]
        ])
        
        min_x = int(np.floor(all_x.min()))
        max_x = int(np.ceil(all_x.max()))
        min_y = int(np.floor(all_y.min()))
        max_y = int(np.ceil(all_y.max()))
        
        # Calculate output dimensions and offset
        output_w = max_x - min_x
        output_h = max_y - min_y
        offset_x = -min_x
        offset_y = -min_y
        
        # Create translation matrix to shift to positive coordinates
        T = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y],
            [0, 0, 1]
        ], dtype=float)
        
        # Warp img2 with combined transformation
        H_total = T @ H
        # Warp img2 with combined transformation
        H_total = T @ H
        warped_img2 = self.warp_image(img2, H_total, (output_h, output_w))
        
        # Create output panorama and place img1
        is_color = len(img1.shape) == 3
        if is_color:
            panorama = np.zeros((output_h, output_w, 3), dtype=img1.dtype)
        else:
            panorama = np.zeros((output_h, output_w), dtype=img1.dtype)
        
        # Place img1 at the offset position
        panorama[offset_y:offset_y+h1, offset_x:offset_x+w1] = img1
        
        # Create masks for blending
        mask1 = np.zeros((output_h, output_w), dtype=float)
        mask1[offset_y:offset_y+h1, offset_x:offset_x+w1] = 1.0
        
        if is_color:
            mask2 = (warped_img2.sum(axis=2) > 0).astype(float)
        else:
            mask2 = (warped_img2 > 0).astype(float)
        
        # Alpha blending with distance-based feathering
        from scipy.ndimage import distance_transform_edt
        
        dist1 = distance_transform_edt(mask1)
        dist2 = distance_transform_edt(mask2)
        
        # Find overlap region
        overlap = (mask1 > 0) & (mask2 > 0)
        
        # Initialize alpha channels
        alpha1 = mask1.copy()
        alpha2 = mask2.copy()
        
        # In overlap region, blend based on distance from edges
        if np.any(overlap):
            alpha1[overlap] = dist1[overlap] / (dist1[overlap] + dist2[overlap] + 1e-10)
            alpha2[overlap] = dist2[overlap] / (dist1[overlap] + dist2[overlap] + 1e-10)
        
        # Blend images
        if is_color:
            for c in range(3):
                panorama[:, :, c] = (
                    alpha1 * panorama[:, :, c] + 
                    alpha2 * warped_img2[:, :, c]
                ).astype(img1.dtype)
        else:
            panorama = (alpha1 * panorama + alpha2 * warped_img2).astype(img1.dtype)
        
        return panorama
    
    # ============================================================
    # BELL 4: Rotation Invariant Descriptors
    # ============================================================
    
    def compute_orientation(self, img, y, x, patch_size=40):
        """
        Compute dominant orientation of gradient at a point.
        
        Args:
            img: Grayscale image
            y, x: Point coordinates
            patch_size: Size of patch to analyze
            
        Returns:
            orientation: Dominant orientation in radians
        """
        half_size = patch_size // 2
        
        # Extract patch
        y1, y2 = max(0, int(y - half_size)), min(img.shape[0], int(y + half_size))
        x1, x2 = max(0, int(x - half_size)), min(img.shape[1], int(x + half_size))
        
        if y2 - y1 < patch_size or x2 - x1 < patch_size:
            return 0.0
        
        patch = img[y1:y2, x1:x2]
        
        # Compute gradients
        dy = np.gradient(patch, axis=0)
        dx = np.gradient(patch, axis=1)
        
        # Compute gradient magnitudes and orientations
        magnitudes = np.sqrt(dx**2 + dy**2)
        orientations = np.arctan2(dy, dx)
        
        # Create orientation histogram (36 bins)
        hist, bin_edges = np.histogram(
            orientations, bins=36, range=(-np.pi, np.pi), 
            weights=magnitudes
        )
        
        # Find dominant orientation
        dominant_bin = np.argmax(hist)
        dominant_orientation = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2
        
        return dominant_orientation
    
    def extract_rotation_invariant_descriptor(self, img, coords, patch_size=40, 
                                               descriptor_size=8):
        """
        Extract rotation-invariant MOPS-like descriptors.
        
        Args:
            img: Input image (color or grayscale)
            coords: 2 x n array of coordinates (y, x)
            patch_size: Size of patch to extract
            descriptor_size: Size of descriptor grid
            
        Returns:
            descriptors: n x d array of descriptors
            valid_coords: 2 x m array of valid coordinates
            orientations: Array of dominant orientations
        """
        from scipy.ndimage import rotate, zoom
        
        half_size = patch_size // 2
        descriptors = []
        valid_coords = []
        orientations = []
        
        # Convert to grayscale for orientation computation if needed
        if len(img.shape) == 3:
            img_gray = np.mean(img, axis=2)
        else:
            img_gray = img
        
        for i in range(coords.shape[1]):
            y, x = int(coords[0, i]), int(coords[1, i])
            
            # Check bounds
            if (y - half_size >= 0 and y + half_size < img.shape[0] and
                x - half_size >= 0 and x + half_size < img.shape[1]):
                
                # Compute dominant orientation
                theta = self.compute_orientation(img_gray, y, x, patch_size)
                orientations.append(theta)
                
                # Extract and rotate patch
                patch = img[y - half_size:y + half_size, 
                           x - half_size:x + half_size]
                
                # Rotate patch to canonical orientation
                angle_deg = -np.degrees(theta)
                if len(patch.shape) == 3:
                    rotated_patch = np.zeros_like(patch)
                    for c in range(patch.shape[2]):
                        rotated_patch[:, :, c] = rotate(
                            patch[:, :, c], angle_deg, 
                            reshape=False, mode='constant', cval=0
                        )
                else:
                    rotated_patch = rotate(
                        patch, angle_deg, 
                        reshape=False, mode='constant', cval=0
                    )
                
                # Downsample
                scale = descriptor_size / patch_size
                if len(patch.shape) == 3:
                    descriptor = zoom(rotated_patch, (scale, scale, 1), order=1)
                else:
                    descriptor = zoom(rotated_patch, scale, order=1)
                
                # Normalize
                descriptor = descriptor.flatten()
                descriptor = descriptor - np.mean(descriptor)
                std = np.std(descriptor)
                if std > 0:
                    descriptor = descriptor / std
                
                descriptors.append(descriptor)
                valid_coords.append([y, x])
        
        if len(descriptors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return np.array(descriptors), np.array(valid_coords).T, np.array(orientations)
    
    # ============================================================
    # BELL 5: Multi-Image Stitching
    # ============================================================
    
    def stitch_multiple_images(self, images, matches_list):
        """
        Stitch multiple images into a panorama (right to left approach).
        
        Following the reference implementation approach:
        - Start with the rightmost two images
        - Stitch them together
        - Progressively add images from right to left
        
        Args:
            images: List of images (ordered left to right)
            matches_list: List of (coords_from_img[i], coords_from_img[i+1]) tuples
            
        Returns:
            panorama: Stitched panorama
        """
        if len(images) < 2:
            return images[0] if len(images) == 1 else None
        
        n = len(images)
        
        # Start by stitching the rightmost two images
        # For images[n-2] and images[n-1], we use matches_list[n-2]
        # matches_list[i] contains (coords from images[i], coords from images[i+1])
        coords_left, coords_right = matches_list[n-2]
        
        # Compute homography to warp images[n-1] (right) to images[n-2] (left)
        H, inliers = self.ransac_homography(coords_right, coords_left)
        print(f"Stitching images {n-1} and {n}: {np.sum(inliers)} inliers")
        
        # Create initial panorama from rightmost two images
        panorama = self.create_panorama(images[n-2], images[n-1], H)
        
        # Progressively add images from right to left
        for i in range(n-3, -1, -1):
            # Now stitch images[i] (left) with current panorama (right)
            # matches_list[i] has (coords from images[i], coords from images[i+1])
            coords_left, coords_right = matches_list[i]
            
            # We need to warp the current panorama to images[i]
            # The matches are between images[i] and images[i+1]
            # But panorama is built from images[i+1] onwards
            # So coords_right are in images[i+1] coordinate space
            H, inliers = self.ransac_homography(coords_right, coords_left)
            print(f"Stitching image {i+1} with panorama: {np.sum(inliers)} inliers")
            
            # Stitch: images[i] is left (reference), panorama is right (to be warped)
            panorama = self.create_panorama(images[i], panorama, H)
        
        return panorama
