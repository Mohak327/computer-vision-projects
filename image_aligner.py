"""
Step 4: Image Alignment using RANSAC
Estimates transformation between images and aligns them.
"""

import numpy as np


class ImageAligner:
    """
    Image Aligner using RANSAC to estimate transformation and stitch images.
    """
    
    def __init__(self, ransac_threshold=5.0, ransac_iterations=1000):
        """
        Initialize Image Aligner.
        
        Args:
            ransac_threshold: Inlier threshold in pixels (default 5.0)
            ransac_iterations: Number of RANSAC iterations (default 1000)
        """
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
        
    def estimate_transform_ransac(self, coords1, coords2):
        """
        Estimate transformation using RANSAC.
        
        Args:
            coords1: 2 x n array of coordinates from image 1
            coords2: 2 x n array of corresponding coordinates from image 2
            
        Returns:
            best_transform: 3x3 transformation matrix
            inliers: Boolean array indicating inliers
        """
        if coords1.shape[1] < 3:
            return np.eye(3), np.array([])
        
        best_inliers = []
        best_transform = np.eye(3)
        
        for _ in range(self.ransac_iterations):
            # Randomly select 3 correspondences
            indices = np.random.choice(coords1.shape[1], 3, replace=False)
            sample_coords1 = coords1[:, indices]
            sample_coords2 = coords2[:, indices]
            
            # Estimate affine transformation from these 3 points
            transform = self._estimate_affine(sample_coords1, sample_coords2)
            
            # Count inliers
            inliers = self._compute_inliers(coords1, coords2, transform)
            
            # Update best model if this one is better
            if np.sum(inliers) > np.sum(best_inliers):
                best_inliers = inliers
                best_transform = transform
        
        # Refine with all inliers
        if np.sum(best_inliers) >= 3:
            inlier_coords1 = coords1[:, best_inliers]
            inlier_coords2 = coords2[:, best_inliers]
            best_transform = self._estimate_affine(inlier_coords1, inlier_coords2)
        
        return best_transform, best_inliers
    
    def _estimate_affine(self, coords1, coords2):
        """
        Estimate affine transformation from point correspondences.
        
        Args:
            coords1: 2 x n array of source coordinates
            coords2: 2 x n array of target coordinates
            
        Returns:
            3x3 transformation matrix
        """
        n = coords1.shape[1]
        
        # Build the system of equations Ax = b
        A = np.zeros((2 * n, 6))
        b = np.zeros(2 * n)
        
        for i in range(n):
            y1, x1 = coords1[0, i], coords1[1, i]
            y2, x2 = coords2[0, i], coords2[1, i]
            
            # First equation (for x)
            A[2*i, :] = [x1, y1, 1, 0, 0, 0]
            b[2*i] = x2
            
            # Second equation (for y)
            A[2*i+1, :] = [0, 0, 0, x1, y1, 1]
            b[2*i+1] = y2
        
        # Solve using least squares
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Construct transformation matrix
        transform = np.array([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [0, 0, 1]
        ])
        
        return transform
    
    def _compute_inliers(self, coords1, coords2, transform):
        """
        Compute inliers based on transformation.
        
        Args:
            coords1: 2 x n array of source coordinates
            coords2: 2 x n array of target coordinates
            transform: 3x3 transformation matrix
            
        Returns:
            Boolean array indicating inliers
        """
        # Transform coords1 using the transformation
        ones = np.ones((1, coords1.shape[1]))
        homog_coords1 = np.vstack([coords1[1:2, :], coords1[0:1, :], ones])  # [x, y, 1]
        transformed = transform @ homog_coords1
        transformed = transformed[:2, :] / transformed[2:3, :]
        
        # Compute distances
        target = np.vstack([coords2[1:2, :], coords2[0:1, :]])  # [x, y]
        distances = np.sqrt(np.sum((transformed - target)**2, axis=0))
        
        # Inliers are points with distance below threshold
        inliers = distances < self.ransac_threshold
        
        return inliers
    
    def stitch_images(self, im1, im2, transform):
        """
        Stitch two images together using the estimated transformation.
        
        Args:
            im1: First image (reference)
            im2: Second image (to be transformed)
            transform: 3x3 transformation matrix
            
        Returns:
            Stitched panorama image
        """
        # TODO: Implement image stitching
        # This is a placeholder that needs full implementation
        print(f"Transform matrix:\n{transform}")
        print(f"Number of inliers: (implement count)")
        
        return im1  # Placeholder
