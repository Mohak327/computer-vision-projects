"""
Step 1: Harris Corner Detection
Detects interest points (corners) in images using Harris corner detector.
"""

import numpy as np
from skimage.feature import corner_harris, peak_local_max


class HarrisDetector:
    """
    Harris Corner Detector for finding interest points in images.
    """
    
    def __init__(self, edge_discard=20, min_distance=1, sigma=1):
        """
        Initialize Harris Detector with parameters.
        
        Args:
            edge_discard: Pixels to discard from edges (minimum 20)
            min_distance: Minimum distance between detected corners
            sigma: Standard deviation for Gaussian kernel
        """
        assert edge_discard >= 20, "edge_discard must be at least 20"
        self.edge_discard = edge_discard
        self.min_distance = min_distance
        self.sigma = sigma
        
    def detect_corners(self, im):
        """
        Detect Harris corners in a grayscale image.
        
        Args:
            im: Grayscale image (2D numpy array)
            
        Returns:
            h: Harris corner response map (same shape as im)
            coords: 2 x n array of corner coordinates (ys, xs)
        """
        # Find harris corners
        h = corner_harris(im, method='eps', sigma=self.sigma)
        coords = peak_local_max(h, min_distance=self.min_distance)
        
        # Discard points on edge
        edge = self.edge_discard
        mask = (coords[:, 0] > edge) & \
               (coords[:, 0] < im.shape[0] - edge) & \
               (coords[:, 1] > edge) & \
               (coords[:, 1] < im.shape[1] - edge)
        coords = coords[mask].T
        
        return h, coords
    
    def visualize_corners(self, im, coords):
        """
        Create visualization with detected corners overlaid on image.
        
        Args:
            im: Original image
            coords: 2 x n array of corner coordinates
            
        Returns:
            Visualization array
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(im, cmap='gray')
        ax.plot(coords[1], coords[0], 'r+', markersize=10, markeredgewidth=2)
        ax.set_title(f'Harris Corners Detected: {coords.shape[1]} points')
        ax.axis('off')
        
        return fig
