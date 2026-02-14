"""
Step 3: Feature Matching
Matches feature descriptors between two images.
"""

import numpy as np


def dist_SSD(x, c):
    """
    Calculate squared Euclidean distance between two sets of points.
    
    Args:
        x: m x n array of m points in n dimensions
        c: l x n array of l points in n dimensions
        
    Returns:
        m x l array of squared distances
    
    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'
    
    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((ndata, 1)) * np.sum((c**2).T, axis=0) - \
            2 * np.inner(x, c)


class FeatureMatcher:
    """
    Feature Matcher for finding correspondences between two sets of descriptors.
    """
    
    def __init__(self, threshold_ratio=0.8):
        """
        Initialize Feature Matcher.
        
        Args:
            threshold_ratio: Ratio threshold for Lowe's ratio test (default 0.8)
        """
        self.threshold_ratio = threshold_ratio
        
    def match_features(self, desc1, desc2, coords1, coords2):
        """
        Match features between two images using nearest neighbor distance ratio.
        
        Args:
            desc1: m1 x d array of descriptors from image 1
            desc2: m2 x d array of descriptors from image 2
            coords1: 2 x m1 array of coordinates from image 1
            coords2: 2 x m2 array of coordinates from image 2
            
        Returns:
            matches: k x 2 array of matched indices (index in desc1, index in desc2)
            match_coords1: 2 x k array of matched coordinates from image 1
            match_coords2: 2 x k array of matched coordinates from image 2
        """
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Compute pairwise distances
        distances = dist_SSD(desc1, desc2)
        
        matches = []
        match_coords1 = []
        match_coords2 = []
        
        # For each descriptor in image 1, find best matches in image 2
        for i in range(desc1.shape[0]):
            # Get distances for this descriptor
            dists = distances[i, :]
            
            # Find two nearest neighbors
            sorted_indices = np.argsort(dists)
            best_idx = sorted_indices[0]
            second_best_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
            
            # Lowe's ratio test
            if second_best_idx is not None:
                ratio = dists[best_idx] / dists[second_best_idx]
                if ratio < self.threshold_ratio:
                    matches.append([i, best_idx])
                    match_coords1.append(coords1[:, i])
                    match_coords2.append(coords2[:, best_idx])
            else:
                # Only one match available
                matches.append([i, best_idx])
                match_coords1.append(coords1[:, i])
                match_coords2.append(coords2[:, best_idx])
        
        matches = np.array(matches) if matches else np.array([])
        match_coords1 = np.array(match_coords1).T if match_coords1 else np.array([])
        match_coords2 = np.array(match_coords2).T if match_coords2 else np.array([])
        
        return matches, match_coords1, match_coords2
    
    def visualize_matches(self, im1, im2, coords1, coords2):
        """
        Visualize matched features between two images.
        
        Args:
            im1: First image
            im2: Second image
            coords1: 2 x k array of matched coordinates from image 1
            coords2: 2 x k array of matched coordinates from image 2
            
        Returns:
            Visualization figure
        """
        import matplotlib.pyplot as plt
        
        # Create side-by-side visualization
        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Concatenate images horizontally
        if len(im1.shape) == 2:
            combined = np.hstack([im1, im2])
        else:
            combined = np.hstack([im1, im2])
        
        ax.imshow(combined, cmap='gray' if len(im1.shape) == 2 else None)
        
        # Draw matches
        if coords1.size > 0 and coords2.size > 0:
            for i in range(coords1.shape[1]):
                y1, x1 = coords1[0, i], coords1[1, i]
                y2, x2 = coords2[0, i] + 0, coords2[1, i] + w1  # Offset x2 by width of first image
                
                ax.plot([x1, x2], [y1, y2], 'g-', linewidth=0.5, alpha=0.5)
                ax.plot(x1, y1, 'ro', markersize=3)
                ax.plot(x2, y2, 'bo', markersize=3)
        
        ax.set_title(f'Feature Matches: {coords1.shape[1]} correspondences')
        ax.axis('off')
        
        return fig
