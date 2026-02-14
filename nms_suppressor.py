"""
Step 2: Non-Maximal Suppression (NMS)
Keeps only the local maximum corners within a window, discarding weaker corners.
"""

import numpy as np
from scipy.ndimage import maximum_filter


class NMSSuppressor:
    """
    Non-Maximal Suppression for selecting strongest Harris corners.
    Keeps only corners that are local maxima within a window.
    """
    
    def __init__(self, window_size=11, max_corners=500):
        """
        Initialize NMS Suppressor.
        
        Args:
            window_size: Size of the window for local maximum detection (default 11)
            max_corners: Maximum number of corners to keep (default 500)
        """
        self.window_size = window_size
        self.max_corners = max_corners
        
    def suppress(self, harris_response, coords):
        """
        Apply Non-Maximal Suppression to select strongest corners.
        
        Args:
            harris_response: Harris corner response map (2D array)
            coords: 2 x n array of corner coordinates (ys, xs)
            
        Returns:
            suppressed_coords: 2 x m array of suppressed corner coordinates (m <= n)
        """
        # Method 1: Use maximum filter to find local maxima
        local_max = maximum_filter(harris_response, size=self.window_size)
        
        # Create mask for pixels that are local maxima
        is_local_max = (harris_response == local_max)
        
        # Filter corners to keep only those that are local maxima
        suppressed_coords = []
        response_values = []
        
        for i in range(coords.shape[1]):
            y, x = int(coords[0, i]), int(coords[1, i])
            if is_local_max[y, x]:
                suppressed_coords.append([y, x])
                response_values.append(harris_response[y, x])
        
        if len(suppressed_coords) == 0:
            return np.array([]).reshape(2, 0)
        
        suppressed_coords = np.array(suppressed_coords).T
        response_values = np.array(response_values)
        
        # If still too many corners, keep only the strongest ones
        if suppressed_coords.shape[1] > self.max_corners:
            # Sort by response strength and keep top max_corners
            top_indices = np.argsort(response_values)[-self.max_corners:]
            suppressed_coords = suppressed_coords[:, top_indices]
        
        return suppressed_coords
    
    def visualize_suppression(self, im, coords_before, coords_after):
        """
        Visualize corners before and after NMS.
        
        Args:
            im: Original image
            coords_before: Corners before NMS
            coords_after: Corners after NMS
            
        Returns:
            Visualization figure
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Before NMS
        axes[0].imshow(im)
        axes[0].plot(coords_before[1], coords_before[0], 'ro', markersize=2, markeredgewidth=0)
        axes[0].set_title(f'Before NMS: {coords_before.shape[1]} corners')
        axes[0].axis('off')
        
        # After NMS
        axes[1].imshow(im)
        axes[1].plot(coords_after[1], coords_after[0], 'ro', markersize=2, markeredgewidth=0)
        axes[1].set_title(f'After NMS: {coords_after.shape[1]} corners')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        return fig
