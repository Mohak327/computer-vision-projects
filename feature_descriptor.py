"""
Step 2: Feature Descriptor Extraction
Extracts feature descriptors around detected interest points.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class FeatureDescriptor:
    """
    Feature Descriptor extractor for creating feature vectors around interest points.
    """
    
    def __init__(self, patch_size=40, descriptor_size=8):
        """
        Initialize Feature Descriptor extractor.
        
        Args:
            patch_size: Size of patch around each corner (default 40x40)
            descriptor_size: Size of descriptor grid (default 8x8)
        """
        self.patch_size = patch_size
        self.descriptor_size = descriptor_size
        
    def extract_descriptors(self, im, coords):
        """
        Extract feature descriptors at given coordinates.
        
        Args:
            im: Color (RGB) or grayscale image
            coords: 2 x n array of corner coordinates (ys, xs)
            
        Returns:
            descriptors: m x d array where m is number of valid corners
                        and d is descriptor dimension
            valid_coords: 2 x m array of coordinates for valid descriptors
        """
        half_patch = self.patch_size // 2
        descriptors = []
        valid_coords = []
        
        for i in range(coords.shape[1]):
            y, x = int(coords[0, i]), int(coords[1, i])
            
            # Check if patch is within image bounds
            if (y - half_patch >= 0 and y + half_patch < im.shape[0] and
                x - half_patch >= 0 and x + half_patch < im.shape[1]):
                
                # Extract patch (handles both grayscale and color)
                patch = im[y - half_patch:y + half_patch,
                          x - half_patch:x + half_patch]
                
                # Create descriptor (simple: downsampled and normalized patch)
                descriptor = self._create_descriptor(patch)
                descriptors.append(descriptor)
                valid_coords.append([y, x])
        
        if len(descriptors) == 0:
            return np.array([]), np.array([])
            
        descriptors = np.array(descriptors)
        valid_coords = np.array(valid_coords).T
        
        return descriptors, valid_coords
    
    def _create_descriptor(self, patch):
        """
        Create descriptor from patch.
        
        Args:
            patch: Image patch around interest point (grayscale or color)
            
        Returns:
            descriptor: Flattened descriptor vector
        """
        # Simple descriptor: downsample and normalize
        from scipy.ndimage import zoom
        
        # Downsample to descriptor_size x descriptor_size
        scale = self.descriptor_size / patch.shape[0]
        
        # Handle color (3 channels) and grayscale differently
        if len(patch.shape) == 3:  # Color image
            # Downsample each channel separately
            descriptor = zoom(patch, (scale, scale, 1), order=1)
        else:  # Grayscale
            descriptor = zoom(patch, scale, order=1)
        
        # Normalize to zero mean and unit variance
        descriptor = descriptor.flatten()
        descriptor = descriptor - np.mean(descriptor)
        std = np.std(descriptor)
        if std > 0:
            descriptor = descriptor / std
            
        return descriptor
