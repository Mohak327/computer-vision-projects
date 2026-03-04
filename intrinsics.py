import numpy as np


def compute_K(
    img_width_px: int,
    img_height_px: int,
    optical_focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
) -> np.ndarray:
    """Compute camera intrinsics K from physical camera parameters."""
    if img_width_px <= 0 or img_height_px <= 0:
        raise ValueError("Image dimensions must be positive")
    if optical_focal_length_mm <= 0 or sensor_width_mm <= 0 or sensor_height_mm <= 0:
        raise ValueError("Focal length and sensor dimensions must be positive")

    fpx_x = optical_focal_length_mm * img_width_px / sensor_width_mm
    fpx_y = optical_focal_length_mm * img_height_px / sensor_height_mm
    fpx = 0.5 * (fpx_x + fpx_y)

    cx = img_width_px / 2.0
    cy = img_height_px / 2.0

    return np.array(
        [
            [fpx, 0.0, cx],
            [0.0, fpx, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
