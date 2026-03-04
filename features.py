import cv2
import numpy as np


def get_sift_features(
    img_gray: np.ndarray,
    edge_discard: int = 20,
    max_features: int = 1200,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10,
    n_octave_layers: int = 3,
    sigma: float = 1.6,
):
    """Extract SIFT keypoints and descriptors.

    Returns:
        coords: (2, N) in (row, col)
        descriptors: (N, 128)
        responses: (N,)
    """
    img_uint8 = (img_gray * 255).astype(np.uint8) if img_gray.max() <= 1.0 else img_gray.astype(np.uint8)

    sift = cv2.SIFT_create(
        nfeatures=max_features,
        nOctaveLayers=n_octave_layers,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )
    keypoints, descriptors = sift.detectAndCompute(img_uint8, None)

    if descriptors is None or len(keypoints) == 0:
        return np.array([[], []]), np.array([]), np.array([])

    coords_xy = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])

    if edge_discard > 0:
        h, w = img_gray.shape[:2]
        mask = (
            (coords_xy[:, 1] > edge_discard)
            & (coords_xy[:, 1] < h - edge_discard)
            & (coords_xy[:, 0] > edge_discard)
            & (coords_xy[:, 0] < w - edge_discard)
        )
        coords_xy, descriptors, responses = coords_xy[mask], descriptors[mask], responses[mask]

    coords = coords_xy[:, [1, 0]].T
    return coords, descriptors, responses


def match_features(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    ratio_threshold: float = 0.8,
):
    """Match SIFT features using BFMatcher + NNDR ratio test.

    Returns:
        correspondence_pairs: (M, 2, 2) in (row, col)
        knn_matches: raw knn results
        nndr_proportions: list of d1/d2 ratios
    """
    if descriptors1.size == 0 or descriptors2.size == 0:
        return np.zeros((0, 2, 2)), [], []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(
        descriptors1.astype(np.float32),
        descriptors2.astype(np.float32),
        k=2,
    )

    good = [m for m, n in knn_matches if n.distance > 0 and m.distance < ratio_threshold * n.distance]

    seen_img2 = set()
    correspondences = []
    for m in good:
        pt2 = (keypoints2[m.trainIdx, 0], keypoints2[m.trainIdx, 1])
        if pt2 in seen_img2:
            continue
        correspondences.append([keypoints1[m.queryIdx], keypoints2[m.trainIdx]])
        seen_img2.add(pt2)

    correspondence_pairs = np.array(correspondences) if correspondences else np.zeros((0, 2, 2))
    nndr_proportions = [m.distance / n.distance for m, n in knn_matches if n.distance > 0]
    return correspondence_pairs, knn_matches, nndr_proportions
