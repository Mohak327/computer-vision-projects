import numpy as np

from triangulation import (
    normalize_points_with_K,
    recover_pose,
)


def compute_E(
    img1_pts: np.ndarray,
    img2_pts: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Compute essential matrix via normalized 8-point algorithm."""
    assert len(img1_pts) >= 8 and len(img1_pts) == len(img2_pts)

    pts1 = normalize_points_with_K(img1_pts[:, [1, 0]], K)
    pts2 = normalize_points_with_K(img2_pts[:, [1, 0]], K)

    A = np.array(
        [
            [x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1.0]
            for (x1, y1), (x2, y2) in zip(pts1, pts2)
        ],
        dtype=float,
    )

    _, _, VT = np.linalg.svd(A, full_matrices=True)
    E = VT[-1].reshape(3, 3)

    U, _, VT = np.linalg.svd(E)
    E = U @ np.diag([1.0, 1.0, 0.0]) @ VT
    return E


def sampson_distance(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    """Squared Sampson distance for all correspondences."""
    pts1_xy = pts1[:, [1, 0]]
    pts2_xy = pts2[:, [1, 0]]

    pts1_h = (K_inv @ np.column_stack([pts1_xy, np.ones(len(pts1_xy))]).T).T
    pts2_h = (K_inv @ np.column_stack([pts2_xy, np.ones(len(pts2_xy))]).T).T

    Ep1 = pts1_h @ E.T
    ETp2 = pts2_h @ E

    numerator = np.sum(pts2_h * Ep1, axis=1)
    denominator = Ep1[:, 0] ** 2 + Ep1[:, 1] ** 2 + ETp2[:, 0] ** 2 + ETp2[:, 1] ** 2
    return numerator**2 / (denominator + 1e-8)


def RANSAC(
    correspondence_pairs: np.ndarray,
    K: np.ndarray,
    s: int,
    epsilon: float,
    num_iters: int,
    output_dir: str | None = None,
):
    """RANSAC for essential matrix.

    Returns:
        (R, t, inlier_mask, E_refined, points_3d, best_inlier_history, inlier_count_history)
    """
    if len(correspondence_pairs) < 8:
        return None, None, None, None, None, None, None

    s = max(8, min(len(correspondence_pairs), s))
    img1_pts = correspondence_pairs[:, 0]
    img2_pts = correspondence_pairs[:, 1]
    K_inv = np.linalg.inv(K)

    best_num_inliers = 0
    best_E = None
    best_inliers_mask = None
    best_inlier_history = []
    inlier_count_history = []

    for _ in range(num_iters):
        idx = np.random.choice(len(img1_pts), s, replace=False)

        try:
            E = compute_E(img1_pts[idx], img2_pts[idx], K)
            R_try, t_try, _ = recover_pose(E, img1_pts[idx], img2_pts[idx], K)
            if R_try is None:
                inlier_count_history.append(0)
                best_inlier_history.append(best_num_inliers)
                continue

            d2 = sampson_distance(E, img1_pts, img2_pts, K_inv)
            inliers_mask = d2 < epsilon
            num_inliers = int(np.sum(inliers_mask))
            inlier_count_history.append(num_inliers)

            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_E = E
                best_inliers_mask = inliers_mask
        except (np.linalg.LinAlgError, AssertionError, ValueError):
            inlier_count_history.append(0)
            best_inlier_history.append(best_num_inliers)
            continue

        best_inlier_history.append(best_num_inliers)

    if best_E is None or best_inliers_mask is None or best_num_inliers < 8:
        return (
            None,
            None,
            None,
            None,
            None,
            np.array(best_inlier_history, dtype=int),
            np.array(inlier_count_history, dtype=int),
        )

    # post-loop refinement
    E_refined = compute_E(img1_pts[best_inliers_mask], img2_pts[best_inliers_mask], K)
    R, t, points_3d = recover_pose(E_refined, img1_pts[best_inliers_mask], img2_pts[best_inliers_mask], K)

    if R is None:
        return (
            None,
            None,
            None,
            None,
            None,
            np.array(best_inlier_history, dtype=int),
            np.array(inlier_count_history, dtype=int),
        )

    return (
        R,
        t,
        best_inliers_mask,
        E_refined,
        points_3d,
        np.array(best_inlier_history, dtype=int),
        np.array(inlier_count_history, dtype=int),
    )
