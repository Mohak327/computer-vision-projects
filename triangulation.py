import numpy as np


def normalize_points_with_K(pts_xy: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts_homo = np.column_stack([pts_xy, np.ones(len(pts_xy))])
    return (np.linalg.inv(K) @ pts_homo.T).T[:, :2]


def decompose_E(E: np.ndarray):
    U, _, VT = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(VT) < 0:
        VT = -VT
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return [
        (U @ W @ VT, U[:, 2]),
        (U @ W @ VT, -U[:, 2]),
        (U @ W.T @ VT, U[:, 2]),
        (U @ W.T @ VT, -U[:, 2]),
    ]


def triangulate_point(pt1, pt2, P1, P2):
    A = np.array(
        [
            pt1[0] * P1[2] - P1[0],
            pt1[1] * P1[2] - P1[1],
            pt2[0] * P2[2] - P2[0],
            pt2[1] * P2[2] - P2[1],
        ]
    )
    _, _, VT = np.linalg.svd(A)
    X = VT[-1]
    return X / X[3]


def check_cheirality(R, t, pts1, pts2, K, check_reprojection=False, max_reproj_error=5.0):
    pts1_xy = pts1[:, [1, 0]]
    pts2_xy = pts2[:, [1, 0]]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    count = 0
    points_3d = []
    for pt1, pt2 in zip(pts1_xy, pts2_xy):
        X = triangulate_point(pt1, pt2, P1, P2)
        points_3d.append(X[:3])

        depth1 = X[2]
        depth2 = (R @ X[:3] + t)[2]
        is_valid = depth1 > 0 and depth2 > 0

        if is_valid and check_reprojection:
            X_h = np.append(X[:3], 1)
            p1 = P1 @ X_h
            p2 = P2 @ X_h
            p1 = p1[:2] / p1[2]
            p2 = p2[:2] / p2[2]
            avg_error = (np.linalg.norm(pt1 - p1) + np.linalg.norm(pt2 - p2)) / 2.0
            is_valid = avg_error < max_reproj_error

        if is_valid:
            count += 1

    return count, np.array(points_3d)


def triangulate_with_reprojection_filter(
    R,
    t,
    pts1,
    pts2,
    K,
    max_reprojection_error=5.0,
    min_depth=0.01,
    max_depth=10000.0,
    verbose=False,
):
    pts1_xy = pts1[:, [1, 0]]
    pts2_xy = pts2[:, [1, 0]]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    points_3d, valid_mask, reproj_errors = [], [], []
    for pt1, pt2 in zip(pts1_xy, pts2_xy):
        X = triangulate_point(pt1, pt2, P1, P2)
        point_3d = X[:3]
        depth1 = point_3d[2]
        depth2 = (R @ point_3d + t)[2]

        X_h = np.append(point_3d, 1)
        p1 = P1 @ X_h
        p2 = P2 @ X_h
        p1 = p1[:2] / p1[2]
        p2 = p2[:2] / p2[2]
        avg_error = (np.linalg.norm(pt1 - p1) + np.linalg.norm(pt2 - p2)) / 2.0

        is_valid = (
            depth1 > min_depth
            and depth2 > min_depth
            and depth1 < max_depth
            and depth2 < max_depth
            and avg_error < max_reprojection_error
        )

        points_3d.append(point_3d)
        valid_mask.append(is_valid)
        reproj_errors.append(avg_error)

    points_3d = np.array(points_3d)
    valid_mask = np.array(valid_mask)
    reproj_errors = np.array(reproj_errors)

    if verbose:
        print(f"Triangulated: {len(points_3d)}, valid: {np.sum(valid_mask)}")

    return points_3d[valid_mask], valid_mask, reproj_errors


def recover_pose(E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray):
    solutions = decompose_E(E)

    best_solution = None
    best_depth_count = -1
    best_reproj_count = -1
    best_3d = None

    for R_cand, t_cand in solutions:
        depth_count, pts3d = check_cheirality(
            R_cand, t_cand, pts1, pts2, K, check_reprojection=False
        )
        reproj_count, _ = check_cheirality(
            R_cand, t_cand, pts1, pts2, K, check_reprojection=True, max_reproj_error=30.0
        )

        is_better = (
            best_solution is None
            or depth_count > best_depth_count
            or (depth_count == best_depth_count and reproj_count > best_reproj_count)
        )
        if is_better:
            best_solution = (R_cand, t_cand)
            best_depth_count = depth_count
            best_reproj_count = reproj_count
            best_3d = pts3d

    if best_solution is None:
        return None, None, None
    return best_solution[0], best_solution[1], best_3d
