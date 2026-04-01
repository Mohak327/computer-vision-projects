from __future__ import annotations

import math

import numpy as np


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def look_at(camera_position: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32) if up is None else up.astype(np.float32)
    forward = normalize(target - camera_position)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))

    world_from_camera = np.eye(4, dtype=np.float32)
    world_from_camera[:3, 0] = right
    world_from_camera[:3, 1] = true_up
    world_from_camera[:3, 2] = -forward
    world_from_camera[:3, 3] = camera_position
    return world_from_camera


def create_camera_rig(
    n_cameras: int,
    radius: float,
    elevation_deg: float,
    target: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> list[dict[str, np.ndarray | str]]:
    target_np = np.array(target, dtype=np.float32)
    elevation = math.radians(elevation_deg)
    rig: list[dict[str, np.ndarray | str]] = []
    for idx in range(n_cameras):
        angle = 2.0 * math.pi * idx / max(n_cameras, 1)
        position = np.array(
            [
                radius * math.cos(angle),
                target_np[1] + radius * math.sin(elevation),
                radius * math.sin(angle),
            ],
            dtype=np.float32,
        )
        rig.append(
            {
                "camera_id": f"cam_{idx:02d}",
                "extrinsics": look_at(position, target_np),
            }
        )
    return rig


def make_intrinsics(width: int, height: int, fov_deg: float = 50.0) -> np.ndarray:
    focal = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def project_points(points_world: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_from_world = np.linalg.inv(extrinsics)
    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=-1)
    points_cam = (camera_from_world @ points_h.T).T[:, :3]
    z = np.clip(-points_cam[:, 2], 1e-4, None)
    x = points_cam[:, 0] / z
    y = points_cam[:, 1] / z
    pixels = (intrinsics @ np.stack([x, y, np.ones_like(x)], axis=0)).T[:, :2]
    return pixels.astype(np.float32), z.astype(np.float32)
