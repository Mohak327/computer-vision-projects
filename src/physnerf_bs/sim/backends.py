from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from .cameras import project_points


@dataclass(slots=True)
class RenderOutput:
    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray
    center_of_mass_height: float
    mass: float


class SimulationBackend(Protocol):
    dof: int
    body_mass: float

    def render(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        control: np.ndarray,
        torque: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_hw: tuple[int, int],
    ) -> RenderOutput:
        ...


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def synthetic_joint_positions(qpos: np.ndarray) -> np.ndarray:
    q = np.pad(qpos.astype(np.float32), (0, max(0, 17 - len(qpos))))[:17]
    pelvis = np.array([0.0, 0.95 + 0.15 * np.cos(q[1]), 0.0], dtype=np.float32)
    spine = pelvis + np.array([0.0, 0.35, 0.0], dtype=np.float32)
    head = spine + np.array([0.0, 0.22, 0.0], dtype=np.float32)

    l_shoulder = spine + np.array([-0.20, 0.10, 0.0], dtype=np.float32)
    r_shoulder = spine + np.array([0.20, 0.10, 0.0], dtype=np.float32)
    l_elbow = l_shoulder + (_rot_z(q[7]) @ _rot_x(q[9]) @ np.array([-0.30, -0.10, 0.0], dtype=np.float32))
    r_elbow = r_shoulder + (_rot_z(-q[8]) @ _rot_x(q[10]) @ np.array([0.30, -0.10, 0.0], dtype=np.float32))
    l_hand = l_elbow + (_rot_z(q[11]) @ np.array([-0.24, -0.08, 0.0], dtype=np.float32))
    r_hand = r_elbow + (_rot_z(-q[12]) @ np.array([0.24, -0.08, 0.0], dtype=np.float32))

    l_hip = pelvis + np.array([-0.12, -0.05, 0.0], dtype=np.float32)
    r_hip = pelvis + np.array([0.12, -0.05, 0.0], dtype=np.float32)
    l_knee = l_hip + (_rot_x(q[1]) @ np.array([0.0, -0.44, 0.06], dtype=np.float32))
    r_knee = r_hip + (_rot_x(q[2]) @ np.array([0.0, -0.44, -0.06], dtype=np.float32))
    l_ankle = l_knee + (_rot_x(q[3]) @ np.array([0.0, -0.42, 0.02], dtype=np.float32))
    r_ankle = r_knee + (_rot_x(q[4]) @ np.array([0.0, -0.42, -0.02], dtype=np.float32))
    l_foot = l_ankle + np.array([-0.05, -0.02, 0.14], dtype=np.float32)
    r_foot = r_ankle + np.array([0.05, -0.02, 0.14], dtype=np.float32)

    return np.stack(
        [
            pelvis,
            spine,
            head,
            l_shoulder,
            l_elbow,
            l_hand,
            r_shoulder,
            r_elbow,
            r_hand,
            l_hip,
            l_knee,
            l_ankle,
            l_foot,
            r_hip,
            r_knee,
            r_ankle,
            r_foot,
        ],
        axis=0,
    )


class SyntheticHumanoidBackend:
    def __init__(self, dof: int = 17, body_mass: float = 75.0) -> None:
        self.dof = dof
        self.body_mass = body_mass
        self.edges = [
            (0, 1), (1, 2),
            (1, 3), (3, 4), (4, 5),
            (1, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
        ]

    def render(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        control: np.ndarray,
        torque: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_hw: tuple[int, int],
    ) -> RenderOutput:
        height, width = image_hw
        joints = synthetic_joint_positions(qpos)
        pixels, depth_values = project_points(joints, intrinsics, extrinsics)

        rgb = Image.new("RGB", (width, height), color=(245, 247, 252))
        seg = Image.new("L", (width, height), color=0)
        rgb_draw = ImageDraw.Draw(rgb)
        seg_draw = ImageDraw.Draw(seg)

        # Draw furthest limbs first so closer segments overwrite them.
        order = np.argsort(depth_values)[::-1]
        sorted_edges = [edge for idx in order for edge in self.edges if edge[1] == idx or edge[0] == idx]
        seen: set[tuple[int, int]] = set()
        for start, end in sorted_edges:
            edge = tuple(sorted((start, end)))
            if edge in seen:
                continue
            seen.add(edge)
            p0 = tuple(pixels[start].tolist())
            p1 = tuple(pixels[end].tolist())
            rgb_draw.line([p0, p1], fill=(74, 111, 165), width=5)
            seg_draw.line([p0, p1], fill=255, width=7)

        for idx, point in enumerate(pixels):
            radius = 5 if idx in {0, 1, 2} else 4
            xy = [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius]
            color = (222, 101, 58) if idx in {2, 5, 8, 12, 16} else (58, 95, 140)
            rgb_draw.ellipse(xy, fill=color)
            seg_draw.ellipse(xy, fill=255)

        depth = np.zeros((height, width), dtype=np.float32)
        mask = np.array(seg, dtype=np.float32) > 0
        if mask.any():
            avg_depth = float(depth_values.mean())
            depth[mask] = avg_depth

        return RenderOutput(
            rgb=np.array(rgb, dtype=np.float32) / 255.0,
            depth=depth,
            segmentation=mask,
            center_of_mass_height=float(joints[:, 1].mean()),
            mass=self.body_mass,
        )


class DmControlHumanoidBackend:
    def __init__(self, dof: int = 17, body_mass: float = 75.0) -> None:
        try:
            from dm_control import suite  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only when installed
            raise RuntimeError("dm_control is not installed.") from exc

        self.suite = suite
        self.dof = dof
        self.body_mass = body_mass
        self.env = suite.load(domain_name="humanoid", task_name="stand")
        self.physics = self.env.physics

    def render(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        control: np.ndarray,
        torque: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_hw: tuple[int, int],
    ) -> RenderOutput:  # pragma: no cover - depends on optional package
        height, width = image_hw
        qpos = np.asarray(qpos, dtype=np.float32)
        qvel = np.asarray(qvel, dtype=np.float32)
        with self.physics.reset_context():
            n_qpos = min(len(qpos), len(self.physics.data.qpos))
            n_qvel = min(len(qvel), len(self.physics.data.qvel))
            self.physics.data.qpos[:n_qpos] = qpos[:n_qpos]
            self.physics.data.qvel[:n_qvel] = qvel[:n_qvel]
        rgb = self.physics.render(height=height, width=width, camera_id=0)
        try:
            depth = self.physics.render(height=height, width=width, camera_id=0, depth=True)
        except TypeError:
            depth = np.zeros((height, width), dtype=np.float32)
        try:
            seg = self.physics.render(height=height, width=width, camera_id=0, segmentation=True)
            seg_mask = np.asarray(seg)[..., 0] > 0
        except TypeError:
            seg_mask = np.zeros((height, width), dtype=bool)
        com_height = float(np.mean(self.physics.named.data.xipos[:, 2])) if hasattr(self.physics.named.data, "xipos") else 1.0
        return RenderOutput(
            rgb=np.asarray(rgb, dtype=np.float32) / 255.0,
            depth=np.asarray(depth, dtype=np.float32),
            segmentation=seg_mask,
            center_of_mass_height=com_height,
            mass=self.body_mass,
        )
