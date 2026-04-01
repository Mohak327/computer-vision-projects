from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SimulationConfig
from ..schemas import PhysicsTargets
from .backends import DmControlHumanoidBackend, RenderOutput, SyntheticHumanoidBackend
from .cameras import create_camera_rig, make_intrinsics
from .motions import generate_motion_trajectory


@dataclass(slots=True)
class SimulatedFrameBundle:
    motion_id: str
    frame_index: int
    timestamp: float
    qpos: np.ndarray
    qvel: np.ndarray
    control: np.ndarray
    torque: np.ndarray
    camera_id: str
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    render: RenderOutput
    physics_targets: PhysicsTargets


class HumanoidSimulationService:
    def __init__(self, config: SimulationConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = seed
        self.intrinsics = make_intrinsics(config.image_width, config.image_height)
        self.camera_rig = create_camera_rig(
            config.n_cameras,
            radius=config.camera_radius,
            elevation_deg=config.camera_elevation_deg,
        )
        self.backend = self._build_backend(config.backend, config.fallback_backend)

    def _build_backend(self, preferred: str, fallback: str):
        if preferred == "synthetic":
            return SyntheticHumanoidBackend(dof=self.config.dof, body_mass=self.config.body_mass)
        if preferred == "dm_control":
            try:
                return DmControlHumanoidBackend(dof=self.config.dof, body_mass=self.config.body_mass)
            except Exception:
                if fallback == "synthetic":
                    return SyntheticHumanoidBackend(dof=self.config.dof, body_mass=self.config.body_mass)
                raise
        raise ValueError(f"Unsupported backend: {preferred}")

    def _physics_targets(self, qpos: np.ndarray, qvel: np.ndarray, render: RenderOutput) -> PhysicsTargets:
        kinetic = float(0.5 * np.sum(qvel**2) * render.mass / max(len(qvel), 1))
        potential = float(render.mass * 9.81 * max(render.center_of_mass_height, 0.0))
        return PhysicsTargets(
            kinetic_energy=kinetic,
            potential_energy=potential,
            total_energy=kinetic + potential,
            center_of_mass_height=render.center_of_mass_height,
            mass=render.mass,
            metadata={"qpos_norm": float(np.linalg.norm(qpos)), "qvel_norm": float(np.linalg.norm(qvel))},
        )

    def generate_motion(self, motion: str) -> list[SimulatedFrameBundle]:
        qpos, qvel, control, torque = generate_motion_trajectory(
            motion=motion,
            num_frames=self.config.frames_per_motion,
            dof=self.config.dof,
            noise_std=self.config.noise_std,
            seed=self.seed,
        )

        bundles: list[SimulatedFrameBundle] = []
        for frame_idx in range(self.config.frames_per_motion):
            for camera in self.camera_rig:
                render = self.backend.render(
                    qpos=qpos[frame_idx],
                    qvel=qvel[frame_idx],
                    control=control[frame_idx],
                    torque=torque[frame_idx],
                    intrinsics=self.intrinsics,
                    extrinsics=np.asarray(camera["extrinsics"], dtype=np.float32),
                    image_hw=(self.config.image_height, self.config.image_width),
                )
                bundles.append(
                    SimulatedFrameBundle(
                        motion_id=motion,
                        frame_index=frame_idx,
                        timestamp=float(frame_idx),
                        qpos=qpos[frame_idx],
                        qvel=qvel[frame_idx],
                        control=control[frame_idx],
                        torque=torque[frame_idx],
                        camera_id=str(camera["camera_id"]),
                        intrinsics=self.intrinsics.copy(),
                        extrinsics=np.asarray(camera["extrinsics"], dtype=np.float32),
                        render=render,
                        physics_targets=self._physics_targets(qpos[frame_idx], qvel[frame_idx], render),
                    )
                )
        return bundles

    def generate_dataset(self) -> list[SimulatedFrameBundle]:
        all_bundles: list[SimulatedFrameBundle] = []
        for motion in self.config.motions:
            all_bundles.extend(self.generate_motion(motion))
        return all_bundles
