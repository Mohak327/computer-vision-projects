from __future__ import annotations

import math

import numpy as np


def _phase_series(num_frames: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * math.pi, num_frames, endpoint=False, dtype=np.float32)


def generate_motion_trajectory(
    motion: str,
    num_frames: int,
    dof: int,
    noise_std: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    phase = _phase_series(num_frames)
    qpos = np.zeros((num_frames, dof), dtype=np.float32)
    qvel = np.zeros((num_frames, dof), dtype=np.float32)
    control = np.zeros((num_frames, dof), dtype=np.float32)
    torque = np.zeros((num_frames, dof), dtype=np.float32)

    if motion == "walk":
        qpos[:, 0] = 0.25 * np.sin(phase)
        qpos[:, 1] = 0.5 * np.sin(phase)
        qpos[:, 2] = -0.5 * np.sin(phase)
        qpos[:, 3] = 0.3 * np.sin(phase + math.pi)
        qpos[:, 4] = -0.3 * np.sin(phase + math.pi)
        qpos[:, 5] = 0.15 * np.cos(phase)
        qpos[:, 6] = -0.15 * np.cos(phase)
    elif motion == "arm_raise":
        lift = 0.5 * (1.0 - np.cos(phase))
        qpos[:, 7] = 1.2 * lift
        qpos[:, 8] = 1.1 * lift
        qpos[:, 9] = 0.2 * np.sin(phase)
        qpos[:, 10] = -0.2 * np.sin(phase)
        qpos[:, 11] = 0.08 * np.sin(2.0 * phase)
    elif motion == "crouch":
        crouch = 0.5 * (1.0 - np.cos(phase))
        qpos[:, 1] = -0.6 * crouch
        qpos[:, 2] = -0.6 * crouch
        qpos[:, 3] = 0.8 * crouch
        qpos[:, 4] = 0.8 * crouch
        qpos[:, 5] = -0.25 * crouch
        qpos[:, 6] = -0.25 * crouch
    else:
        qpos[:, : min(dof, 6)] = 0.2 * np.sin(phase[:, None] + np.arange(min(dof, 6), dtype=np.float32))

    if noise_std > 0.0:
        qpos += rng.normal(scale=noise_std, size=qpos.shape).astype(np.float32)

    qvel[1:] = qpos[1:] - qpos[:-1]
    qvel[0] = qvel[1]
    control[:] = qvel
    torque[:] = qvel * 0.5 + qpos * 0.1
    return qpos, qvel, control, torque
