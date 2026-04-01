from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PhysicsTargets:
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    center_of_mass_height: float
    mass: float
    metadata: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FrameRecord:
    frame_id: str
    sequence_id: str
    motion_id: str
    split: str
    timestamp: float
    camera_id: str
    rgb_path: str
    depth_path: str
    segmentation_path: str
    qpos: list[float]
    qvel: list[float]
    control: list[float]
    torque: list[float]
    camera_intrinsics: list[list[float]]
    camera_extrinsics: list[list[float]]
    physics_targets: PhysicsTargets

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["physics_targets"] = self.physics_targets.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrameRecord":
        payload = dict(payload)
        payload["physics_targets"] = PhysicsTargets(**payload["physics_targets"])
        return cls(**payload)


@dataclass(slots=True)
class SequenceRecord:
    sequence_id: str
    motion_id: str
    split: str
    frame_ids: list[str]
    motor_stream: list[list[float]]
    timestamps: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SequenceRecord":
        return cls(**payload)


@dataclass(slots=True)
class ClarisNetAdapterIO:
    sequence_id: str
    timestamps: list[float]
    joint_positions: list[list[float]]
    joint_velocities: list[list[float]]
    control: list[list[float]]
    torque: list[list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
