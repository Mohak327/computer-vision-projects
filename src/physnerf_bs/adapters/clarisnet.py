from __future__ import annotations

from ..schemas import ClarisNetAdapterIO, SequenceRecord


def normalize_clarisnet_sequence(payload: ClarisNetAdapterIO, split: str = "external") -> SequenceRecord:
    if not payload.timestamps:
        raise ValueError("CLARISNet adapter payload must include timestamps.")
    if not (len(payload.timestamps) == len(payload.joint_positions) == len(payload.torque)):
        raise ValueError("CLARISNet adapter payload lists must be aligned in length.")
    return SequenceRecord(
        sequence_id=payload.sequence_id,
        motion_id=payload.metadata.get("motion_id", "clarisnet_external"),
        split=split,
        frame_ids=[f"{payload.sequence_id}_{idx:04d}" for idx in range(len(payload.timestamps))],
        motor_stream=[list(step) for step in payload.torque],
        timestamps=list(payload.timestamps),
    )
