from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import DataConfig
from ..schemas import FrameRecord, SequenceRecord
from ..sim.simulator import SimulatedFrameBundle
from ..utils.io import ensure_dir, save_depth_array, save_json, save_jsonl, save_mask_image, save_rgb_image


@dataclass(slots=True)
class DatasetWriteResult:
    root: Path
    frames_manifest: Path
    sequences_manifest: Path
    num_frames: int
    num_sequences: int


def _camera_split(camera_id: str, n_cameras: int, train_split: float, val_split: float) -> str:
    cam_index = int(camera_id.split("_")[-1])
    train_cut = max(1, int(n_cameras * train_split))
    val_cut = max(train_cut + 1, int(n_cameras * (train_split + val_split)))
    if cam_index < train_cut:
        return "train"
    if cam_index < min(val_cut, n_cameras):
        return "val"
    return "test"


def write_simulation_dataset(
    bundles: list[SimulatedFrameBundle],
    data_config: DataConfig,
    n_cameras: int,
) -> DatasetWriteResult:
    root = ensure_dir(data_config.root)
    rgb_dir = ensure_dir(root / "rgb")
    depth_dir = ensure_dir(root / "depth")
    seg_dir = ensure_dir(root / "segmentation")

    frame_records: list[FrameRecord] = []
    sequence_map: dict[str, SequenceRecord] = {}

    for bundle in bundles:
        split = _camera_split(bundle.camera_id, n_cameras, data_config.train_split, data_config.val_split)
        sequence_id = f"{bundle.motion_id}_{bundle.camera_id}"
        frame_id = f"{sequence_id}_{bundle.frame_index:04d}"
        rgb_path = rgb_dir / f"{frame_id}.png"
        depth_path = depth_dir / f"{frame_id}.npy"
        seg_path = seg_dir / f"{frame_id}.png"

        save_rgb_image(bundle.render.rgb, rgb_path)
        save_depth_array(bundle.render.depth, depth_path)
        save_mask_image(bundle.render.segmentation, seg_path)

        record = FrameRecord(
            frame_id=frame_id,
            sequence_id=sequence_id,
            motion_id=bundle.motion_id,
            split=split,
            timestamp=bundle.timestamp,
            camera_id=bundle.camera_id,
            rgb_path=str(rgb_path.relative_to(root)),
            depth_path=str(depth_path.relative_to(root)),
            segmentation_path=str(seg_path.relative_to(root)),
            qpos=bundle.qpos.tolist(),
            qvel=bundle.qvel.tolist(),
            control=bundle.control.tolist(),
            torque=bundle.torque.tolist(),
            camera_intrinsics=bundle.intrinsics.tolist(),
            camera_extrinsics=bundle.extrinsics.tolist(),
            physics_targets=bundle.physics_targets,
        )
        frame_records.append(record)

        if sequence_id not in sequence_map:
            sequence_map[sequence_id] = SequenceRecord(
                sequence_id=sequence_id,
                motion_id=bundle.motion_id,
                split=split,
                frame_ids=[],
                motor_stream=[],
                timestamps=[],
            )
        sequence_map[sequence_id].frame_ids.append(frame_id)
        sequence_map[sequence_id].motor_stream.append(bundle.torque.tolist())
        sequence_map[sequence_id].timestamps.append(bundle.timestamp)

    frames_manifest = root / "frames.jsonl"
    sequences_manifest = root / "sequences.json"
    save_jsonl((record.to_dict() for record in frame_records), frames_manifest)
    save_json([sequence.to_dict() for sequence in sequence_map.values()], sequences_manifest)
    save_json(
        {
            "num_frames": len(frame_records),
            "num_sequences": len(sequence_map),
            "splits": {
                split: sum(1 for record in frame_records if record.split == split)
                for split in ["train", "val", "test"]
            },
        },
        root / "dataset_summary.json",
    )

    return DatasetWriteResult(
        root=root,
        frames_manifest=frames_manifest,
        sequences_manifest=sequences_manifest,
        num_frames=len(frame_records),
        num_sequences=len(sequence_map),
    )
