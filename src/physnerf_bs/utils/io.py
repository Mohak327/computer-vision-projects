from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_rgb_image(image: np.ndarray, path: str | Path) -> None:
    array = np.clip(image, 0.0, 1.0)
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


def save_mask_image(mask: np.ndarray, path: str | Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def save_depth_array(depth: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, depth.astype(np.float32))


def load_rgb_image(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask_image(path: str | Path) -> np.ndarray:
    return (np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0) > 0.5


def load_depth_array(path: str | Path) -> np.ndarray:
    return np.load(path).astype(np.float32)
