"""
Visualize HW3 pose-estimation output in Viser.

Usage:
    python visualize_viser.py <path_to_scene_data.npz> [port]
"""

from __future__ import annotations

import os
import re
import sys
import time
import numpy as np
import viser
from PIL import Image


def _make_server(start_port: int = 8081, max_tries: int = 50):
    port = start_port
    for _ in range(max_tries):
        try:
            server = viser.ViserServer(port=port)
            return server, port
        except OSError:
            port += 1
    raise RuntimeError(f"Could not bind a Viser server port after {max_tries} attempts")


def visualize_scene(npz_path: str, port: int = 8081, block: bool = True):
    data = np.load(npz_path, allow_pickle=True)

    points_3d = data["points_3d"]
    point_colors = data.get("point_colors", None)
    camera_poses = data["camera_poses"]  # (2, 4, 4), camera-to-world
    K = data["K"]

    img1 = data.get("img1", None)
    img2 = data.get("img2", None)

    num_inliers = int(data.get("num_inliers", 0))
    baseline = float(data.get("baseline", 0.0))

    if point_colors is None or len(point_colors) != len(points_3d):
        point_colors = np.tile(np.array([[0.2, 0.6, 1.0]]), (len(points_3d), 1))
    elif point_colors.max() > 1.0:
        point_colors = point_colors / 255.0

    server, actual_port = _make_server(port)
    print(f"Viser running at: http://localhost:{actual_port}")
    print("Open the URL above in your browser.")

    # Initial camera look direction based on camera 1 reference
    cam0_forward = camera_poses[0][:3, 2]
    cam0_up = -camera_poses[0][:3, 1]

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = -cam0_forward * 2.5 + cam0_up * 0.6
        client.camera.look_at = np.zeros(3)
        client.camera.up_direction = cam0_up

    pc_handle = server.scene.add_point_cloud(
        name="/point_cloud",
        points=points_3d,
        colors=point_colors,
        point_size=0.01,
    )

    camera_names = ["Camera 1 (Reference)", "Camera 2"]
    camera_colors = [(0, 255, 0), (255, 0, 0)]
    camera_images = [img1, img2]

    frustums = []
    labels = []
    images = []

    for i, (T, name, color, img) in enumerate(zip(camera_poses, camera_names, camera_colors, camera_images)):
        R_c2w = T[:3, :3]
        pos = T[:3, 3]

        fy = K[1, 1]
        if img is not None:
            img_h = img.shape[0]
            fov = 2 * np.arctan(img_h / (2 * fy))
            aspect = img.shape[1] / img.shape[0]
        else:
            cy = K[1, 2]
            fov = 2 * np.arctan(cy / fy)
            aspect = 1.0

        fr = server.scene.add_camera_frustum(
            name=f"/camera_{i}",
            fov=fov,
            aspect=aspect,
            scale=0.3,
            wxyz=viser.transforms.SO3.from_matrix(R_c2w).wxyz,
            position=pos,
            color=color,
        )
        frustums.append(fr)

        labels.append(
            server.scene.add_label(
                name=f"/camera_{i}_label",
                text=name,
                position=pos + np.array([0, 0, 0.15]),
            )
        )

        if img is not None:
            img_uint8 = (img * 255).astype(np.uint8) if img.dtype != np.uint8 and img.max() <= 1.0 else img.astype(np.uint8)
            if img_uint8.ndim == 2:
                img_uint8 = np.stack([img_uint8] * 3, axis=-1)

            image_distance = 0.5
            image_height = 2 * image_distance * np.tan(fov / 2)
            image_width = image_height * aspect
            forward = R_c2w[:, 2]
            image_center = pos + forward * image_distance

            images.append(
                server.scene.add_image(
                    name=f"/camera_{i}_image",
                    image=img_uint8,
                    render_width=image_width,
                    render_height=image_height,
                    wxyz=viser.transforms.SO3.from_matrix(R_c2w).wxyz,
                    position=image_center,
                )
            )
        else:
            images.append(None)

    server.scene.add_frame(
        name="/world",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=np.array([0.0, 0.0, 0.0]),
        axes_length=0.3,
        axes_radius=0.01,
    )

    baseline_handle = server.scene.add_spline_catmull_rom(
        name="/baseline",
        positions=np.array([camera_poses[0][:3, 3], camera_poses[1][:3, 3]]),
        color=(255, 255, 0),
        line_width=3.0,
    )

    with server.gui.add_folder("Scene Info"):
        server.gui.add_text("Points", initial_value=str(len(points_3d)), disabled=True)
        server.gui.add_text("Inliers", initial_value=str(num_inliers), disabled=True)
        server.gui.add_text("Baseline", initial_value=f"{baseline:.4f}", disabled=True)

    with server.gui.add_folder("Visibility"):
        pc_vis = server.gui.add_checkbox("Point Cloud", initial_value=True)
        cam_vis = server.gui.add_checkbox("Cameras", initial_value=True)
        img_vis = server.gui.add_checkbox("Camera Images", initial_value=True)
        bl_vis = server.gui.add_checkbox("Baseline", initial_value=True)

        @pc_vis.on_update
        def _(_evt):
            pc_handle.visible = pc_vis.value

        @cam_vis.on_update
        def _(_evt):
            for h in frustums:
                h.visible = cam_vis.value
            for h in labels:
                h.visible = cam_vis.value

        @img_vis.on_update
        def _(_evt):
            for h in images:
                if h is not None:
                    h.visible = img_vis.value

        @bl_vis.on_update
        def _(_evt):
            baseline_handle.visible = bl_vis.value

    if block:
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down Viser server...")

    return server


def _axis_word(delta: float, pos_word: str, neg_word: str, threshold: float):
    if delta > threshold:
        return pos_word
    if delta < -threshold:
        return neg_word
    return None


def _baseline_text(c_ref: np.ndarray, c_other: np.ndarray) -> str:
    d = c_other - c_ref
    scale = max(np.linalg.norm(d), 1e-8)
    thr = 0.12 * scale

    words = []
    for word in [
        _axis_word(d[0], "right", "left", thr),
        _axis_word(d[1], "higher", "lower", thr),
        _axis_word(d[2], "forward", "behind", thr),
    ]:
        if word is not None:
            words.append(word)

    if len(words) == 0:
        return "near camera 1"
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def _planarity_score(points: np.ndarray) -> float:
    if points is None or len(points) < 10:
        return 1.0
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    if len(s) < 3 or s[0] <= 1e-10:
        return 1.0
    return float(s[2] / s[0])


def _sorted_image_paths(folder: str):
    exts = (".png", ".jpg", ".jpeg")
    names = [name for name in os.listdir(folder) if name.lower().endswith(exts)]

    def _key(name: str):
        nums = re.findall(r"\d+", name)
        return (int(nums[-1]) if nums else 10**9, name.lower())

    names.sort(key=_key)
    return [os.path.join(folder, name) for name in names]


def _analyze_viser_image(path: str):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    h, w = img.shape[:2]

    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    contrast = float(np.std(gray) / 255.0)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    grad = np.sqrt(gx * gx + gy * gy)
    edge_threshold = np.percentile(grad, 85)
    edge_density = float(np.mean(grad > edge_threshold))

    corner_samples = np.vstack(
        [
            img[0:30, 0:30].reshape(-1, 3),
            img[0:30, -30:].reshape(-1, 3),
            img[-30:, 0:30].reshape(-1, 3),
            img[-30:, -30:].reshape(-1, 3),
        ]
    )
    bg = np.median(corner_samples, axis=0)
    dist = np.linalg.norm(img - bg, axis=2)
    fg = dist > 28.0
    occupancy = float(np.mean(fg))

    if np.any(fg):
        ys, xs = np.where(fg)
        cx = float(np.mean(xs) / max(w - 1, 1))
        cy = float(np.mean(ys) / max(h - 1, 1))
    else:
        cx, cy = 0.5, 0.5

    return {
        "contrast": contrast,
        "edge_density": edge_density,
        "occupancy": occupancy,
        "centroid_x": cx,
        "centroid_y": cy,
    }


def _detail_phrase(metrics) -> str:
    if metrics["edge_density"] > 0.20 and metrics["contrast"] > 0.16:
        return "dense detail"
    if metrics["edge_density"] > 0.14:
        return "moderate detail"
    return "sparser detail"


def _frame_phrase(metrics) -> str:
    if metrics["occupancy"] > 0.28:
        return "fills much of the frame"
    if metrics["occupancy"] > 0.16:
        return "is well centered in frame"
    return "sits compactly in frame"


def _compact_caption(i: int, metrics, rel_text: str, planarity: float, hint: str | None):
    detail = _detail_phrase(metrics)
    frame = _frame_phrase(metrics)

    if i == 0:
        base = f"... {detail}; scene {frame}."
    elif i == 1:
        base = f"... camera 2 appears {rel_text}; pose looks consistent."
    else:
        if planarity < 0.07:
            plane = "poster/planar surface looks flat"
        elif planarity < 0.14:
            plane = "planarity is partly visible"
        else:
            plane = "structure looks more volumetric"
        base = f"... {plane}; scene {frame}."

    if hint is not None and str(hint).strip() != "":
        return f"{base} I see {hint}."
    return base


def _normalize_landmark_hints(landmark_hints, expected_count: int) -> list[str | None]:
    """Normalize user hints to match screenshot count.

    Accepts None, a single string, or an iterable of values. Empty values are
    converted to None, and the list is padded/truncated to expected_count.
    """
    if expected_count <= 0:
        return []

    if landmark_hints is None:
        hints = []
    elif isinstance(landmark_hints, str):
        hints = [landmark_hints]
    else:
        try:
            hints = list(landmark_hints)
        except TypeError:
            hints = [str(landmark_hints)]

    normalized: list[str | None] = []
    for hint in hints:
        if hint is None:
            normalized.append(None)
            continue
        text = str(hint).strip()
        normalized.append(text if text else None)

    if len(normalized) < expected_count:
        normalized.extend([None] * (expected_count - len(normalized)))
    else:
        normalized = normalized[:expected_count]

    return normalized


def generate_viser_captions(npz_path: str, viser_image_paths: list[str], landmark_hints: list[str] | None = None):
    data = np.load(npz_path, allow_pickle=True)
    points = data["points_3d"] if "points_3d" in data else np.zeros((0, 3))
    camera_poses = data["camera_poses"]

    c0 = camera_poses[0][:3, 3]
    c1 = camera_poses[1][:3, 3]
    rel = _baseline_text(c0, c1)
    planarity = _planarity_score(points) if len(points) > 0 else 1.0

    normalized_hints = _normalize_landmark_hints(landmark_hints, len(viser_image_paths))

    captions = []
    for i, img_path in enumerate(viser_image_paths):
        metrics = _analyze_viser_image(img_path)
        hint = normalized_hints[i]
        captions.append(_compact_caption(i, metrics, rel, planarity, hint))
    return captions


def generate_viser_captions_from_folder(
    npz_path: str,
    viser_dir: str,
    landmark_hints: list[str] | None = None,
    output_filename: str = "viser_screenshot_captions.txt",
):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Scene file not found: {npz_path}")
    if not os.path.isdir(viser_dir):
        raise FileNotFoundError(f"Viser screenshot folder not found: {viser_dir}")

    viser_image_paths = _sorted_image_paths(viser_dir)
    if len(viser_image_paths) == 0:
        raise RuntimeError(f"No screenshots found in: {viser_dir}")

    captions = generate_viser_captions(
        npz_path=npz_path,
        viser_image_paths=viser_image_paths,
        landmark_hints=landmark_hints,
    )

    out_path = os.path.join(viser_dir, output_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (img_path, cap) in enumerate(zip(viser_image_paths, captions), start=1):
            name = os.path.basename(img_path)
            line = f"Screenshot {i} ({name}): {cap}"
            print(line)
            f.write(line + "\n")

    print("\nSaved caption file:", out_path)
    return captions, out_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_viser.py <path_to_scene_data.npz> [port]")
        sys.exit(1)

    npz_path = sys.argv[1]
    if not os.path.exists(npz_path):
        print(f"File not found: {npz_path}")
        sys.exit(1)

    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8081
    visualize_scene(npz_path, port, block=True)


if __name__ == "__main__":
    main()
