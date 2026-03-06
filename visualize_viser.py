"""
Visualize HW3 pose-estimation output in Viser.

Usage:
    python visualize_viser.py <path_to_scene_data.npz> [port]
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import viser


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
