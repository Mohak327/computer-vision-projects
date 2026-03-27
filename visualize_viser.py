import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import viser

from dataset_3d import RaysData, load_data
from rendering import sample_along_rays


@dataclass
class Config:
    data_path: str = "lego_200x200.npz"
    near: float = 2.0
    far: float = 6.0
    num_samples_along_ray: int = 64
    num_rays: int = 300
    num_cameras: int = 1
    camera_start_idx: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    port: int = 8080


def make_viser_server(port: int, share: bool = True) -> viser.ViserServer:
    """Create and return a Viser server instance."""
    server = viser.ViserServer(port=port, share=share)
    return server


def start_viser_server(port: int, share: bool = True) -> viser.ViserServer:
    """Create, start, and return a Viser server instance."""
    server = make_viser_server(port=port, share=share)
    print(f"Viser server running on port {port}. Press Ctrl+C to stop.")
    return server


def main(cfg: Config):
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(data_path=cfg.data_path)

    H, W = images_train.shape[1], images_train.shape[2]

    images_train_t = torch.as_tensor(images_train, dtype=torch.float32, device=cfg.device)
    c2ws_train_t = torch.as_tensor(c2ws_train, dtype=torch.float32, device=cfg.device)
    K_t = torch.as_tensor(K, dtype=torch.float32, device=cfg.device)

    dataset = RaysData(images_train_t, K_t, c2ws_train_t, split="train", device=cfg.device)

    # Verify uvs aren't flipped
    uvs_start = 0
    uvs_end = 40_000
    sample_uvs = dataset.uvs[uvs_start:uvs_end].cpu()
    assert torch.allclose(
        images_train_t[0, sample_uvs[:, 1], sample_uvs[:, 0]].cpu(),
        dataset.gt_rgbs[uvs_start:uvs_end].cpu(),
        atol=1e-6,
    )
    print("UVs assertion passed!")

    # Sample random rays from a selectable camera range.
    num_pixels_per_image = H * W
    cam_start = int(np.clip(cfg.camera_start_idx, 0, max(0, dataset.num_images - 1)))
    cam_count = int(np.clip(cfg.num_cameras, 1, dataset.num_images - cam_start))
    cam_end = cam_start + cam_count
    global_start = cam_start * num_pixels_per_image
    global_end = cam_end * num_pixels_per_image
    indices = np.random.randint(low=global_start, high=global_end, size=cfg.num_rays)

    rays_o = dataset.rays_o[indices]
    rays_d = dataset.rays_d[indices]

    points, _ = sample_along_rays(
        ray_origins=rays_o,
        ray_directions=rays_d,
        near=cfg.near,
        far=cfg.far,
        n_samples=cfg.num_samples_along_ray,
        perturb=True,
    )  # (num_rays, num_samples, 3)

    # Convert to numpy for viser
    rays_o_np = rays_o.cpu().detach().numpy()
    rays_d_np = rays_d.cpu().detach().numpy()
    points_np = points.cpu().detach().numpy()
    images_np = images_train_t.cpu().detach().numpy()
    c2ws_np = c2ws_train_t.cpu().detach().numpy()
    K_np = K_t.cpu().detach().numpy()

    server = start_viser_server(port=cfg.port, share=True)

    fov = float(2 * np.arctan2(H / 2, K_np[0, 0]))
    aspect = float(W / H)

    for i, (image, c2w) in enumerate(zip(images_np[cam_start:cam_end], c2ws_np[cam_start:cam_end])):
        server.scene.add_camera_frustum(
            f"/cameras/{cam_start + i}",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image,
        )

    for i, (o, d) in enumerate(zip(rays_o_np, rays_d_np)):
        positions = np.stack((o, o + d * cfg.far))
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}",
            positions=positions,
            line_width=2.0,
        )

    server.scene.add_point_cloud(
        "/samples",
        colors=np.zeros_like(points_np).reshape(-1, 3),
        points=points_np.reshape(-1, 3),
        point_size=0.03,
    )

    print(
        f"Visualizing rays from cameras [{cam_start}, {cam_end - 1}] "
        f"({cam_count} camera(s))."
    )
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viser visualization for camera frustums, rays, and ray samples.")
    parser.add_argument("--data_path", type=str, default=Config.data_path)
    parser.add_argument("--near", type=float, default=Config.near)
    parser.add_argument("--far", type=float, default=Config.far)
    parser.add_argument("--num_samples_along_ray", type=int, default=Config.num_samples_along_ray)
    parser.add_argument("--num_rays", type=int, default=Config.num_rays)
    parser.add_argument("--num_cameras", type=int, default=Config.num_cameras)
    parser.add_argument("--camera_start_idx", type=int, default=Config.camera_start_idx)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--port", type=int, default=Config.port)

    args = parser.parse_args()
    cfg = Config(
        data_path=args.data_path,
        near=args.near,
        far=args.far,
        num_samples_along_ray=args.num_samples_along_ray,
        num_rays=args.num_rays,
        num_cameras=args.num_cameras,
        camera_start_idx=args.camera_start_idx,
        device=args.device,
        port=args.port,
    )
    main(cfg)