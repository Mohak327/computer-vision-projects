from __future__ import annotations

import torch
from torch import nn


def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """Apply NeRF sinusoidal positional encoding.

    Args:
        x: (..., C)
        num_freqs: number of frequency bands

    Returns:
        (..., C * (2 * num_freqs + 1))
    """
    if num_freqs <= 0:
        return x

    enc = [x]
    freq_bands = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)
    for freq in freq_bands:
        enc.append(torch.sin(freq * x))
        enc.append(torch.cos(freq * x))
    return torch.cat(enc, dim=-1)


class NeRFMLP(nn.Module):
    """Compact NeRF MLP producing density and color."""

    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 8,
        skip_layer: int = 4,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")

        self.pos_freqs = pos_freqs
        self.dir_freqs = dir_freqs
        self.skip_layer = skip_layer

        pos_in_dim = 3 * (2 * pos_freqs + 1)
        dir_in_dim = 3 * (2 * dir_freqs + 1)

        self.pts_layers = nn.ModuleList()
        self.pts_layers.append(nn.Linear(pos_in_dim, hidden_dim))
        for i in range(1, n_layers):
            in_dim = hidden_dim
            if i == skip_layer:
                in_dim += pos_in_dim
            self.pts_layers.append(nn.Linear(in_dim, hidden_dim))

        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)

        self.color_layers = nn.Sequential(
            nn.Linear(hidden_dim + dir_in_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, xyz: torch.Tensor, ray_dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run NeRF MLP.

        Args:
            xyz: (N, S, 3)
            ray_dirs: (N, 3)

        Returns:
            sigma: (N, S, 1)
            rgb: (N, S, 3)
        """
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape (N, S, 3)")
        if ray_dirs.ndim != 2 or ray_dirs.shape[-1] != 3:
            raise ValueError("ray_dirs must have shape (N, 3)")

        param_dtype = self.pts_layers[0].weight.dtype
        param_device = self.pts_layers[0].weight.device
        xyz = xyz.to(device=param_device, dtype=param_dtype)
        ray_dirs = ray_dirs.to(device=param_device, dtype=param_dtype)

        n_rays, n_samples, _ = xyz.shape

        xyz_flat = xyz.reshape(-1, 3)
        dirs_expanded = ray_dirs[:, None, :].expand(-1, n_samples, -1).reshape(-1, 3)

        xyz_enc = positional_encoding(xyz_flat, self.pos_freqs)
        dir_enc = positional_encoding(dirs_expanded, self.dir_freqs)

        h = xyz_enc
        for i, layer in enumerate(self.pts_layers):
            if i == self.skip_layer:
                h = torch.cat([h, xyz_enc], dim=-1)
            h = self.relu(layer(h))

        sigma = self.sigma_head(h)
        features = self.feature_head(h)
        rgb = self.color_layers(torch.cat([features, dir_enc], dim=-1))

        sigma = sigma.reshape(n_rays, n_samples, 1)
        rgb = rgb.reshape(n_rays, n_samples, 3)
        return sigma, rgb
