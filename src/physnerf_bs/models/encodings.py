from __future__ import annotations

import torch


def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    if num_freqs <= 0:
        return x
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)
    encoded = [x]
    for freq in freqs:
        encoded.append(torch.sin(freq * x))
        encoded.append(torch.cos(freq * x))
    return torch.cat(encoded, dim=-1)
