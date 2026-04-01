from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class WorldModelOutput:
    predicted_frames: torch.Tensor
    latent_states: torch.Tensor
    gating_values: torch.Tensor


class MotorGatedWorldModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        motor_dim: int,
        hidden_channels: int = 24,
        gated_channels: int = 8,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gated_channels = min(gated_channels, hidden_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.state_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.motor_gate = nn.Sequential(
            nn.Linear(motor_dim, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, self.gated_channels),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def step(self, state: torch.Tensor, observation: torch.Tensor, motor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_obs = self.encoder(observation)
        recurrent = self.state_conv(state)
        gate = torch.sigmoid(self.motor_gate(motor)).unsqueeze(-1).unsqueeze(-1)
        gated_prefix = recurrent[:, : self.gated_channels] * gate
        gated_suffix = recurrent[:, self.gated_channels :]
        gated = torch.cat([gated_prefix, gated_suffix], dim=1)
        next_state = torch.tanh(gated + encoded_obs)
        return next_state, gate.squeeze(-1).squeeze(-1)

    def forward(self, frames: torch.Tensor, motor: torch.Tensor) -> WorldModelOutput:
        if frames.ndim != 5:
            raise ValueError("Expected frames with shape [B, T, C, H, W].")
        batch, steps, channels, height, width = frames.shape
        state = torch.zeros((batch, self.hidden_channels, height, width), device=frames.device, dtype=frames.dtype)
        preds = []
        states = []
        gates = []
        for t in range(steps - 1):
            state, gate = self.step(state, frames[:, t], motor[:, t])
            preds.append(self.decoder(state))
            states.append(state)
            gates.append(gate)
        return WorldModelOutput(
            predicted_frames=torch.stack(preds, dim=1),
            latent_states=torch.stack(states, dim=1),
            gating_values=torch.stack(gates, dim=1),
        )
