from __future__ import annotations

import torch


def estimate_linear_velocity(data_seq: torch.Tensor, dt: float) -> torch.Tensor:
    """Estimate per-frame velocity with mixed finite differences.

    Input shape: (B, T, ...).
    """
    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / dt
    middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2 * dt)
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / dt
    return torch.cat([init_vel, middle_vel, final_vel], dim=1)


def velocity2position_mixeddiff(
    vel_seq: torch.Tensor,
    dt: float,
    init_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `estimate_linear_velocity` for the same mixed scheme.

    Returns:
        positions: (B, T, ...)
        final_pos: extra final-position estimate using backward difference.
    """
    b, t = vel_seq.shape[:2]
    positions = torch.zeros_like(vel_seq)
    positions[:, 0] = init_pos
    if t == 1:
        return positions, positions[:, 0]

    positions[:, 1] = positions[:, 0] + vel_seq[:, 0] * dt
    for i in range(1, t - 1):
        positions[:, i + 1] = positions[:, i - 1] + vel_seq[:, i] * (2.0 * dt)

    final_pos = positions[:, -1] + vel_seq[:, -1] * dt
    return positions, final_pos
