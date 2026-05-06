from __future__ import annotations

import torch
import torch.nn as nn


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: type[nn.Module] = nn.ELU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class TraversabilityEstimator(nn.Module):
    """Estimate per-cell traversability score from pooled LiDAR embeddings.

    This module maps a fused LiDAR embedding (ground + forward branches, already pooled)
    to a scalar traversability score in ``[0, 1]`` — higher meaning more traversable terrain.

    Typical usage::

        lidar_out = lidar_encoder(ground_pts, forward_pts, pool="mean")
        lidar_embed = torch.cat([lidar_out["ground_embed"], lidar_out["forward_embed"]], dim=-1)
        traversability = traversability_estimator(lidar_embed)   # (B, 1)

    Args:
        lidar_embed_dim: Dimensionality of the concatenated pooled LiDAR embedding
            (``ground_embed_dim + forward_embed_dim``, i.e. ``2 * token_dim`` by default).
        hidden_dims: Hidden layer sizes for the MLP head.
        activation: Activation class to use between linear layers.
    """

    def __init__(
        self,
        lidar_embed_dim: int,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()
        if lidar_embed_dim <= 0:
            raise ValueError(f"lidar_embed_dim must be > 0, got {lidar_embed_dim}")

        hidden_dims = hidden_dims or [128, 64]
        self.net = _build_mlp(lidar_embed_dim, hidden_dims, 1, activation=activation)
        self.lidar_embed_dim = lidar_embed_dim

    def forward(self, lidar_embed: torch.Tensor) -> torch.Tensor:
        """Estimate traversability from pooled LiDAR embedding.

        Args:
            lidar_embed: Pooled LiDAR feature vector ``(B, lidar_embed_dim)``.

        Returns:
            Traversability score ``(B, 1)`` in ``[0, 1]`` after sigmoid.
        """
        return torch.sigmoid(self.net(lidar_embed))
