from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class LidarRayEncoder(nn.Module):
    """Encode LiDAR points per-ray into token embeddings.

    Input:
        - ``points``: ``(B, N, 3)`` where ``N`` is number of rays.
    Output:
        - token sequence ``(B, N, token_dim)``.
    """

    def __init__(
        self,
        token_dim: int = 256,
        point_dim: int = 3,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError(f"token_dim must be > 0, got {token_dim}")
        if point_dim <= 0:
            raise ValueError(f"point_dim must be > 0, got {point_dim}")

        self.proj = nn.Linear(point_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim) if use_layer_norm else nn.Identity()
        self.token_dim = token_dim

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(points)
        tokens = self.norm(tokens)
        return tokens


class LidarEncoder(nn.Module):
    """Two-branch LiDAR encoder for ground and forward rays.

    This module keeps branch-specific encoders (ground/forward) and returns either:
    - separate token sequences (for LSTM token pipeline), or
    - pooled embeddings per branch.
    """

    def __init__(
        self,
        token_dim: int = 256,
        point_dim: int = 3,
        share_weights: bool = False,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.ground_encoder = LidarRayEncoder(
            token_dim=token_dim,
            point_dim=point_dim,
            use_layer_norm=use_layer_norm,
        )
        if share_weights:
            self.forward_encoder = self.ground_encoder
        else:
            self.forward_encoder = LidarRayEncoder(
                token_dim=token_dim,
                point_dim=point_dim,
                use_layer_norm=use_layer_norm,
            )
        self.token_dim = token_dim

    def encode_tokens(self, ground_points: torch.Tensor, forward_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both ray sets to token sequences.

        Args:
            ground_points: ``(B, N_ground, 3)``
            forward_points: ``(B, N_forward, 3)``
        Returns:
            ``(ground_tokens, forward_tokens)`` with shapes
            ``(B, N_ground, D)`` and ``(B, N_forward, D)``.
        """
        g_tokens = self.ground_encoder(ground_points)
        f_tokens = self.forward_encoder(forward_points)
        return g_tokens, f_tokens

    def forward(
        self,
        ground_points: torch.Tensor,
        forward_points: torch.Tensor,
        pool: Literal["none", "mean", "max"] = "none",
    ) -> dict[str, torch.Tensor]:
        """Encode LiDAR inputs and optionally pool tokens.

        Returns:
            Dict with:
            - ``ground_tokens`` / ``forward_tokens`` always.
            - ``ground_embed`` / ``forward_embed`` only when ``pool != "none"``.
        """
        ground_tokens, forward_tokens = self.encode_tokens(ground_points, forward_points)
        out = {
            "ground_tokens": ground_tokens,
            "forward_tokens": forward_tokens,
        }
        if pool == "mean":
            out["ground_embed"] = ground_tokens.mean(dim=1)
            out["forward_embed"] = forward_tokens.mean(dim=1)
        elif pool == "max":
            out["ground_embed"] = ground_tokens.max(dim=1).values
            out["forward_embed"] = forward_tokens.max(dim=1).values
        elif pool != "none":
            raise ValueError(f"Unknown pool mode: {pool}")
        return out
