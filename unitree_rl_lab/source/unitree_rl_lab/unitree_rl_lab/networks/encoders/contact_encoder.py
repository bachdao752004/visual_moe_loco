from __future__ import annotations

import torch
import torch.nn as nn


class ContactEncoder(nn.Module):
    """Encode implicit privileged contact features into a compact latent vector.

    Expected input is ``(batch, contact_dim)`` where ``contact_dim`` can be:
    - per-link force magnitudes (e.g. from ``contact_forces_norm``), or
    - concatenated contact-related features prepared by the trainer/policy.
    """

    def __init__(
        self,
        contact_dim: int,
        hidden_dims: tuple[int, int] = (32, 16),
        latent_dim: int = 4,
        activation: type[nn.Module] = nn.ELU,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if contact_dim <= 0:
            raise ValueError(f"contact_dim must be > 0, got {contact_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")

        layers: list[nn.Module] = []
        in_dim = contact_dim
        for hid in hidden_dims:
            layers.append(nn.Linear(in_dim, hid))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hid))
            layers.append(activation())
            in_dim = hid
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

        self.contact_dim = contact_dim
        self.latent_dim = latent_dim

    def forward(self, contact_features: torch.Tensor) -> torch.Tensor:
        """Encode contact features to latent ``z_t``.

        Args:
            contact_features: Tensor with shape ``(B, contact_dim)``.
        Returns:
            Latent tensor with shape ``(B, latent_dim)``.
        """
        return self.net(contact_features)
