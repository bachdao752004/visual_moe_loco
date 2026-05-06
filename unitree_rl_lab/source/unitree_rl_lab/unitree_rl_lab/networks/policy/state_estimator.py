from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class StateEstimator(nn.Module):
    """Phase-2 privileged state estimator.

    Estimate privileged channels from deploy-available inputs.

    Typical usage:
    - Input: policy stream ``[p_t | c_t]`` (+ optional LiDAR embedding).
    - Output: ``e_hat_t`` and ``i_hat_t`` (or concatenated ``l_hat_t = [e_hat_t | i_hat_t]``).
    """

    def __init__(
        self,
        num_policy_obs: int,
        num_et_obs: int,
        num_it_obs: int,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        head_hidden_dims: list[int] | None = None,
        lidar_embed_dim: int = 0,
    ) -> None:
        super().__init__()
        if num_policy_obs <= 0:
            raise ValueError(f"num_policy_obs must be > 0, got {num_policy_obs}")
        if num_et_obs < 0 or num_it_obs < 0:
            raise ValueError("num_et_obs and num_it_obs must be >= 0")
        if num_et_obs + num_it_obs == 0:
            raise ValueError("At least one of num_et_obs or num_it_obs must be > 0")

        head_hidden_dims = head_hidden_dims or [256, 128]

        self.num_policy_obs = num_policy_obs
        self.num_et_obs = num_et_obs
        self.num_it_obs = num_it_obs
        self.lidar_embed_dim = lidar_embed_dim
        self.num_latent = num_et_obs + num_it_obs

        input_dim = num_policy_obs + lidar_embed_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.head = _build_mlp(lstm_hidden_dim, head_hidden_dims, self.num_latent, activation=nn.ELU)
        self._hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None

    def _fuse_inputs(
        self,
        policy_obs_seq: torch.Tensor,
        lidar_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if policy_obs_seq.ndim == 2:
            policy_obs_seq = policy_obs_seq.unsqueeze(1)
        if lidar_embed is None:
            return policy_obs_seq
        if lidar_embed.ndim == 2:
            lidar_embed = lidar_embed.unsqueeze(1)
        return torch.cat([policy_obs_seq, lidar_embed], dim=-1)

    def _resolve_hidden_state(
        self, hidden_state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._hidden_state if hidden_state is None else hidden_state

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset recurrent estimator state globally or for selected done envs."""
        if dones is None or self._hidden_state is None:
            self._hidden_state = None
            return
        h, c = self._hidden_state
        done_mask = dones.bool().flatten()
        h[:, done_mask, :] = 0.0
        c[:, done_mask, :] = 0.0
        self._hidden_state = (h, c)

    def forward(
        self,
        policy_obs: torch.Tensor,
        lidar_embed: torch.Tensor | None = None,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        update_internal_state: bool = True,
    ) -> torch.Tensor:
        """Return concatenated latent prediction ``l_hat_t = [e_hat_t | i_hat_t]``."""
        x = self._fuse_inputs(policy_obs, lidar_embed)
        use_hidden_state = self._resolve_hidden_state(hidden_state)
        out, next_state = self.lstm(x, use_hidden_state)
        if update_internal_state:
            self._hidden_state = next_state
        return self.head(out[:, -1])

    def estimate(
        self,
        policy_obs: torch.Tensor,
        lidar_embed: torch.Tensor | None = None,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        update_internal_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return split estimate ``(e_hat_t, i_hat_t)``."""
        latent = self.forward(policy_obs, lidar_embed, hidden_state=hidden_state, update_internal_state=update_internal_state)
        e_hat = latent[:, : self.num_et_obs] if self.num_et_obs > 0 else latent.new_zeros((latent.shape[0], 0))
        i_hat = latent[:, self.num_et_obs :] if self.num_it_obs > 0 else latent.new_zeros((latent.shape[0], 0))
        return e_hat, i_hat

    def compute_loss(
        self,
        policy_obs: torch.Tensor,
        et_target: torch.Tensor | None = None,
        it_target: torch.Tensor | None = None,
        lidar_embed: torch.Tensor | None = None,
        et_weight: float = 1.0,
        it_weight: float = 1.0,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        update_internal_state: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute weighted MSE loss for Phase-2 training."""
        e_hat, i_hat = self.estimate(
            policy_obs=policy_obs,
            lidar_embed=lidar_embed,
            hidden_state=hidden_state,
            update_internal_state=update_internal_state,
        )

        losses: dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=policy_obs.device)

        if et_target is not None and self.num_et_obs > 0:
            e_loss = F.mse_loss(e_hat, et_target)
            losses["et_mse"] = e_loss
            total_loss = total_loss + et_weight * e_loss

        if it_target is not None and self.num_it_obs > 0:
            i_loss = F.mse_loss(i_hat, it_target)
            losses["it_mse"] = i_loss
            total_loss = total_loss + it_weight * i_loss

        losses["total"] = total_loss
        return total_loss, losses
