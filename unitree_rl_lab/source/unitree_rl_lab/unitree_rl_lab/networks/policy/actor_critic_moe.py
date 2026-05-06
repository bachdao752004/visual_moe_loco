from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from unitree_rl_lab.networks.encoders import ContactEncoder, LidarEncoder


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


class ActorCriticMoE(nn.Module):
    """B4 main policy: MoE + LSTM + ContactEncoder + LiDAR encoder.

    Observation layout
    ------------------
    - ``pt_obs``: proprioceptive state (joint pos/vel, base ang vel, projected gravity, last action).
    - ``ct_obs``: velocity command (e.g. vx, vy, yaw-rate target).
    - ``et_obs``: exteroceptive privileged channels (base lin vel, foot pos, terrain height, ...).
      **At deployment, replace with ``e_hat_t`` from** :class:`~unitree_rl_lab.networks.policy.state_estimator.StateEstimator`.
    - ``it_obs``: implicit privileged channels (contact forces, friction, ...).
      **At deployment, replace with ``i_hat_t`` from** :class:`~unitree_rl_lab.networks.policy.state_estimator.StateEstimator`.

    Training phase contract
    -----------------------
    - **Phase 1** (oracle): pass ground-truth ``et_obs`` / ``it_obs`` from the critic observation.
    - **Phase 2** (PAS):    pass ``et_mix`` / ``it_mix`` blended by :class:`Phase2Trainer`.
    - **Deployment**:       pass ``e_hat_t`` / ``i_hat_t`` from :class:`StateEstimator` — oracle signals unavailable.
    """

    is_recurrent = True

    def __init__(
        self,
        num_pt_obs: int,
        num_ct_obs: int,
        num_et_obs: int,
        num_it_obs: int,
        num_actions: int,
        num_experts: int = 3,
        token_dim: int = 256,
        contact_latent_dim: int = 4,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 2,
        gating_hidden_dim: int = 128,
        expert_actor_hidden_dims: list[int] | None = None,
        expert_critic_hidden_dims: list[int] | None = None,
        init_noise_std: float = 1.0,
        clip_min_std: float = 0.05,
    ) -> None:
        super().__init__()
        if init_noise_std <= 0.0:
            raise ValueError(f"init_noise_std must be > 0, got {init_noise_std}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")

        expert_actor_hidden_dims = expert_actor_hidden_dims or [256, 128, 128]
        expert_critic_hidden_dims = expert_critic_hidden_dims or [256, 128, 128]

        self.num_experts = num_experts
        self.num_pt_obs = num_pt_obs
        self.num_ct_obs = num_ct_obs
        self.num_et_obs = num_et_obs
        self.num_it_obs = num_it_obs
        self.clip_min_std = clip_min_std

        self.contact_encoder = ContactEncoder(
            contact_dim=num_it_obs,
            hidden_dims=(32, 16),
            latent_dim=contact_latent_dim,
        )
        self.lidar_encoder = LidarEncoder(token_dim=token_dim, point_dim=3, share_weights=False, use_layer_norm=True)

        # l_t = [p_t | e_t | z_t | c_t] -> proprio token (1 x D)
        lt_dim = num_pt_obs + num_et_obs + contact_latent_dim + num_ct_obs
        self.proprio_proj = nn.Linear(lt_dim, token_dim)

        self.lstm = nn.LSTM(
            input_size=token_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(lstm_hidden_dim, gating_hidden_dim),
            nn.ELU(),
            nn.Linear(gating_hidden_dim, num_experts),
        )

        self.actor_experts = nn.ModuleList(
            [_build_mlp(lstm_hidden_dim, expert_actor_hidden_dims, num_actions, activation=nn.ELU) for _ in range(num_experts)]
        )
        self.critic_experts = nn.ModuleList(
            [_build_mlp(lstm_hidden_dim, expert_critic_hidden_dims, 1, activation=nn.ELU) for _ in range(num_experts)]
        )

        self.log_std = nn.Parameter(torch.full((num_actions,), math.log(init_noise_std)))
        self.distribution: Normal | None = None
        self._hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_gates: torch.Tensor | None = None

    @classmethod
    def split_policy_obs(
        cls, policy_obs: torch.Tensor, num_pt: int, num_ct: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split concatenated policy observation into ``(p_t, c_t)``."""
        return policy_obs[:, :num_pt], policy_obs[:, num_pt : num_pt + num_ct]

    @classmethod
    def split_critic_obs(
        cls, critic_obs: torch.Tensor, num_et: int, num_it: int, et_start: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split critic observation into ``(e_t, i_t)`` by explicit offsets."""
        et = critic_obs[:, et_start : et_start + num_et]
        it = critic_obs[:, et_start + num_et : et_start + num_et + num_it]
        return et, it

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None or self._hidden_state is None:
            self._hidden_state = None
            return
        h, c = self._hidden_state
        done_mask = dones.bool().flatten()
        h[:, done_mask, :] = 0.0
        c[:, done_mask, :] = 0.0
        self._hidden_state = (h, c)

    def _resolve_hidden_state(
        self, hidden_state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._hidden_state if hidden_state is None else hidden_state

    def _build_lt(
        self,
        pt_obs: torch.Tensor,
        ct_obs: torch.Tensor,
        et_obs: torch.Tensor,
        it_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Build the LSTM input token ``l_t = [p_t | e_t | z_t | c_t]``.

        Args:
            pt_obs: Proprioceptive state ``(B, num_pt_obs)``.
            ct_obs: Velocity command ``(B, num_ct_obs)``.
            et_obs: Exteroceptive privileged channels ``(B, num_et_obs)``.
                    Pass estimator output ``e_hat_t`` at deployment.
            it_obs: Implicit privileged channels ``(B, num_it_obs)``.
                    Encoded into contact latent ``z_t``; pass ``i_hat_t`` at deployment.
        """
        z_t = self.contact_encoder(it_obs)
        return torch.cat([pt_obs, et_obs, z_t, ct_obs], dim=-1)

    def _encode_core(
        self,
        pt_obs: torch.Tensor,
        ct_obs: torch.Tensor,
        et_obs: torch.Tensor,
        it_obs: torch.Tensor,
        lidar_ground_points: torch.Tensor,
        lidar_forward_points: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        l_t = self._build_lt(pt_obs=pt_obs, ct_obs=ct_obs, et_obs=et_obs, it_obs=it_obs)
        proprio_token = self.proprio_proj(l_t).unsqueeze(1)

        lidar_out = self.lidar_encoder(lidar_ground_points, lidar_forward_points, pool="none")
        ground_tokens = lidar_out["ground_tokens"]
        forward_tokens = lidar_out["forward_tokens"]

        seq = torch.cat([proprio_token, ground_tokens, forward_tokens], dim=1)
        out, next_state = self.lstm(seq, hidden_state)
        return out[:, -1], next_state

    def _gate_and_blend(
        self,
        h_last: torch.Tensor,
        experts: nn.ModuleList,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(h_last)
        gates = torch.softmax(gate_logits, dim=-1)
        expert_out = torch.stack([expert(h_last) for expert in experts], dim=1)  # (B, E, D)
        blended = torch.sum(gates.unsqueeze(-1) * expert_out, dim=1)
        return blended, gates

    def update_distribution(
        self,
        pt_obs: torch.Tensor,
        ct_obs: torch.Tensor,
        et_obs: torch.Tensor,
        it_obs: torch.Tensor,
        lidar_ground_points: torch.Tensor,
        lidar_forward_points: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h_last, next_state = self._encode_core(
            pt_obs=pt_obs,
            ct_obs=ct_obs,
            et_obs=et_obs,
            it_obs=it_obs,
            lidar_ground_points=lidar_ground_points,
            lidar_forward_points=lidar_forward_points,
            hidden_state=hidden_state,
        )
        mean, gates = self._gate_and_blend(h_last, self.actor_experts)
        std = torch.exp(self.log_std).clamp_min(self.clip_min_std).expand_as(mean)
        self.distribution = Normal(mean, std)
        self._last_gates = gates
        return h_last, next_state

    def act(
        self,
        pt_obs: torch.Tensor,
        ct_obs: torch.Tensor,
        et_obs: torch.Tensor,
        it_obs: torch.Tensor,
        lidar_ground_points: torch.Tensor,
        lidar_forward_points: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        use_hidden_state = self._resolve_hidden_state(hidden_state)
        _, next_state = self.update_distribution(
            pt_obs=pt_obs,
            ct_obs=ct_obs,
            et_obs=et_obs,
            it_obs=it_obs,
            lidar_ground_points=lidar_ground_points,
            lidar_forward_points=lidar_forward_points,
            hidden_state=use_hidden_state,
        )
        self._hidden_state = next_state
        return self.distribution.sample(), next_state

    def act_inference(
        self,
        pt_obs: torch.Tensor,
        ct_obs: torch.Tensor,
        et_obs: torch.Tensor,
        it_obs: torch.Tensor,
        lidar_ground_points: torch.Tensor,
        lidar_forward_points: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        use_hidden_state = self._resolve_hidden_state(hidden_state)
        h_last, next_state = self._encode_core(
            pt_obs=pt_obs,
            ct_obs=ct_obs,
            et_obs=et_obs,
            it_obs=it_obs,
            lidar_ground_points=lidar_ground_points,
            lidar_forward_points=lidar_forward_points,
            hidden_state=use_hidden_state,
        )
        action_mean, gates = self._gate_and_blend(h_last, self.actor_experts)
        self._last_gates = gates
        self._hidden_state = next_state
        return action_mean, next_state

    def evaluate(
        self,
        pt_obs: torch.Tensor,
        ct_obs: torch.Tensor,
        et_obs: torch.Tensor,
        it_obs: torch.Tensor,
        lidar_ground_points: torch.Tensor,
        lidar_forward_points: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        use_hidden_state = self._resolve_hidden_state(hidden_state)
        h_last, next_state = self._encode_core(
            pt_obs=pt_obs,
            ct_obs=ct_obs,
            et_obs=et_obs,
            it_obs=it_obs,
            lidar_ground_points=lidar_ground_points,
            lidar_forward_points=lidar_forward_points,
            hidden_state=use_hidden_state,
        )
        value, _ = self._gate_and_blend(h_last, self.critic_experts)
        return value, next_state

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized. Call act() first.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized. Call act() first.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized. Call act() first.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized. Call act() first.")
        return self.distribution.entropy().sum(dim=-1)

    @property
    def gates(self) -> torch.Tensor | None:
        """Last gating weights from ``act()`` / ``act_inference()``.

        Note:
            ``evaluate()`` does not update this cache.
        """
        return self._last_gates
