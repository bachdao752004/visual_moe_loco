from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Normal


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


class ActorCriticRNN(nn.Module):
    """B2 baseline: LSTM + LiDAR token fusion (no MoE)."""

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        token_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 2,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        init_noise_std: float = 1.0,
    ) -> None:
        super().__init__()
        if init_noise_std <= 0.0:
            raise ValueError(f"init_noise_std must be > 0, got {init_noise_std}")
        actor_hidden_dims = actor_hidden_dims or [256, 128, 128]
        critic_hidden_dims = critic_hidden_dims or [256, 128, 128]

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.policy_proj = nn.Linear(num_actor_obs, token_dim)
        self.lstm = nn.LSTM(
            input_size=token_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.actor = _build_mlp(lstm_hidden_dim, actor_hidden_dims, num_actions, activation=nn.ELU)
        self.critic = _build_mlp(lstm_hidden_dim, critic_hidden_dims, 1, activation=nn.ELU)
        self.log_std = nn.Parameter(torch.full((num_actions,), math.log(init_noise_std)))

        self.distribution: Normal | None = None
        self._hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None

    def _build_sequence(
        self,
        obs: torch.Tensor,
        ground_tokens: torch.Tensor,
        forward_tokens: torch.Tensor,
        proj: nn.Linear,
    ) -> torch.Tensor:
        """Build sequence: [proprio_token | ground_tokens | forward_tokens].

        Proprio is placed first so the ego-state anchors the LSTM temporal dynamics
        before the LiDAR context tokens follow.
        """
        proprio_token = proj(obs).unsqueeze(1)
        return torch.cat([proprio_token, ground_tokens, forward_tokens], dim=1)

    def _encode_last_hidden(
        self,
        obs: torch.Tensor,
        ground_tokens: torch.Tensor,
        forward_tokens: torch.Tensor,
        proj: nn.Linear,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        seq = self._build_sequence(obs, ground_tokens, forward_tokens, proj)
        out, next_state = self.lstm(seq, hidden_state)
        return out[:, -1], next_state

    def _resolve_hidden_state(
        self, hidden_state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._hidden_state if hidden_state is None else hidden_state

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset recurrent state globally or for selected done environments."""
        if dones is None or self._hidden_state is None:
            self._hidden_state = None
            return
        h, c = self._hidden_state
        done_mask = dones.bool().flatten()
        h[:, done_mask, :] = 0.0
        c[:, done_mask, :] = 0.0
        self._hidden_state = (h, c)

    def update_distribution(
        self,
        observations: torch.Tensor,
        ground_tokens: torch.Tensor,
        forward_tokens: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h_last, next_state = self._encode_last_hidden(
            observations,
            ground_tokens,
            forward_tokens,
            proj=self.policy_proj,
            hidden_state=hidden_state,
        )
        mean = self.actor(h_last)
        std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)
        return h_last, next_state

    def act(
        self,
        observations: torch.Tensor,
        ground_tokens: torch.Tensor,
        forward_tokens: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        use_hidden_state = self._resolve_hidden_state(hidden_state)
        _, next_state = self.update_distribution(
            observations=observations,
            ground_tokens=ground_tokens,
            forward_tokens=forward_tokens,
            hidden_state=use_hidden_state,
        )
        self._hidden_state = next_state
        return self.distribution.sample(), next_state

    def act_inference(
        self,
        observations: torch.Tensor,
        ground_tokens: torch.Tensor,
        forward_tokens: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        use_hidden_state = self._resolve_hidden_state(hidden_state)
        h_last, next_state = self._encode_last_hidden(
            observations,
            ground_tokens,
            forward_tokens,
            proj=self.policy_proj,
            hidden_state=use_hidden_state,
        )
        self._hidden_state = next_state
        return self.actor(h_last), next_state

    def evaluate(
        self,
        critic_observations: torch.Tensor,
        ground_tokens: torch.Tensor,
        forward_tokens: torch.Tensor,
        hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        use_hidden_state = self._resolve_hidden_state(hidden_state)
        # critic_observations may include privileged channels (e_t + i_t), but B2 baseline
        # intentionally uses only actor-observation slice through the shared LSTM.
        # Privileged fusion is handled in actor_critic_moe.py.
        critic_obs_for_lstm = critic_observations[:, : self.num_actor_obs]
        h_last, next_state = self._encode_last_hidden(
            critic_obs_for_lstm,
            ground_tokens,
            forward_tokens,
            proj=self.policy_proj,
            hidden_state=use_hidden_state,
        )
        return self.critic(h_last), next_state

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
