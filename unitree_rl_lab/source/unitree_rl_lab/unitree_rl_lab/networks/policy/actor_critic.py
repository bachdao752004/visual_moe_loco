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


class ActorCritic(nn.Module):
    """B1 baseline: plain MLP actor-critic (no RNN, no MoE, no LiDAR branch)."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        init_noise_std: float = 1.0,
    ) -> None:
        super().__init__()
        if init_noise_std <= 0.0:
            raise ValueError(f"init_noise_std must be > 0, got {init_noise_std}")
        actor_hidden_dims = actor_hidden_dims or [256, 128, 128]
        critic_hidden_dims = critic_hidden_dims or [256, 128, 128]

        self.actor = _build_mlp(num_actor_obs, actor_hidden_dims, num_actions, activation=nn.ELU)
        self.critic = _build_mlp(num_critic_obs, critic_hidden_dims, 1, activation=nn.ELU)
        self.log_std = nn.Parameter(torch.full((num_actions,), math.log(init_noise_std)))

        self.distribution: Normal | None = None

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(observations)
        std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.actor(observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution is not initialized. Call act() first.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

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
