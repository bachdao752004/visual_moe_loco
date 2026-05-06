from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from unitree_rl_lab.networks.policy.actor_critic_moe import ActorCriticMoE
from unitree_rl_lab.networks.policy.state_estimator import StateEstimator
from unitree_rl_lab.networks.trainer.stage_manager import StageManager, TrainingStage


@dataclass
class Phase2TrainerCfg:
    """Config for Stage-2 PAS + estimator training."""

    rollout_steps: int = 24
    num_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    device: str = "cuda"
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.01

    # PAS schedule: probability of using estimator output.
    p_start: float = 0.0
    p_end: float = 1.0
    p_ramp_iterations: int = 10000

    et_loss_weight: float = 0.1
    it_loss_weight: float = 0.1

    # Observation split sizes (must match env observation layout).
    num_pt_obs: int = 42
    num_ct_obs: int = 6
    # Critic layout in velocity_env_cfg.py (current):
    # e_t: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(6)
    #      + joint_pos_rel(12) + joint_vel_rel(12) + joint_effort(12) + last_action(12) = 63
    # i_t: foot_contact_bool(4) + contact_forces_norm(all bodies -> 17) = 21
    num_et_obs: int = 63
    num_it_obs: int = 21
    et_start: int = 0


class _EstimatorBatch:
    def __init__(self) -> None:
        self.policy_obs: list[torch.Tensor] = []
        self.critic_obs: list[torch.Tensor] = []
        self.ground_rays: list[torch.Tensor] = []
        self.forward_rays: list[torch.Tensor] = []
        self.et_mix: list[torch.Tensor] = []
        self.it_mix: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []

    def stack(self) -> dict[str, torch.Tensor]:
        return {
            "policy_obs": torch.stack(self.policy_obs, dim=0),  # (T, N, Dp)
            "critic_obs": torch.stack(self.critic_obs, dim=0),  # (T, N, Dc)
            "ground_rays": torch.stack(self.ground_rays, dim=0),  # (T, N, 12, 3)
            "forward_rays": torch.stack(self.forward_rays, dim=0),  # (T, N, 16, 3)
            "et_mix": torch.stack(self.et_mix, dim=0),
            "it_mix": torch.stack(self.it_mix, dim=0),
            "actions": torch.stack(self.actions, dim=0),
            "log_probs": torch.stack(self.log_probs, dim=0),
            "values": torch.stack(self.values, dim=0),
            "rewards": torch.stack(self.rewards, dim=0),
            "dones": torch.stack(self.dones, dim=0),  # (T, N)
        }


class Phase2Trainer:
    """Joint Stage-2 training: PPO + reconstruction with PAS."""

    def __init__(
        self,
        env: Any,
        policy: ActorCriticMoE,
        estimator: StateEstimator,
        cfg: Phase2TrainerCfg,
        stage_manager: StageManager | None = None,
    ) -> None:
        self.env = env
        self.policy = policy.to(cfg.device)
        self.estimator = estimator.to(cfg.device)
        self.cfg = cfg
        self.stage_manager = stage_manager or StageManager(initial_stage=TrainingStage.STAGE2_ESTIMATOR)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.estimator.parameters()),
            lr=cfg.learning_rate,
        )
        self.iteration = 0

        # Freeze policy, unfreeze estimator as stage defaults.
        self.stage_manager.apply_stage(
            {
                "actor_critic": self.policy,
                "state_estimator": self.estimator,
                "contact_encoder": self.policy.contact_encoder,
                "lidar_encoder": self.policy.lidar_encoder,
            }
        )

    def _current_mix_prob(self) -> float:
        if self.cfg.p_ramp_iterations <= 0:
            return self.cfg.p_end
        alpha = min(max(self.iteration / self.cfg.p_ramp_iterations, 0.0), 1.0)
        return self.cfg.p_start + (self.cfg.p_end - self.cfg.p_start) * alpha

    def _extract_obs(self, obs_dict: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_obs = obs_dict["policy"].to(self.cfg.device)
        critic_obs = obs_dict["critic"].to(self.cfg.device)
        lidar_group = obs_dict["lidar"]
        ground_rays = lidar_group["ground_rays"].to(self.cfg.device)
        forward_rays = lidar_group["forward_rays"].to(self.cfg.device)
        return policy_obs, critic_obs, ground_rays, forward_rays

    def _split_obs(
        self, policy_obs: torch.Tensor, critic_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pt_obs, ct_obs = self.policy.split_policy_obs(
            policy_obs=policy_obs,
            num_pt=self.cfg.num_pt_obs,
            num_ct=self.cfg.num_ct_obs,
        )
        et_obs, it_obs = self.policy.split_critic_obs(
            critic_obs=critic_obs,
            num_et=self.cfg.num_et_obs,
            num_it=self.cfg.num_it_obs,
            et_start=self.cfg.et_start,
        )
        return pt_obs, ct_obs, et_obs, it_obs

    def _blend_privileged(
        self,
        et_oracle: torch.Tensor,
        it_oracle: torch.Tensor,
        et_hat: torch.Tensor,
        it_hat: torch.Tensor,
        p: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if p <= 0.0:
            return et_oracle, it_oracle
        if p >= 1.0:
            return et_hat, it_hat
        mask = (torch.rand((et_oracle.shape[0], 1), device=et_oracle.device) < p).float()
        et_mix = mask * et_hat + (1.0 - mask) * et_oracle
        it_mix = mask * it_hat + (1.0 - mask) * it_oracle
        return et_mix, it_mix

    def build_lidar_embed_helper(
        self,
        ground_rays: torch.Tensor,
        forward_rays: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build pooled LiDAR embedding only when estimator expects it."""
        expected = int(self.estimator.lidar_embed_dim)
        if expected <= 0:
            return None
        with torch.no_grad():
            lidar_out = self.policy.lidar_encoder(ground_rays, forward_rays, pool="mean")
            lidar_embed = torch.cat([lidar_out["ground_embed"], lidar_out["forward_embed"]], dim=-1)
        got = int(lidar_embed.shape[-1])
        if got != expected:
            raise RuntimeError(
                f"StateEstimator lidar_embed_dim mismatch: expected {expected}, got {got}. "
                "Set estimator.lidar_embed_dim to match pooled LiDAR feature width."
            )
        return lidar_embed

    def collect_rollout(self, obs_dict: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        batch = _EstimatorBatch()
        policy_obs, critic_obs, ground_rays, forward_rays = self._extract_obs(obs_dict)
        p_mix = self._current_mix_prob()

        for _ in range(self.cfg.rollout_steps):
            pt_obs, ct_obs, et_obs, it_obs = self._split_obs(policy_obs, critic_obs)

            with torch.no_grad():
                lidar_embed = self.build_lidar_embed_helper(ground_rays, forward_rays)
                et_hat, it_hat = self.estimator.estimate(
                    policy_obs=policy_obs,
                    lidar_embed=lidar_embed,
                    update_internal_state=True,
                )
                et_mix, it_mix = self._blend_privileged(et_obs, it_obs, et_hat, it_hat, p=p_mix)

                actions, _ = self.policy.act(
                    pt_obs=pt_obs,
                    ct_obs=ct_obs,
                    et_obs=et_mix,
                    it_obs=it_mix,
                    lidar_ground_points=ground_rays,
                    lidar_forward_points=forward_rays,
                )
                log_prob = self.policy.get_actions_log_prob(actions)
                value, _ = self.policy.evaluate(
                    pt_obs=pt_obs,
                    ct_obs=ct_obs,
                    et_obs=et_mix,
                    it_obs=it_mix,
                    lidar_ground_points=ground_rays,
                    lidar_forward_points=forward_rays,
                )

            step_result = self.env.step(actions)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, infos = step_result
                done = torch.logical_or(terminated, truncated)
            elif len(step_result) == 4:
                next_obs, reward, done, infos = step_result
            else:
                raise RuntimeError(f"Unexpected env.step() return length: {len(step_result)}")
            done = done.to(self.cfg.device)

            batch.policy_obs.append(policy_obs)
            batch.critic_obs.append(critic_obs)
            batch.ground_rays.append(ground_rays)
            batch.forward_rays.append(forward_rays)
            batch.et_mix.append(et_mix)
            batch.it_mix.append(it_mix)
            batch.actions.append(actions)
            batch.log_probs.append(log_prob)
            batch.values.append(value.squeeze(-1))
            batch.rewards.append(reward.to(self.cfg.device))
            batch.dones.append(done)

            self.policy.reset(done)
            self.estimator.reset(done)
            policy_obs, critic_obs, ground_rays, forward_rays = self._extract_obs(next_obs)

        with torch.no_grad():
            pt_obs, ct_obs, et_obs, it_obs = self._split_obs(policy_obs, critic_obs)
            lidar_embed = self.build_lidar_embed_helper(ground_rays, forward_rays)
            et_hat, it_hat = self.estimator.estimate(policy_obs=policy_obs, lidar_embed=lidar_embed, update_internal_state=False)
            et_mix, it_mix = self._blend_privileged(et_obs, it_obs, et_hat, it_hat, p=p_mix)
            last_value, _ = self.policy.evaluate(
                pt_obs=pt_obs,
                ct_obs=ct_obs,
                et_obs=et_mix,
                it_obs=it_mix,
                lidar_ground_points=ground_rays,
                lidar_forward_points=forward_rays,
            )
            last_value = last_value.squeeze(-1)

        stacked = batch.stack()
        advantages, returns = self._compute_gae(
            rewards=stacked["rewards"],
            values=stacked["values"],
            dones=stacked["dones"],
            last_value=last_value,
        )
        stacked["advantages"] = advantages
        stacked["returns"] = returns
        extras = {
            "last_obs": {
                "policy": policy_obs,
                "critic": critic_obs,
                "lidar": {"ground_rays": ground_rays, "forward_rays": forward_rays},
            },
            "p_mix": p_mix,
        }
        return stacked, extras

    @staticmethod
    def _flatten_time_env(x: torch.Tensor) -> torch.Tensor:
        t, n = x.shape[:2]
        return x.reshape(t * n, *x.shape[2:])

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        gae = torch.zeros_like(last_value)
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t].float()
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * next_value * nonterminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * nonterminal * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    def _update_joint(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        policy_obs = self._flatten_time_env(rollout["policy_obs"])
        critic_obs = self._flatten_time_env(rollout["critic_obs"])
        ground_rays = self._flatten_time_env(rollout["ground_rays"])
        forward_rays = self._flatten_time_env(rollout["forward_rays"])
        et_mix = self._flatten_time_env(rollout["et_mix"])
        it_mix = self._flatten_time_env(rollout["it_mix"])
        actions = self._flatten_time_env(rollout["actions"])
        old_log_probs = self._flatten_time_env(rollout["log_probs"])
        returns = self._flatten_time_env(rollout["returns"])
        advantages = self._flatten_time_env(rollout["advantages"])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Targets from oracle privileged channels in critic obs.
        _pt_obs, _ct_obs, et_target, it_target = self._split_obs(policy_obs, critic_obs)

        total = policy_obs.shape[0]
        mb = max(total // self.cfg.num_minibatches, 1)
        metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "recon_loss": 0.0,
            "et_mse": 0.0,
            "it_mse": 0.0,
        }
        updates = 0

        self.policy.reset(dones=None)
        self.estimator.reset(dones=None)

        for _ in range(self.cfg.num_epochs):
            perm = torch.randperm(total, device=policy_obs.device)
            for start in range(0, total, mb):
                idx = perm[start : start + mb]
                mb_policy = policy_obs[idx]
                mb_et = et_target[idx]
                mb_it = it_target[idx]
                mb_ground = ground_rays[idx]
                mb_forward = forward_rays[idx]
                mb_et_mix = et_mix[idx]
                mb_it_mix = it_mix[idx]
                mb_actions = actions[idx]
                mb_old_logp = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_adv = advantages[idx]

                mb_critic = critic_obs[idx]
                pt_obs, ct_obs, _et_obs, _it_obs = self._split_obs(mb_policy, mb_critic)

                self.policy.update_distribution(
                    pt_obs=pt_obs,
                    ct_obs=ct_obs,
                    et_obs=mb_et_mix,
                    it_obs=mb_it_mix,
                    lidar_ground_points=mb_ground,
                    lidar_forward_points=mb_forward,
                    hidden_state=None,
                )
                new_logp = self.policy.get_actions_log_prob(mb_actions)
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                values, _ = self.policy.evaluate(
                    pt_obs=pt_obs,
                    ct_obs=ct_obs,
                    et_obs=mb_et_mix,
                    it_obs=mb_it_mix,
                    lidar_ground_points=mb_ground,
                    lidar_forward_points=mb_forward,
                    hidden_state=None,
                )
                value_loss = torch.mean((values.squeeze(-1) - mb_returns) ** 2)
                entropy = self.policy.entropy.mean()

                if int(self.estimator.lidar_embed_dim) > 0:
                    lidar_out = self.policy.lidar_encoder(mb_ground, mb_forward, pool="mean")
                    lidar_embed = torch.cat([lidar_out["ground_embed"], lidar_out["forward_embed"]], dim=-1)
                    if int(lidar_embed.shape[-1]) != int(self.estimator.lidar_embed_dim):
                        raise RuntimeError(
                            "StateEstimator lidar_embed_dim mismatch during update: "
                            f"expected {int(self.estimator.lidar_embed_dim)}, got {int(lidar_embed.shape[-1])}."
                        )
                else:
                    lidar_embed = None
                recon_loss, logs = self.estimator.compute_loss(
                    policy_obs=mb_policy,
                    et_target=mb_et,
                    it_target=mb_it,
                    lidar_embed=lidar_embed,
                    et_weight=self.cfg.et_loss_weight,
                    it_weight=self.cfg.it_loss_weight,
                    hidden_state=None,
                    update_internal_state=False,
                )
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                    + recon_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.estimator.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                metrics["loss"] += float(loss.detach())
                metrics["policy_loss"] += float(policy_loss.detach())
                metrics["value_loss"] += float(value_loss.detach())
                metrics["entropy"] += float(entropy.detach())
                metrics["recon_loss"] += float(recon_loss.detach())
                metrics["et_mse"] += float(logs.get("et_mse", torch.tensor(0.0, device=loss.device)).detach())
                metrics["it_mse"] += float(logs.get("it_mse", torch.tensor(0.0, device=loss.device)).detach())
                updates += 1

        for k in metrics:
            metrics[k] /= max(updates, 1)
        return metrics

    def train_iteration(self, obs_dict: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
        rollout, extras = self.collect_rollout(obs_dict)
        metrics = self._update_joint(rollout)

        self.iteration += 1
        advanced = self.stage_manager.maybe_advance(self.iteration, metrics=metrics)
        if advanced:
            self.stage_manager.apply_stage(
                {
                    "actor_critic": self.policy,
                    "state_estimator": self.estimator,
                    "contact_encoder": self.policy.contact_encoder,
                    "lidar_encoder": self.policy.lidar_encoder,
                }
            )

        out = {
            **metrics,
            "iteration": float(self.iteration),
            "stage": self.stage_manager.stage.value,
            "p_mix": float(extras["p_mix"]),
        }
        return out, extras["last_obs"]
