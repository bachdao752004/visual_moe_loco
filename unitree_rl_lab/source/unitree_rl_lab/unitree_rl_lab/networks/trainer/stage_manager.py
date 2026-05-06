from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class TrainingStage(str, Enum):
    """Canonical stage names for multi-phase training."""

    STAGE1_ORACLE = "stage1_oracle"
    STAGE2_ESTIMATOR = "stage2_estimator"
    STAGE3_SAFE = "stage3_safe"


@dataclass
class StageTransitionCfg:
    """Transition condition from one stage to the next."""

    min_iteration: int = 0
    metric_name: str | None = None
    metric_threshold: float | None = None

    def is_satisfied(self, iteration: int, metrics: dict[str, float] | None = None) -> bool:
        if iteration < self.min_iteration:
            return False
        if self.metric_name is None:
            return True
        if metrics is None or self.metric_name not in metrics:
            return False
        if self.metric_threshold is None:
            return True
        return float(metrics[self.metric_name]) >= float(self.metric_threshold)


@dataclass
class StageCfg:
    """Per-stage runtime behavior."""

    freeze_modules: list[str] = field(default_factory=list)
    unfreeze_modules: list[str] = field(default_factory=list)
    use_oracle_privileged: bool = True
    estimator_mix_prob: float = 0.0
    use_cbf_reward_penalty: bool = False
    notes: str = ""


class StageManager:
    """State machine for phase-based training (Stage 1/2/3).

    This manager is intentionally framework-agnostic so both trainer_phase_1.py and
    trainer_phase_2.py can reuse it with minimal coupling.
    """

    def __init__(
        self,
        initial_stage: TrainingStage = TrainingStage.STAGE1_ORACLE,
        stage_cfgs: dict[TrainingStage, StageCfg] | None = None,
        transitions: dict[TrainingStage, StageTransitionCfg] | None = None,
    ) -> None:
        self._stage = initial_stage
        self._stage_cfgs = stage_cfgs or {
            TrainingStage.STAGE1_ORACLE: StageCfg(
                freeze_modules=[],
                unfreeze_modules=["actor_critic", "contact_encoder", "lidar_encoder"],
                use_oracle_privileged=True,
                estimator_mix_prob=0.0,
                use_cbf_reward_penalty=False,
                notes="Train nominal locomotion with oracle privileged signals.",
            ),
            TrainingStage.STAGE2_ESTIMATOR: StageCfg(
                freeze_modules=[],
                # lidar_encoder and contact_encoder are intentionally unfrozen here so
                # the encoders can adapt to minimise the estimator reconstruction loss.
                # If you want frozen encoders, move them to freeze_modules.
                unfreeze_modules=["actor_critic", "state_estimator", "lidar_encoder", "contact_encoder"],
                use_oracle_privileged=False,
                estimator_mix_prob=1.0,
                use_cbf_reward_penalty=False,
                notes="Joint PPO and estimator training with PAS.",
            ),
            # NOTE: Stage 3 is a placeholder for future CBF safety fine-tuning.
            # "safe_expert" and "cbf_filter" are NOT yet implemented; StageManager
            # will emit a [WARN] and skip them harmlessly until they are registered.
            TrainingStage.STAGE3_SAFE: StageCfg(
                freeze_modules=["actor_critic", "state_estimator"],
                unfreeze_modules=["safe_expert", "cbf_filter"],
                use_oracle_privileged=False,
                estimator_mix_prob=1.0,
                use_cbf_reward_penalty=True,
                notes="Safety fine-tuning with CBF-oriented objectives (placeholder — not yet implemented).",
            ),
        }
        self._transitions = transitions or {
            TrainingStage.STAGE1_ORACLE: StageTransitionCfg(min_iteration=10000),
            TrainingStage.STAGE2_ESTIMATOR: StageTransitionCfg(min_iteration=20000),
            TrainingStage.STAGE3_SAFE: StageTransitionCfg(min_iteration=10**12),  # terminal stage
        }

    @property
    def stage(self) -> TrainingStage:
        return self._stage

    @property
    def stage_cfg(self) -> StageCfg:
        return self._stage_cfgs[self._stage]

    def maybe_advance(self, iteration: int, metrics: dict[str, float] | None = None) -> bool:
        """Advance to next stage if transition condition is satisfied."""
        transition_cfg = self._transitions[self._stage]
        if not transition_cfg.is_satisfied(iteration=iteration, metrics=metrics):
            return False

        if self._stage == TrainingStage.STAGE1_ORACLE:
            self._stage = TrainingStage.STAGE2_ESTIMATOR
            return True
        if self._stage == TrainingStage.STAGE2_ESTIMATOR:
            self._stage = TrainingStage.STAGE3_SAFE
            return True
        return False

    def apply_stage(self, module_registry: dict[str, nn.Module]) -> None:
        """Apply freeze/unfreeze plan to model submodules.

        Args:
            module_registry: Mapping from symbolic module names to torch modules.
                Example keys: "actor_critic", "state_estimator", "cbf_filter".
        """
        cfg = self.stage_cfg
        for name in cfg.freeze_modules:
            module = module_registry.get(name)
            if module is None:
                print(f"[WARN] StageManager: freeze target '{name}' not found in registry, skipping.")
                continue
            self._set_requires_grad(module, False)
        for name in cfg.unfreeze_modules:
            module = module_registry.get(name)
            if module is None:
                print(f"[WARN] StageManager: unfreeze target '{name}' not found in registry, skipping.")
                continue
            self._set_requires_grad(module, True)

    def get_runtime_flags(self) -> dict[str, Any]:
        """Return lightweight runtime flags trainer can consume directly."""
        cfg = self.stage_cfg
        return {
            "stage": self.stage.value,
            "use_oracle_privileged": cfg.use_oracle_privileged,
            "estimator_mix_prob": cfg.estimator_mix_prob,
            "use_cbf_reward_penalty": cfg.use_cbf_reward_penalty,
            "notes": cfg.notes,
        }

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        iteration: int = 0,
        estimator: nn.Module | None = None,
    ) -> str:
        """Save model/optimizer and current stage into a stage-scoped checkpoint file."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"{self._stage.value}_iter{iteration}.pt")
        payload: dict[str, Any] = {
            "stage": self._stage.value,
            "iteration": int(iteration),
            "model_state": model.state_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if estimator is not None:
            payload["estimator_state"] = estimator.state_dict()
        torch.save(payload, path)
        return path

    def load_checkpoint(
        self,
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        estimator: nn.Module | None = None,
        map_location: str = "cpu",
    ) -> int:
        """Load checkpoint and restore stage; return the loaded training iteration."""
        payload = torch.load(path, map_location=map_location)
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if estimator is not None and "estimator_state" in payload:
            estimator.load_state_dict(payload["estimator_state"])
        self._stage = TrainingStage(payload["stage"])
        return int(payload.get("iteration", 0))

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str, stage: TrainingStage) -> str | None:
        """Find latest checkpoint path for a given stage by iteration suffix."""
        if not os.path.exists(checkpoint_dir):
            return None
        files = [f for f in os.listdir(checkpoint_dir) if f.startswith(stage.value) and f.endswith(".pt")]
        if not files:
            return None
        files.sort(key=lambda f: int(f.split("iter")[-1].replace(".pt", "")))
        return os.path.join(checkpoint_dir, files[-1])

    @staticmethod
    def _set_requires_grad(module: nn.Module, value: bool) -> None:
        for param in module.parameters():
            param.requires_grad = value
