# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

try:
    from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg
except ImportError:
    # Older isaaclab_rl builds may not expose a dedicated recurrent cfg class.
    # Keep config importable by falling back to the base actor-critic cfg.
    RslRlPpoActorCriticRecurrentCfg = RslRlPpoActorCriticCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Default PPO runner (kept backward compatible)."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False
    # NOTE:
    # - For built-in rsl_rl policies, class_name is used directly.
    # - For custom local policies (e.g. ActorCriticMoE), class_name only takes effect when
    #   a custom trainer/script maps it to the actual Python class.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MLPPPORunnerCfg(BasePPORunnerCfg):
    """B1 baseline: plain MLP actor-critic (no RNN, no MoE, no LiDAR branch)."""

    experiment_name = "go2_mlp"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )


@configclass
class LSTMPPORunnerCfg(BasePPORunnerCfg):
    """B2 baseline: recurrent policy core (LSTM), no MoE."""

    experiment_name = "go2_lstm"
    # NOTE: `class_name="ActorCriticRNN"` requires custom trainer wiring.
    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRNN",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )


@configclass
class MoEPPORunnerCfg(BasePPORunnerCfg):
    """B4 main: MoE + LSTM + ContactEncoder + LiDAR (wired by custom trainer/script)."""

    experiment_name = "go2_moe"
    # NOTE: `class_name="ActorCriticMoE"` requires custom trainer wiring.
    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticMoE",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
