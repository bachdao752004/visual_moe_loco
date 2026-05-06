"""Custom training entrypoint for local MLP/LSTM/MoE policies.

This script bypasses rsl_rl's default policy factory and uses local classes under
``unitree_rl_lab.networks.policy`` with custom phase trainers.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import inspect
import os
import pathlib
import platform
import shutil
import sys
from datetime import datetime

import gymnasium as gym

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argcomplete
from packaging import version
import torch
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# argparse
parser = argparse.ArgumentParser(description="Train custom MoE/LSTM policies.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="Trainer phase: 1 or 2.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory override.")
parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint save interval.")
parser.add_argument("--resume_path", type=str, default=None, help="Explicit checkpoint path to resume from.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Video length (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Video interval (steps).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.networks.policy import ActorCritic, ActorCriticMoE, ActorCriticRNN, StateEstimator
from unitree_rl_lab.networks.trainer import (
    Phase1Trainer,
    Phase1TrainerCfg,
    Phase2Trainer,
    Phase2TrainerCfg,
    StageManager,
    TrainingStage,
)
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if getattr(args_cli, "distributed", False) and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    raise SystemExit(1)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _policy_class_from_name(class_name: str):
    mapping = {
        "ActorCritic": ActorCritic,
        "ActorCriticRNN": ActorCriticRNN,
        "ActorCriticMoE": ActorCriticMoE,
    }
    if class_name not in mapping:
        raise ValueError(f"Unsupported policy.class_name='{class_name}'. Supported: {list(mapping.keys())}")
    return mapping[class_name]


def _extract_obs_dims(obs: dict) -> tuple[int, int]:
    policy_dim = int(obs["policy"].shape[-1])
    critic_dim = int(obs["critic"].shape[-1])
    return policy_dim, critic_dim


def _safe_env_reset(env):
    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) >= 1:
        return reset_out[0]
    return reset_out


def _infer_privileged_split(num_ct_obs: int, critic_dim: int) -> tuple[int, int]:
    """Infer (e_t, i_t) split from current critic layout.

    Current e_t layout:
    base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(num_ct_obs)
    + joint_pos_rel(12) + joint_vel_rel(12) + joint_effort(12) + last_action(12)
    """
    num_et_obs = 3 + 3 + 3 + num_ct_obs + 12 + 12 + 12 + 12
    num_it_obs = critic_dim - num_et_obs
    if num_it_obs <= 0:
        raise RuntimeError(
            f"Invalid privileged split inferred from critic_obs: critic_dim={critic_dim}, "
            f"num_et_obs={num_et_obs}, num_it_obs={num_it_obs}."
        )
    return num_et_obs, num_it_obs


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if getattr(args_cli, "distributed", False):
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root}")
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {run_name}")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    class_name = getattr(agent_cfg.policy, "class_name", "ActorCriticMoE")
    policy_class = _policy_class_from_name(class_name)
    obs = _safe_env_reset(env)
    obs_dim_policy, obs_dim_critic = _extract_obs_dims(obs)
    num_actions = int(env.unwrapped.action_manager.action.shape[-1])
    num_pt_obs = 42
    num_ct_obs = 6
    num_et_obs, num_it_obs = _infer_privileged_split(num_ct_obs=num_ct_obs, critic_dim=obs_dim_critic)

    if policy_class is not ActorCriticMoE:
        raise NotImplementedError(
            "train_moe.py currently wires custom phase trainers for ActorCriticMoE only. "
            "For MLP/LSTM baselines, use scripts/rsl_rl/train.py or add dedicated trainer wiring."
        )

    # MoE split defaults aligned with current Go2 configs.
    policy = ActorCriticMoE(
        num_pt_obs=num_pt_obs,
        num_ct_obs=num_ct_obs,
        num_et_obs=num_et_obs,
        num_it_obs=num_it_obs,
        num_actions=num_actions,
        init_noise_std=agent_cfg.policy.init_noise_std,
    )

    estimator: StateEstimator | None = None
    if args_cli.phase == 1:
        stage_manager = StageManager(initial_stage=TrainingStage.STAGE1_ORACLE)
        trainer_cfg = Phase1TrainerCfg(
            device=env_cfg.sim.device,
            rollout_steps=24,
            learning_rate=1.0e-3,
            num_pt_obs=num_pt_obs,
            num_ct_obs=num_ct_obs,
            num_et_obs=num_et_obs,
            num_it_obs=num_it_obs,
        )
        estimator = StateEstimator(
            num_policy_obs=trainer_cfg.num_pt_obs + trainer_cfg.num_ct_obs,
            num_et_obs=trainer_cfg.num_et_obs,
            num_it_obs=trainer_cfg.num_it_obs,
            lidar_embed_dim=512,
        )
        trainer = Phase1Trainer(
            env=env,
            policy=policy,
            cfg=trainer_cfg,
            estimator=estimator,
            stage_manager=stage_manager,
        )
    else:
        stage_manager = StageManager(initial_stage=TrainingStage.STAGE2_ESTIMATOR)
        trainer_cfg = Phase2TrainerCfg(
            device=env_cfg.sim.device,
            rollout_steps=24,
            learning_rate=1.0e-3,
            num_pt_obs=num_pt_obs,
            num_ct_obs=num_ct_obs,
            num_et_obs=num_et_obs,
            num_it_obs=num_it_obs,
        )
        estimator = StateEstimator(
            num_policy_obs=trainer_cfg.num_pt_obs + trainer_cfg.num_ct_obs,
            num_et_obs=trainer_cfg.num_et_obs,
            num_it_obs=trainer_cfg.num_it_obs,
            lidar_embed_dim=512,
        )
        trainer = Phase2Trainer(env=env, policy=policy, estimator=estimator, cfg=trainer_cfg, stage_manager=stage_manager)

    expected_policy = trainer_cfg.num_pt_obs + trainer_cfg.num_ct_obs
    expected_critic = trainer_cfg.et_start + trainer_cfg.num_et_obs + trainer_cfg.num_it_obs
    assert obs_dim_policy == expected_policy, (
        f"Policy obs dim mismatch: env={obs_dim_policy}, cfg={expected_policy}. "
        "Check num_pt_obs + num_ct_obs in trainer cfg."
    )
    assert obs_dim_critic >= expected_critic, (
        f"Critic obs dim too small: env={obs_dim_critic}, cfg needs >={expected_critic}."
    )

    # resume
    start_iter = 0
    ckpt_dir = args_cli.checkpoint_dir or os.path.join(log_dir, "checkpoints")
    resume_path = None
    if args_cli.resume or args_cli.resume_path:
        if args_cli.resume_path:
            resume_path = args_cli.resume_path
        else:
            resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    if resume_path:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        start_iter = stage_manager.load_checkpoint(
            resume_path,
            model=policy,
            estimator=estimator,
        )

    # dump configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    max_iters = int(agent_cfg.max_iterations)
    for it in range(start_iter, max_iters):
        metrics, obs = trainer.train_iteration(obs)
        if (it + 1) % 10 == 0:
            print(
                f"[Iter {it+1:06d}] stage={metrics['stage']} "
                f"loss={metrics.get('loss', 0.0):.4f}"
            )
        if (it + 1) % int(args_cli.save_interval) == 0:
            stage_manager.save_checkpoint(
                ckpt_dir,
                model=policy,
                estimator=estimator,
                iteration=it + 1,
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
