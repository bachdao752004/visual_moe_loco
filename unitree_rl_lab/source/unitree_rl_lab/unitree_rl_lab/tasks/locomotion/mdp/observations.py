from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def foot_contact_bool(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Binary foot contact state per selected body id."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    return (contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()


def contact_forces_norm(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    normalize_by: float = 100.0,
) -> torch.Tensor:
    """Per-link contact force norm normalized for stable learning."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return torch.linalg.norm(forces, dim=-1) / max(normalize_by, 1e-6)


def lidar_3d_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    flatten: bool = False,
) -> torch.Tensor:
    """Return LiDAR hit points in robot base-relative coordinates."""
    lidar: RayCaster = env.scene[sensor_cfg.name]
    hits_w = lidar.data.ray_hits_w
    robot = env.scene["robot"]
    rel = (hits_w - robot.data.root_pos_w[:, None, :]) / 5.0
    rel = torch.nan_to_num(rel.clamp(-1.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)
    if flatten:
        return rel.reshape(rel.shape[0], -1)
    return rel
