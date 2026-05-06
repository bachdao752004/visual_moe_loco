# 02 - Rewards and Curriculum

Relevant files:
- `tasks/locomotion/robots/go2/velocity_env_cfg.py` (`RewardsCfg`, `CurriculumCfg`)
- `tasks/locomotion/mdp/rewards.py`

## 1) Core rewards currently kept

Core quadruped rewards in use:
- Tracking:
  - `track_lin_vel_xy`
  - `track_ang_vel_z`
  - `termination`
  - `alive`
- Regularization:
  - `joint_pos`
  - `joint_vel`
  - `joint_acc`
  - `ang_vel_xy_stability`
  - `feet_air_time`
  - `front_hip_pos`
  - `rear_hip_pos`
  - `base_height`
  - `balance`
  - `joint_limits`
  - `torque_exceed`

## 2) Extra rewards currently commented out

Temporarily disabled (to stay closer to the paper core objective):
- `flat_orientation_l2`
- `energy`
- `action_rate`
- `undesired_contacts`
- `feet_slide`
- safe-loco / CBF group:
  - `vel_safe`
  - `step_penalty`
  - `collision_penalty`
  - `cbf_action_penalty`
  - `cbf_psi_penalty`

These can be re-enabled later for practical robustness tuning.

## 3) Reward function implementation (`mdp/rewards.py`)

This file contains custom reward utilities:
- stability/energy/torque/contact penalties,
- joint/hip-specific penalties,
- safe-loco and CBF helper terms,
- termination helper (`root_height_below_minimum`).

## 4) Curriculum

`CurriculumCfg` currently uses:
- `terrain_levels_vel`
- `lin_vel_cmd_levels`
- `ang_vel_cmd_levels`

Effect:
- terrain difficulty increases over training,
- command ranges expand over time,
- helps stabilize learning and improve robustness.
