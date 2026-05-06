# 01 - Environment Setup, Observations, Terrain

Primary files:
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/observations.py`

## 1) Terrain setup

`COBBLESTONE_ROAD_CFG` uses a multi-terrain generator with mixed sub-terrains:
- Heightfield: rough, wave, slope, inverted slope, rough_slope, steep slope, stepping stones, discrete obstacles.
- Mesh: stairs, rails, star, floating ring, repeated primitives, gaps, pits, random grid obstacles.

Why this matters:
- Improves robustness by exposing the policy to diverse terrain classes.
- Terrain curriculum is enabled through `CurriculumCfg.terrain_levels`.

## 2) Commands (`c_t`)

`CommandsCfg.base_velocity` currently includes:
- `lin_vel_x`
- `lin_vel_y`
- `ang_vel_z`
- `body_height`
- `swing_height`
- `body_pitch`

So current command dimension is **6**.

## 3) Observation groups

### Policy group (`p_t + c_t`)

`ObservationsCfg.PolicyCfg`:
- `base_ang_vel` (3)
- `projected_gravity` (3)
- `joint_pos_rel` (12)
- `joint_vel_rel` (12)
- `last_action` (12)
- `velocity_commands` (6)

Totals:
- `p_t = 42` (without command channels)
- `c_t = 6`
- policy observation concat = **48**

### Critic group (privileged stream)

`ObservationsCfg.CriticCfg` in order:
- `base_lin_vel` (3)
- `base_ang_vel` (3)
- `projected_gravity` (3)
- `velocity_commands` (6)
- `joint_pos_rel` (12)
- `joint_vel_rel` (12)
- `joint_effort` (12)
- `last_action` (12)
- `foot_contact_bool` (4)
- `contact_forces_norm` (dynamic; currently 17 for this robot body selection)

Current critic total is **84**.

In `train_moe.py`:
- `e_t` uses the explicit privileged core block.
- `i_t` uses the contact-related block.
- Split is inferred at runtime from critic dimension to reduce hardcoded mismatch risk.

### LiDAR group (`u_t`)

Two raycasters:
- `lidar_ground` (12 rays, downward geometry)
- `lidar_forward` (16 rays, forward geometry)

`mdp.lidar_3d_scan` returns base-relative normalized hit points.

## 4) Events / Domain randomization

Main startup randomization:
- Friction range `(0.0, 2.0)`
- Base mass add `(1.0, 3.0)`
- Link mass scale `(0.9, 1.1)`
- PD gains scale `(0.9, 1.1)`
- Base COM randomization on x/y/z
- Joint default position random add `(-0.1, 0.1)`

## 5) Summary

Current environment provides:
- Rich terrain diversity + curriculum,
- 6D command interface,
- Separate policy / critic / LiDAR streams,
- Domain randomization at startup/reset/interval.

This gives the MoE/LSTM pipeline enough signal for robust agile locomotion training.
