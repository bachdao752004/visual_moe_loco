# 06 - Dimension Map Cheatsheet

Quick lookup for observation split debugging (`shape mismatch`, wrong slicing, bad cfg values).

## 1) High-level groups

- `p_t` (policy proprio) = 42
- `c_t` (commands) = 6
- policy input total (`p_t + c_t`) = 48
- critic total (current layout) = 84
- LiDAR streams: ground rays + forward rays (tokenized in encoder)

## 2) Policy observation map (`p_t + c_t`)

| Term | Dim | Group |
|---|---:|---|
| `base_ang_vel` | 3 | `p_t` |
| `projected_gravity` | 3 | `p_t` |
| `joint_pos_rel` | 12 | `p_t` |
| `joint_vel_rel` | 12 | `p_t` |
| `last_action` | 12 | `p_t` |
| `velocity_commands` (`lin_x, lin_y, yaw, body_height, swing_height, body_pitch`) | 6 | `c_t` |

## 3) Critic observation map (privileged stream)

Current critic order in `CriticCfg`:

| Term | Dim | Group |
|---|---:|---|
| `base_lin_vel` | 3 | `e_t` |
| `base_ang_vel` | 3 | `e_t` |
| `projected_gravity` | 3 | `e_t` |
| `velocity_commands` | 6 | `e_t` |
| `joint_pos_rel` | 12 | `e_t` |
| `joint_vel_rel` | 12 | `e_t` |
| `joint_effort` | 12 | `e_t` |
| `last_action` | 12 | `e_t` |
| `foot_contact_bool` | 4 | `i_t` |
| `contact_forces_norm(body_names=".*")` | dynamic (currently 17) | `i_t` |

With current robot body selection:
- `e_t = 63`
- `i_t = 21` (`4 + 17`)
- `63 + 21 = 84` (matches critic total)

## 4) LiDAR (`u_t`) channels

| Sensor | Rays | Per-ray values |
|---|---:|---|
| `lidar_ground` | 12 | 3D point (`x,y,z`) |
| `lidar_forward` | 16 | 3D point (`x,y,z`) |

Encoded by `LidarEncoder` into tokens and pooled embeddings, then consumed by RNN/MoE pipelines.

## 5) Runtime split logic and safety checks

`train_moe.py` infers privileged split from runtime critic dimension:
- keeps `e_t` fixed to known explicit block size from layout,
- computes `i_t` from residual contact block,
- initializes policy/trainer/estimator with inferred values.

Recommended checks when changing robot asset/body regex:
- verify `contact_forces_norm` output size changed as expected,
- confirm inferred `num_it_obs` matches runtime,
- keep runtime asserts enabled to fail fast on mismatch.
