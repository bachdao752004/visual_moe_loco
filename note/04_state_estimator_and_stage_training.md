# 04 - State Estimator, PAS, Stage 1/2 Training

Relevant files:
- `networks/policy/state_estimator.py`
- `networks/trainer/stage_manager.py`
- `networks/trainer/trainer_phase_1.py`
- `networks/trainer/trainer_phase_2.py`
- `scripts/rsl_rl/train_moe.py`

## 1) StateEstimator

Current `StateEstimator`:
- Backbone: LSTM.
- Head: MLP (`[256, 128]` default).
- Inputs:
  - `policy_obs = [p_t, c_t]`
  - optional `lidar_embed` (pooled LiDAR features)
- Outputs:
  - `e_hat`
  - `i_hat`

## 2) Stage 1 (Oracle + reconstruction)

`Phase1Trainer` behavior:
- Collects rollout using oracle privileged channels.
- PPO losses:
  - surrogate policy loss,
  - value loss,
  - entropy regularization.
- Optional reconstruction loss (if estimator is enabled):
  - estimator predicts `(e_t, i_t)` from `(p_t, c_t, lidar_embed)`.

Overall:
- `L = L_ppo + recon_coef * L_recon`

## 3) Stage 2 (PAS + joint update)

`Phase2Trainer` behavior:
- Estimator predicts `(e_hat, i_hat)`.
- PAS blending mixes oracle and estimated privileged channels using a ramped probability `p_mix`.
- PPO is computed on the mixed privileged inputs.
- Policy and estimator are updated jointly in the same optimization loop.

Overall:
- `L = L_surro + value_coef * L_value - entropy_coef * H + L_recon`

## 4) StageManager

Responsibilities:
- Stage state control (`stage1_oracle`, `stage2_estimator`, `stage3_safe`).
- Module freeze/unfreeze according to stage config.
- Checkpoint save/load including stage + model + optional estimator state.

## 5) Runtime split robustness

In `train_moe.py`, `e_t/i_t` split is inferred from runtime critic dimension:
- avoids brittle hardcoding when contact/body regex changes,
- trainer config and policy are initialized from inferred split values.
