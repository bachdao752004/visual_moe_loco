# Isaac Lab Go2 Agile Notes

This note set summarizes the current `unitree_rl_lab` source code in reading order.

## Recommended Reading Order

1. `01_env_setup_obs_terrain.md`  
   Environment setup: terrain, commands, observations, and randomization.
2. `02_rewards_and_curriculum.md`  
   Reward terms and curriculum design.
3. `03_policy_architectures_mlp_rnn_moe.md`  
   Policy architecture breakdown (MLP / RNN / MoE).
4. `04_state_estimator_and_stage_training.md`  
   State estimator, PAS, and Stage 1/2 trainers.
5. `05_train_and_run_workflow.md`  
   Task IDs, train/play/export commands.
6. `06_dim_map_cheatsheet.md`  
   Fast dimension reference for `p_t`, `c_t`, `e_t`, `i_t`, and LiDAR streams.
   Includes the most common shape-mismatch debug checks.

## Current Scope

- Active focus: Go2 agile locomotion.
- Main custom pipeline: `scripts/rsl_rl/train_moe.py` + `ActorCriticMoE` + `StateEstimator`.
- Baselines: MLP/RNN through `scripts/rsl_rl/train.py`.
