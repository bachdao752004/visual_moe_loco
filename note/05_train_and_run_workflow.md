# 05 - Train / Run / Export Workflow

Relevant files:
- `tasks/locomotion/robots/go2/__init__.py`
- `tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
- `scripts/rsl_rl/train.py`
- `scripts/rsl_rl/train_moe.py`
- `scripts/rsl_rl/play.py`
- `utils/export_deploy_cfg.py`

## 1) Task IDs for fast architecture switching

Go2 is split into three task IDs:
- `Unitree-Go2-Agile-MLP`
- `Unitree-Go2-Agile-RNN`
- `Unitree-Go2-Agile-MoE`

## 2) Training commands

### Baselines (standard runner)
```bash
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Agile-MLP
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Agile-RNN
```

### MoE staged pipeline
```bash
python scripts/rsl_rl/train_moe.py --headless --task Unitree-Go2-Agile-MoE --phase 1
python scripts/rsl_rl/train_moe.py --headless --task Unitree-Go2-Agile-MoE --phase 2 --resume <phase1_ckpt>
```

## 3) Resume and checkpoints

`train_moe.py`:
- saves with `StageManager.save_checkpoint(...)`,
- loads with `StageManager.load_checkpoint(...)`,
- persists both policy and estimator states.

## 4) Deployment exports

`play.py` can export:
- `policy.pt` (TorchScript)
- `policy.onnx`

`export_deploy_cfg.py` writes `deploy.yaml` (obs/action/command mapping) into `logs/.../params`.

## 5) Recommended practical sequence

1. Run MLP baseline to validate environment and reward stability.
2. Run RNN baseline to measure temporal gains.
3. Run MoE Stage 1.
4. Run MoE Stage 2 (PAS + estimator joint learning).
5. Run play/eval and export deployment artifacts.

## 6) As-Implemented Pipeline Notes (Important)

- Environment policy observation is **48D total**:
  - `p_t = 42`
  - `c_t = 6`
  - Trainer splits `obs["policy"]` into `(pt_obs, ct_obs)`.
- Critic observation is currently **84D total**:
  - `e_t = 63`
  - `i_t = 21` (depends on contact body selection).
- Typical MoE training flow in this repo is **two explicit runs**:
  - Stage 1: `train_moe.py --phase 1`
  - Stage 2: `train_moe.py --phase 2 --resume <phase1_ckpt>`
- `StageManager.maybe_advance()` exists, but practical usage is still phase-driven via CLI.
- Stage 3 (`safe_expert`, `cbf_filter`) is currently a placeholder and not fully implemented.
