# Unitree RL Lab (Project Scope Note)

## Active Scope (this fork)

- Primary active target: Go2 agile locomotion in Isaac Lab.
- Active training entrypoint for current work: `scripts/rsl_rl/train_moe.py`.
- Active policy pipeline: MoE + LSTM + privileged estimator (Stage 1/2).

## Inactive/Upstream Kept Modules

The following parts are kept mostly from upstream for compatibility/reference, but are **not active focus** in this fork:

- G1 locomotion tasks
- H1 locomotion tasks
- Mimic tasks and related conversion/replay scripts
- Deployment folders for non-Go2 workflows

## Quick Start

1. Install Isaac Lab (matching your local setup).
2. Install this package in editable mode:

```bash
./unitree_rl_lab.sh -i
```

3. List available tasks:

```bash
./unitree_rl_lab.sh -l
```

4. Train Go2 variants:

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Agile-MLP
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Agile-RNN
```

5. Train Go2 MoE pipeline (custom stage trainers):

```bash
python scripts/rsl_rl/train_moe.py --headless --task Unitree-Go2-Agile-MoE --phase 1
python scripts/rsl_rl/train_moe.py --headless --task Unitree-Go2-Agile-MoE --phase 2
```

## Notes

- If you are extending this fork, keep changes aligned with Go2 Agile unless explicitly enabling other robot families.
- For legacy/upstream flows, verify task registration and runner config before use.
- Task mapping:
  - `Unitree-Go2-Agile-MLP` -> `MLPPPORunnerCfg` (`train.py`)
  - `Unitree-Go2-Agile-RNN` -> `LSTMPPORunnerCfg` (`train.py`)
  - `Unitree-Go2-Agile-MoE` -> `MoEPPORunnerCfg` (`train.py`) or custom phase flow (`train_moe.py`)
