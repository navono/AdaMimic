# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AdaMimic is a PyTorch/Isaac Gym reinforcement learning framework for training Unitree G1 humanoid robots (27 DoF) to perform dynamic sports motions via adaptive motion tracking. It implements a two-stage training pipeline: **Stage 1** learns precise motion mimicry, **Stage 2** fine-tunes for robustness against temporal variations. Paper: "Towards Adaptable Humanoid Control via Adaptive Motion Tracking" (arXiv 2510.14454).

## Common Commands

All commands require `conda activate adamimic` and `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` (the Makefile handles both automatically).

### Training & Evaluation (via Makefile — preferred)

```bash
make stage1 TASK=badminton_hit                    # Stage 1 training
make stage1 TASK=badminton_hit RESUME=<ckpt>      # Resume stage 1
make stage2 TASK=badminton_hit CHECKPOINT=<s1_ckpt>  # Stage 2 (requires stage1 ckpt)
make play TASK=badminton_hit RESUME=<s2_ckpt>     # Evaluate trained policy
make tensorboard TASK=badminton_hit               # Launch tensorboard on port 6006
make logs TASK=badminton_hit                      # Tail training log
```

Available tasks: `badminton_hit`, `tennis_hit`, `high_jump`, `far_jump`, `triple_jump`, `jump_step_up`, `jump_step_down`.

### Direct Python Commands

The Makefile targets wrap these. Note: README omits the required `+robot=g1_dof27` parameter — always include it.

```bash
# Train stage 1
python legged_gym/legged_gym/scripts/train.py +dataset=g1_dof27/<task> +algorithm=adamimic/stage1 +robot=g1_dof27

# Train stage 2
python legged_gym/legged_gym/scripts/train.py +dataset=g1_dof27/<task> +algorithm=adamimic/stage2 +robot=g1_dof27 checkpoint_path=<stage1_ckpt>

# Play
python legged_gym/legged_gym/scripts/play.py +dataset=g1_dof27/<task> +algorithm=adamimic/stage2 +robot=g1_dof27 resume_path=<stage2_ckpt>
```

### Baselines

```bash
python legged_gym/legged_gym/scripts/train.py +dataset=g1_dof27/<task> +algorithm=<baseline> +robot=g1_dof27
```

Baseline configs: `adamimic`, `amp`, `deepmimic` (under `legged_gym/legged_gym/configs/algorithm/`).

## Architecture

### Two Subpackages (both installed editable via `pip install -e .`)

- **`legged_gym/`** — Isaac Gym simulation environment, task/robot configs, training & eval entry points
- **`rsl_rl/`** — PPO algorithm implementation with motion-tracking extensions

### Configuration System (Hydra)

Configs compose from four groups specified on the CLI:

```
+algorithm=adamimic/stage1   → legged_gym/legged_gym/configs/algorithm/adamimic/stage1.yaml
+dataset=g1_dof27/<task>     → legged_gym/legged_gym/configs/dataset/g1_dof27/<task>.yaml
+robot=g1_dof27              → legged_gym/legged_gym/configs/robot/g1_dof27.yaml
```

Base configs: `train.yaml` (training defaults, 4096 envs, headless), `eval.yaml` (evaluation, 10 envs, GUI). Hydra output dir determines checkpoint/log path: `exp/<robot>/<task>/<stage>/<timestamp>/`.

### Key Source Files

| File | Role |
|------|------|
| `legged_gym/legged_gym/scripts/train.py` | Hydra entry point → `task_registry.make_env_hydra()` → `make_alg_runner_hydra()` → `ppo_runner.learn()` |
| `legged_gym/legged_gym/scripts/play.py` | Evaluation + JIT/ONNX export of trained policies |
| `legged_gym/legged_gym/envs/base/motion_tracking.py` | Core Isaac Gym environment — robot physics, motion data loading, reward computation |
| `legged_gym/legged_gym/envs/base/legged_robot.py` | Base robot class |
| `rsl_rl/rsl_rl/runners/track_on_policy_runner.py` | Training loop orchestration |
| `rsl_rl/rsl_rl/algorithms/track_ppo.py` | PPO with motion-tracking extensions |
| `rsl_rl/rsl_rl/modules/track_actor_critic.py` | Actor-critic with keyframe time inference and dual-policy (low-level frozen / high-level trainable) |
| `rsl_rl/rsl_rl/storage/track_rollout_storage.py` | Rollout buffer for motion tracking data |

### Two-Stage Pipeline Design

**Stage 1** (`TrackActorCritic`, 40k iterations): Full network trains to replicate reference motions. No time variation (`actor_time_scale_range: [0, 0]`). Uses curriculum learning on termination thresholds and reward limits.

**Stage 2** (`TrackActorCriticDelta`, 10k iterations): Low-level policy frozen (`freeze: true`), only high-level policy trains. Introduces temporal perturbation (`actor_time_scale_range: [-0.015, 0.02]`) to learn adaptive time adjustments. Loads stage 1 checkpoint via `checkpoint_path`.

### Motion Data

Motion datasets live in `legged_gym/resources/dataset/g1_dof27_data/`. The `MotionLib` class (in `legged_gym/legged_gym/utils/motionlib.py`) loads and serves keyframe trajectories during training.

## Environment Setup Notes

- **Python**: Must be 3.8 (Isaac Gym has precompiled `.so` for 3.6/3.7/3.8 only)
- **Isaac Gym**: Modify `isaacgym/python/setup.py` — change `python_requires` from `>=3.6,<3.9` to `>=3.6,<3.11` to bypass pip version check
- **numpy**: Must be `<1.24` (Isaac Gym uses deprecated `np.float`). `conda_env.yml` pins `numpy=1.20.3`
- **LD_LIBRARY_PATH**: Must include `$CONDA_PREFIX/lib` for `libpython3.8.so` at runtime
- **License**: CC BY-NC-SA 4.0 — commercial use not allowed without authorization
