# AdaMimic 两阶段训练原理

## Stage 1：运动模仿（Motion Mimicry）

**目标**：让机器人学会精确复现参考运动轨迹（如羽毛球挥拍动作）。

**核心方法**：
- **AMP 判别器**（Adversarial Motion Priors）：一个二分类网络，区分"策略产生的动作"和"参考数据集中的动作"。判别器输出奖励信号，驱动策略生成的运动风格趋近参考运动
- **任务奖励**：针对具体任务的奖励（如击球位置、拍面朝向等）
- **总奖励 = AMP 奖励 + 任务奖励**

**关键配置**：
| 参数 | 值 | 含义 |
|------|-----|------|
| `actor_time_scale_range` | `[0, 0]` | 无时间扰动，策略严格按参考运动的时间节奏执行 |
| `freeze` | `false` | 所有网络正常训练 |
| `train_high` | `false` | 不训练高层策略 |
| `max_iterations` | 40000 | 最大迭代次数 |
| `save_interval` | 500 | 每 500 次迭代保存 checkpoint |

## Stage 2：鲁棒性微调（Robustness Fine-tuning）

**目标**：在 stage1 的基础上，提升策略对时间偏移和动态变化的鲁棒性。

**关键变化**：
| 参数 | Stage 1 | Stage 2 | 含义 |
|------|---------|---------|------|
| `freeze` | false | true | 冻结 stage1 学到的低层网络，防止微调时遗忘 |
| `actor_time_scale_range` | `[0, 0]` | `[-0.015, 0.02]` | 对动作的时间节奏加入随机扰动，迫使策略适应节奏变化 |
| `train_high` | false | true | 启用高层策略训练，在冻结的低层之上学习自适应调整 |
| `max_iterations` | 40000 | 10000 | 微调迭代较少 |
| `save_interval` | 500 | 250 | 更频繁地保存 checkpoint |

**直观理解**：
- Stage 1 = 学会"标准动作"（像模仿教练挥拍）
- Stage 2 = 学会"灵活应变"（面对不同球速、节奏偏差时仍能打出好球）

两个阶段缺一不可：没有 stage1 动作不标准，没有 stage2 动作不鲁棒。

## 训练流程

```
Stage 1 (40000 iters) → checkpoint → Stage 2 (10000 iters) → checkpoint → Play
```

Stage 2 必须加载 Stage 1 的 checkpoint 作为初始化权重（`checkpoint_path` 参数）。

## Checkpoint 位置

训练输出保存在 `exp/<robot>/<task>/<stage>/<timestamp>/` 目录下：
- `model_<iter>.pt` — PyTorch checkpoint
- `model_jit_<iter>.pt` — JIT 编译版本（用于部署）
- `events.out.tfevents.*` — Tensorboard 日志

## Tensorboard 指标说明

启动：`make tensorboard TASK=badminton_hit`，浏览器访问 `http://<机器IP>:6006`。

顶部导航栏切换 tab，主要看 **SCALARS**。左侧面板支持搜索/正则过滤指标（如 `rew_.*`）、勾选显示、调整曲线平滑度。

### 关键指标

| 指标 | 含义 | 参考标准 |
|------|------|----------|
| `Train/mean_reward` | 平均总奖励 | 越高越好 |
| `Train/mean_episode_length` | 平均 episode 长度 | 越高越好（说明机器人存活更久） |
| `Train/mean_entropy` | 策略熵 | 初期较高，后期逐渐降低为宜 |
| `Loss/loss` | 总损失 | 越低越好 |
| `Loss/value_loss` | 价值函数损失 | 越低越好 |
| `Loss/action_loss` | 策略损失 | 越低越好 |
| `Loss/surrogate_loss` | PPO surrogate 损失 | 越低越好 |

### 运动跟踪指标

| 指标 | 含义 | 参考标准 |
|------|------|----------|
| `rew_sparse_tracking_trunk_height` | 躯干高度跟踪奖励 | 越高越好 |
| `rew_sparse_tracking_*` | 各部位的稀疏跟踪奖励 | 越高越好 |
| `joint_pos_diff_norm` | 关节位置偏差 | 越低越好 |
| `upper_body_diff_norm` | 上半身偏差 | 越低越好 |
| `lower_body_diff_norm` | 下半身偏差 | 越低越好 |

训练正常时：`mean_reward` 逐步上升，`joint_pos_diff_norm` 逐步下降，`mean_episode_length` 逐步增长。