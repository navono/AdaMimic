# IsaacLab 替换 IsaacGym — 可行性评估与迁移方案

> 评估日期：2026-06-05

## 一、结论

**迁移可行，但工程量大（估计 3-6 人周）。** AdaMimic 对 IsaacGym 的依赖很深，核心文件共 ~4900 行代码中包含 **~203 处 IsaacGym API 调用**。但好消息是：

1. IsaacLab 的 `DirectRLEnv` 工作流与 AdaMimic 当前架构几乎 1:1 对应
2. `rsl_rl` 库本身 **零 IsaacGym 依赖**，无需改动
3. Unitree G1 在 IsaacLab 中有官方 USD 资产和预置环境
4. NVIDIA 提供了官方迁移文档

## 二、背景：IsaacLab 与 IsaacGym 的关系

```
NVIDIA Omniverse  —  底层平台 (USD, RTX 渲染, PhysX)
  Isaac Sim        —  通用机器人仿真器 (闭源)
    Isaac Lab      —  机器人学习框架 (开源) <-- 替代 IsaacGym
```

- **IsaacGym** (Preview Release)：**已停止维护**，不再积极开发。
- **Isaac Sim**：通用机器人仿真器，现已吸收 IsaacGym 的全部物理仿真能力（GPU 加速 PhysX、Tensor API）。
- **Isaac Lab**：基于 Isaac Sim 的统一机器人学习框架，开源于 [github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)。

### 版本兼容性

| Isaac Lab 版本 | Isaac Sim 版本 | Python |
|---------------|---------------|--------|
| v2.0.x        | Isaac Sim 4.5 | 3.10   |
| v2.1.x        | Isaac Sim 4.5 | 3.10   |
| v2.2.x        | Isaac Sim 5.0 | 3.10   |
| v2.3.x        | Isaac Sim 5.0+| 3.10   |
| v3.0.0-beta   | Isaac Sim 6.0 | 3.11   |

## 三、API 对应关系

### 3.1 核心映射表

| IsaacGym | IsaacLab (DirectRLEnv) | 改动程度 |
|----------|----------------------|---------|
| `gymapi.acquire_gym()` | 自动创建 | 删除 |
| `gym.create_sim()` | `_setup_scene()` | 重写 |
| `gym.create_env()` + 循环创建 actor | `InteractiveSceneCfg` + `clone_environments()` | 重写 |
| `gymtorch.wrap_tensor()` / `unwrap_tensor()` | **不需要** — 直接返回 PyTorch tensor | 删除 |
| `gym.acquire_dof_state_tensor()` | `self.robot.data.joint_pos/vel` | 替换 |
| `gym.acquire_actor_root_state_tensor()` | `self.robot.data.root_pos_w/quat_w` | 替换 |
| `gym.acquire_net_contact_force_tensor()` | `self.contact_sensor.data.net_forces_w` | 替换 |
| `gym.set_dof_actuation_force_tensor()` | `self.robot.set_joint_effort_target()` | 替换 |
| `gym.set_dof_state_tensor_indexed()` | `self.robot.write_joint_state_to_sim()` | 替换 |
| `gym.simulate()` + `fetch_results()` | 自动处理 | 删除 |
| `gym.create_viewer()` | `AppLauncher` | 替换 |
| `terrain_utils.*` | `TerrainImporterCfg` | 替换 |

### 3.2 关键破坏性变更

| 变更 | 影响 | AdaMimic 中的影响范围 |
|------|------|---------------------|
| **四元数惯例 xyzw → wxyz** | 所有旋转计算 | `motionlib.py`(26处)、`motion_tracking.py`(156处)、`math.py`(62处) |
| **关节排序 深度优先 → 广度优先** | 关节索引映射 | G1 27-DoF 的所有关节索引需重新校准 |
| **DOF → Joint 命名** | API 调用 | 全局替换 |
| **观测返回值 tensor → dict** | RL 交互接口 | `rsl_rl` runner 需适配 |
| **PD 增益单位 1/rad → 1/deg** | 控制器参数 | PD 参数需 × π/180 |
| **URDF → USD 资产格式** | 机器人资产 | 需转换 G1 URDF 为 USD |

## 四、环境要求变更

| 项目 | 当前 (IsaacGym) | 迁移后 (IsaacLab) |
|------|----------------|------------------|
| Python | 3.8 | **3.10+** (Isaac Sim 4.x/5.x) |
| numpy | <1.24 | 无限制 |
| CUDA | 12.1 | 12.x+ |
| GPU 驱动 | - | ≥ 535.129.03 |
| 安装方式 | `pip install -e .` + 本地 isaacgym | `isaaclab -i` (Omniverse 隔离环境) |

## 五、受影响文件清单

| 文件 | 行数 | IsaacGym 调用数 | 迁移动作 |
|------|------|----------------|---------|
| `legged_gym/envs/base/base_task.py` | 165 | 20 | **重写** — 改为继承 `DirectRLEnv` |
| `legged_gym/envs/base/legged_robot.py` | 1,243 | 88 | **重写** — 所有 gym/gymtorch 调用替换 |
| `legged_gym/envs/base/motion_tracking.py` | 2,957 | 95 | **重写** — 最核心、改动最大 |
| `legged_gym/utils/motionlib.py` | 551 | 0 | **修改** — 四元数 xyzw→wxyz |
| `legged_gym/utils/math.py` | - | 0 | **修改** — 四元数函数适配 |
| `legged_gym/utils/terrain.py` | - | 5+ | **重写** — 使用 IsaacLab terrain API |
| `legged_gym/utils/helpers.py` | - | 3 | **修改** |
| `legged_gym/utils/task_registry.py` | - | 1 | **重写** — 改用 gymnasium 注册 |
| `legged_gym/scripts/train.py` | - | 1 | **修改** — 入口点适配 |
| `legged_gym/scripts/play.py` | - | 1 | **修改** — 入口点适配 |
| `rsl_rl/` | - | **0** | **无需改动** ✅ |
| 资产文件 | - | - | **转换** — G1 URDF → USD |

## 六、分阶段迁移方案

### Phase 0: 环境准备（2-3 天）

```
1. 安装 Isaac Sim 5.0 + Isaac Lab v2.3.x
2. 运行 IsaacLab 自带 Unitree G1 示例验证基础可用
3. 将 G1 27-DoF URDF 转换为 USD (isaaclab 的 convert 工具)
4. 创建新的 conda 环境 (Python 3.10)
```

### Phase 1: 骨架迁移 — base_task + 骨骼验证（1 周）

```
1. 继承 DirectRLEnv 创建新的 AdaMimicBaseTask
2. 实现 _setup_scene()：加载 G1 USD 资产 + 地面
3. 验证仿真可运行：robot 能站立
4. 验证 tensor 访问：joint_pos/vel, root_state 可读
```

### Phase 2: 核心环境迁移 — legged_robot + motion_tracking（2-3 周）

```
1. 迁移 legged_robot.py
   - 所有 gym.* API → DirectRLEnv 对应方法
   - 删除所有 gymtorch.wrap/unwrap
   - 四元数 xyzw → wxyz 全局适配
   - PD 增益单位转换

2. 迁移 motion_tracking.py (最复杂)
   - MotionLib 加载的参考运动数据四元数转换
   - 奖励函数中的旋转计算适配
   - Domain randomization API 适配

3. 关节索引校准
   - 对比 IsaacGym 和 IsaacLab 中 G1 27-DoF 的关节顺序
   - 建立映射表
```

### Phase 3: 训练流程适配（1 周）

```
1. 适配 rsl_rl runner
   - 观测格式: tensor → {"policy": obs}
   - 确认 TrackPPO/TrackOnPolicyRunner 兼容性
   - Hydra 配置迁移到 IsaacLab 的配置体系

2. Stage 1 训练验证
   - 运行 badminton_hit stage1
   - 对比奖励曲线与 IsaacGym 版本
```

### Phase 4: Stage 2 + 完整验证（1 周）

```
1. Stage 2 迁移（TrackActorCriticDelta）
2. 全部 7 个 task 训练验证
3. 性能基准对比 (吞吐量 envs/sec)
4. play.py 评估 + 模型导出
```

## 七、风险与缓解

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| 四元数转换遗漏导致奖励计算偏差 | 高 | 先在 IsaacLab 中还原 G1 站立场景，对比 joint/body pose 数值与 IsaacGym 一致 |
| 关节顺序不一致导致动作映射错乱 | 高 | 打印两套框架的 joint_names 并建立显式映射 |
| USD 资产与 URDF 物理属性不一致 | 中 | 使用 IsaacLab 官方 G1 资产为基线，逐步调整质量/惯性参数 |
| 训练收敛性差异 | 中 | 先在简单 task（badminton_hit）上对标，确认 reward 曲线一致后再扩展 |
| IsaacLab 版本稳定性 | 低 | 使用 v2.3.x stable，避免 v3.0-beta |
| IsaacGym 已停止维护，未来不可用 | — | 这正是迁移的动机 |

## 八、工作量估算

| 阶段 | 估计时间 | 优先级 |
|------|---------|--------|
| Phase 0: 环境准备 | 2-3 天 | P0 |
| Phase 1: 骨架验证 | 1 周 | P0 |
| Phase 2: 核心环境迁移 | 2-3 周 | P0 |
| Phase 3: 训练流程适配 | 1 周 | P0 |
| Phase 4: 完整验证 | 1 周 | P1 |
| **总计** | **5-8 周** | |

## 九、建议策略

**推荐渐进式迁移**，而非一次性切换：

1. 先在 IsaacLab 中实现一个 **最小可用版本**（G1 站立 + 简单运动跟踪）
2. 与 IsaacGym 版本进行 **数值对比验证**（同一 motion data → 同一 reward 值）
3. 验证通过后再逐步迁移所有 task 和 domain randomization
4. 保留 IsaacGym 代码作为参考，在新分支上开发 IsaacLab 版本

## 十、参考资料

- [IsaacLab 官方迁移文档](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)
- [IsaacLab 与 IsaacGym 仿真对比](https://isaac-sim.github.io/IsaacLab/main/source/migration/comparing_simulation_isaacgym.html)
- [IsaacLab GitHub](https://github.com/isaac-sim/IsaacLab)
- [Unitree RL Lab (IsaacLab-based)](https://github.com/unitreerobotics/unitree_rl_lab) — Unitree 官方基于 IsaacLab 的 RL 训练仓库，含 G1 支持
- [IsaacLab 论文 (arXiv 2511.04831)](https://arxiv.org/abs/2511.04831)
