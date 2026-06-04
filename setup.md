# Setup Notes

## isaacgym/python/setup.py

Isaac Gym Preview 4 原始的 `python_requires` 限制为 `>=3.6,<3.9`，需要修改为 `>=3.6,<3.11`：

```diff
-          python_requires='>=3.6,<3.9',
+          python_requires='>=3.6,<3.11',
```

此修改仅为绕过 pip 的版本检查。Isaac Gym 的预编译 bindings 仅有 `gym_36.so`、`gym_37.so`、`gym_38.so`，实际仍需 Python 3.8。

## LD_LIBRARY_PATH

Isaac Gym 运行时需要加载 `libpython3.8.so`。conda activate 后需设置：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

否则报错：`ImportError: libpython3.8.so.1.0: cannot open shared object file`

Makefile 中已内置此设置。若手动运行训练命令，需先执行此 export。

## numpy 版本

Isaac Gym 的 `isaacgym/torch_utils.py` 使用了 `np.float`（已在 NumPy 1.24 中移除）。`conda_env.yml` 中指定的 `numpy=1.20.3` 不受影响。若手动升级 numpy，需确保 `numpy<1.24`。

## 训练命令

README 中的训练命令缺少 `+robot` 参数，正确命令为：

```bash
# Stage 1
python legged_gym/legged_gym/scripts/train.py +dataset=g1_dof27/<task> +algorithm=adamimic/stage1 +robot=g1_dof27

# Stage 2
python legged_gym/legged_gym/scripts/train.py +dataset=g1_dof27/<task> +algorithm=adamimic/stage2 +robot=g1_dof27 checkpoint_path=<stage1_ckpt>

# Play
python legged_gym/legged_gym/scripts/play.py +dataset=g1_dof27/<task> +algorithm=adamimic/stage2 +robot=g1_dof27 resume_path=<stage2_ckpt>
```

也可直接使用 Makefile（见 `make help`）。

## 训练输出目录

Checkpoint 和日志保存在项目目录下的 `exp/<robot>/<task>/<stage>/<timestamp>/`。
