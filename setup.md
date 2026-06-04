# Setup Notes

## isaacgym/python/setup.py

Isaac Gym Preview 4 原始的 `python_requires` 限制为 `>=3.6,<3.9`，需要修改为 `>=3.6,<3.11`：

```diff
-          python_requires='>=3.6,<3.9',
+          python_requires='>=3.6,<3.11',
```

此修改仅为绕过 pip 的版本检查。Isaac Gym 的预编译 bindings 仅有 `gym_36.so`、`gym_37.so`、`gym_38.so`，实际仍需 Python 3.8。
