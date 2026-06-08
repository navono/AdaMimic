#!/usr/bin/env python3
"""Print training status with trend-based benchmarks."""
import glob, os, re, sys
import time
from datetime import datetime, timedelta
from wcwidth import wcswidth

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Please run in adamimic conda environment"); sys.exit(1)


GREEN, YELLOW, RED, BOLD, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"


def display_width(text):
    """wcswidth ignoring ANSI escape codes."""
    return wcswidth(re.sub(r'\033\[[0-9;]*m', '', text))


def pad(text, width, align="<", color=None):
    """Pad text to width, handling CJK chars and ANSI color codes."""
    if color:
        text = f"{color}{text}{RESET}"
    spaces = max(width - display_width(text), 0)
    if align == ">":
        return " " * spaces + text
    elif align == "^":
        left = spaces // 2
        return " " * left + text + " " * (spaces - left)
    return text + " " * spaces


exp_dir = sys.argv[1] if len(sys.argv) > 1 else "exp/g1_dof27/badminton_hit/adamimic_stage1"
stage = sys.argv[2] if len(sys.argv) > 2 else "stage1"

# tag, description, direction, target range per stage (完成时预期值)
# Stage1: 精确跟踪参考动作，无时间扰动
# Stage2: 引入时间扰动 (±0.015~0.02s)，冻结低层策略，跟踪指标会略低
METRICS = {
    "stage1": {
        "Train/mean_raw_reward":                       ("平均原始奖励",       "higher", "30~60"),
        "Train/mean_episode_length":                   ("平均episode长度",   "higher", "78~120"),
        "Episode/rew_sparse_tracking_body_position":   ("身体位置跟踪",       "higher", "1.2~1.5"),
        "Episode/rew_sparse_tracking_body_rot":        ("身体旋转跟踪",       "higher", "0.7~0.8"),
        "Episode/rew_sparse_tracking_trunk_height":    ("躯干高度跟踪",       "higher", "0.7~0.8"),
        "Episode/joint_pos_diff_norm":                 ("关节位置偏差",       "lower",  "0.8~1.1"),
        "Episode/upper_body_diff_norm":                ("上半身偏差",         "lower",  "0.1~0.3"),
        "Episode/lower_body_diff_norm":                ("下半身偏差",         "lower",  "0.1~0.2"),
        "Env/time_outs":                               ("超时比例",           "higher", "0.005~0.02"),
    },
    "stage2": {
        "Train/mean_raw_reward":                       ("平均原始奖励",       "higher", "20~50"),
        "Train/mean_raw_reward_high":                  ("高层策略奖励",       "higher", "20~50"),
        "Train/mean_episode_length":                   ("平均episode长度",   "higher", "60~100"),
        "Episode/rew_sparse_tracking_body_position":   ("身体位置跟踪",       "higher", "1.0~1.4"),
        "Episode/rew_sparse_tracking_body_position_feet": ("脚部位置跟踪",   "higher", "1.5~2.5"),
        "Episode/rew_sparse_tracking_body_rot":        ("身体旋转跟踪",       "higher", "0.5~0.8"),
        "Episode/rew_sparse_tracking_trunk_height":    ("躯干高度跟踪",       "higher", "0.5~0.8"),
        "Episode/rew_dense_tracking_dof_pos":          ("关节角度跟踪",       "higher", "0.1~0.3"),
        "Episode/rew_dense_tracking_body_position_local": ("局部位置跟踪",   "higher", "0.1~0.3"),
        "Episode/joint_pos_diff_norm":                 ("关节位置偏差",       "lower",  "0.8~1.2"),
        "Episode/upper_body_diff_norm":                ("上半身偏差",         "lower",  "0.1~0.3"),
        "Episode/lower_body_diff_norm":                ("下半身偏差",         "lower",  "0.1~0.25"),
        "Env/delta_time":                              ("时间调整量",         "zero",   "-0.005~0.005"),
        "Env/motion_time":                             ("动作维持时长",       "higher", "2.0~4.0"),
        "Env/time_outs":                               ("超时比例",           "higher", "0.005~0.03"),
    },
}

abs_dir = os.path.abspath(exp_dir)
dirs = sorted(glob.glob(os.path.join(abs_dir, "*")), key=os.path.getmtime)
if not dirs:
    print(f"No runs found in {abs_dir}"); sys.exit(1)
latest = dirs[-1]

ea = EventAccumulator(latest)
ea.Reload()

all_tags = ea.Tags()["scalars"]
iter_tag = "Train/mean_raw_reward"
if iter_tag in all_tags:
    current_iter = ea.Scalars(iter_tag)[-1].step
    max_iter = 40000 if stage == "stage1" else 10000
    progress_pct = current_iter / max_iter * 100
    progress = f"{current_iter}/{max_iter} ({progress_pct:.1f}%)"
else:
    progress = "N/A"
    progress_pct = 0

# Calculate training time info from first event
first_event_time = None
if iter_tag in all_tags:
    first_event_time = ea.Scalars(iter_tag)[0].wall_time
if first_event_time is None:
    first_event_time = os.path.getmtime(latest)

elapsed_s = time.time() - first_event_time
elapsed_h = elapsed_s / 3600

print(f"运行: {os.path.basename(latest)}")
if progress_pct > 0 and progress_pct < 100:
    total_h = elapsed_h / (progress_pct / 100)
    remaining_h = total_h - elapsed_h
    eta = datetime.now() + timedelta(hours=remaining_h)
    print(f"迭代进度: {progress}  |  开始: {datetime.fromtimestamp(first_event_time).strftime('%m-%d %H:%M')}  |  已训练: {elapsed_h:.1f}h  |  预计完成: {eta.strftime('%m-%d %H:%M')}")
else:
    print(f"迭代进度: {progress}  |  开始: {datetime.fromtimestamp(first_event_time).strftime('%m-%d %H:%M')}  |  已训练: {elapsed_h:.1f}h")
print()

header = (
    pad("指标", 52)
    + pad("说明", 14)
    + pad("当前值", 10, ">")
    + pad("完成时预期", 12, ">")
    + pad("趋势", 6, ">")
    + pad("状态", 8, ">")
)
print(header)
print("-" * wcswidth(header))

stage_metrics = METRICS.get(stage, METRICS["stage1"])

for tag, (desc, direction, target_range) in stage_metrics.items():
    try:
        events = ea.Scalars(tag)
        val = events[-1].value
        compare_idx = max(0, len(events) - 6)
        old_val = events[compare_idx].value
        diff = val - old_val

        if direction == "higher":
            arrow = "↑" if diff > 0 else "↓"
            lo = float(target_range.split("~")[0])
            if val >= lo:
                status, color = "OK", GREEN
            elif val >= lo * 0.5:
                status, color = "进展中", YELLOW
            else:
                status, color = "LOW", RED
        elif direction == "zero":
            # 值应在 range 中心(0) 附近波动
            lo, hi = float(target_range.split("~")[0]), float(target_range.split("~")[1])
            arrow = "→"
            if lo <= val <= hi:
                status, color = "OK", GREEN
            else:
                deviation = max(abs(val - lo), abs(val - hi))
                if deviation < abs(hi) * 2:
                    status, color = "进展中", YELLOW
                else:
                    status, color = "HIGH", RED
        else:
            arrow = "↓" if diff < 0 else "↑"
            hi = float(target_range.split("~")[1])
            if val <= hi:
                status, color = "OK", GREEN
            elif val <= hi * 2:
                status, color = "进展中", YELLOW
            else:
                status, color = "HIGH", RED

        row = (
            pad(tag, 52)
            + pad(desc, 14)
            + pad(f"{val:.3f}", 10, ">")
            + pad(target_range, 12, ">")
            + pad(arrow, 6, ">")
            + pad(status, 8, ">", color)
        )
        print(row)

    except Exception:
        row = (
            pad(tag, 52)
            + pad(desc, 14)
            + pad("N/A", 10, ">")
            + pad(target_range, 12, ">")
            + pad("N/A", 6, ">")
            + pad("N/A", 8, ">")
        )
        print(row)

print()
print(f"完成时预期 = 训练完成后的合理范围（进度 {progress_pct:.0f}%，超出预期是正常的）")
print(f"{BOLD}趋势{RESET}: ↑上升 ↓下降  |  {GREEN}OK{RESET}=已达标  {YELLOW}进展中{RESET}=正在接近  {RED}HIGH/LOW{RESET}=偏离较大（训练前期正常）")
