#!/usr/bin/env python3
"""Visualize reference motion data as 3D skeleton animation and joint trajectory plots."""
import os, sys
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Link names matching data.pt link_position order (17 links)
LINK_NAMES = [
    "pelvis", "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "head", "torso",
    "left_collar", "left_shoulder", "left_elbow", "left_wrist",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist",
]

# Skeleton edges: (parent_link, child_link)
SKELETON = [
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("pelvis", "torso"), ("torso", "head"),
    ("torso", "left_collar"), ("left_collar", "left_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("torso", "right_collar"), ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
]

JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]


def load_data(path):
    data = torch.load(path, weights_only=False)
    return {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def find_link_idx(name):
    for i, ln in enumerate(LINK_NAMES):
        if ln == name:
            return i
    return None


def plot_skeleton_3d(data, sample=0, out_dir="videos/ref_motion"):
    """Save 3D skeleton animation as mp4."""
    os.makedirs(out_dir, exist_ok=True)
    link_pos = data["link_position"]  # [T, 17, 3]
    T = link_pos.shape[0]

    edges = []
    for parent, child in SKELETON:
        pi, ci = find_link_idx(parent), find_link_idx(child)
        if pi is not None and ci is not None:
            edges.append((pi, ci))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Compute axis limits from all frames
    all_pos = link_pos.reshape(-1, 3)
    margin = 0.2
    xlim = [all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin]
    ylim = [all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin]
    zlim = [all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin]

    def update(frame):
        ax.cla()
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"Reference Motion (frame {frame}/{T-1})")

        pos = link_pos[frame]  # [17, 3]

        # Draw edges
        for pi, ci in edges:
            ax.plot([pos[pi, 0], pos[ci, 0]],
                    [pos[pi, 1], pos[ci, 1]],
                    [pos[pi, 2], pos[ci, 2]], "b-o", markersize=3, linewidth=2)

        # Draw joints
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="red", s=20, zorder=5)

    anim = FuncAnimation(fig, update, frames=T, interval=33)
    video_path = os.path.join(out_dir, "skeleton_3d.mp4")
    anim.save(video_path, writer="ffmpeg", fps=30, dpi=100)
    plt.close()
    print(f"3D skeleton animation: {video_path} ({os.path.getsize(video_path)/1024:.0f} KB)")


def plot_joint_trajectories(data, out_dir="videos/ref_motion"):
    """Plot joint position trajectories over time."""
    os.makedirs(out_dir, exist_ok=True)
    joint_pos = data["joint_position"]  # [T, 27]
    T = joint_pos.shape[0]
    t = np.arange(T) / 30.0  # 30 fps

    # Split into legs, waist, arms
    groups = {
        "Left Leg": list(range(0, 6)),
        "Right Leg": list(range(6, 12)),
        "Waist": [12],
        "Left Arm": list(range(13, 20)),
        "Right Arm": list(range(20, 27)),
    }

    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 2.5 * len(groups)), squeeze=False)
    for ax, (group_name, indices) in zip(axes.flat, groups.items()):
        for idx in indices:
            ax.plot(t, joint_pos[:, idx], label=JOINT_NAMES[idx], linewidth=1)
        ax.set_title(group_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (rad)")
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "joint_trajectories.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Joint trajectories: {path} ({os.path.getsize(path)/1024:.0f} KB)")


def plot_base_trajectory(data, out_dir="videos/ref_motion"):
    """Plot base position and orientation over time."""
    os.makedirs(out_dir, exist_ok=True)
    T = data["base_position"].shape[0]
    t = np.arange(T) / 30.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    labels = ["x", "y", "z"]
    for i, lb in enumerate(labels):
        ax1.plot(t, data["base_position"][:, i], label=lb, linewidth=1)
    ax1.set_title("Base Position")
    ax1.set_ylabel("Position (m)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    for i, lb in enumerate(["roll", "pitch", "yaw"]):
        ax2.plot(t, data["base_pose"][:, i], label=lb, linewidth=1)
    ax2.set_title("Base Orientation")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Angle (rad)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "base_trajectory.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Base trajectory: {path} ({os.path.getsize(path)/1024:.0f} KB)")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else \
        "legged_gym/resources/dataset/g1_dof27_data/badminton_hit/output/data.pt"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "videos/ref_motion"

    print(f"Loading: {data_path}")
    data = load_data(data_path)
    T = data["link_position"].shape[0]
    print(f"Frames: {T}, Duration: {T/30:.1f}s, Links: {data['link_position'].shape[1]}, Joints: {data['joint_position'].shape[1]}")
    print()

    plot_joint_trajectories(data, out_dir)
    plot_base_trajectory(data, out_dir)
    plot_skeleton_3d(data, out_dir=out_dir)

    print(f"\nAll outputs saved to {out_dir}/")
