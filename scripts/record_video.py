#!/usr/bin/env python3
"""Record a video of the trained policy using Isaac Gym viewer screenshots."""
import os, sys, subprocess, shutil

os.environ.setdefault("DISPLAY", ":99")

from isaacgym import gymapi
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, AttrDict
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../legged_gym/legged_gym/configs", config_name="eval", version_base="1.1")
def main(cfg):
    # Save resume_path before converting to AttrDict
    ckpt_path = cfg.resume_path

    cfg.env.terrain.curriculum = False
    cfg.env.termination_curriculum.terminate_when_motion_far_curriculum = False
    cfg.env.termination_curriculum.terminate_when_motion_far_initial_threshold = 1000
    cfg.env.termination.height_termination = False
    cfg.env.termination.rot_termination = False
    cfg.env.termination.dof_termination = False
    cfg.env.algorithm.rsi = False
    cfg.env.noise.add_noise = False
    cfg.env.domain_rand.use_random = False

    cfg = AttrDict(OmegaConf.to_container(cfg, resolve=True))
    cfg.run_dir = "."

    cfg.env.terrain.num_rows = 2
    cfg.env.terrain.num_cols = 2
    cfg.env.env.test = True
    cfg.algo.policy.checkpoint_path = None
    # Pass resume_path to runner
    cfg.algo.runner.resume_path = ckpt_path

    env, env_cfg = task_registry.make_env_hydra(cfgs=cfg)
    obs = env.get_observations()

    cfg.algo.runner.resume = True
    cfg.algo.policy.resume = False

    ppo_runner, train_cfg = task_registry.make_alg_runner_hydra(env=env, env_cfg=env_cfg, cfgs=cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    _, _ = env.reset()

    # Output paths
    task = cfg.task_id
    ckpt_name = os.path.basename(ckpt_path).replace(".pt", "")
    out_dir = "videos"
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    max_steps = int(env.max_episode_length)
    num_episodes = 3
    frame_count = 0

    print(f"Recording {num_episodes} episodes x {max_steps} steps...")
    for ep in range(num_episodes):
        obs, _ = env.reset()
        for step in range(max_steps):
            actions = policy(obs.detach())
            step_out = env.step(actions.detach())
            obs = step_out[0]
            reset_buf = step_out[4]

            if env.viewer:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.draw_viewer(env.viewer, env.sim, False)

                img_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.png")
                env.gym.write_viewer_image_to_file(env.viewer, img_path)
                frame_count += 1

                if frame_count % 100 == 0:
                    print(f"  ep {ep+1}, step {step}, frame {frame_count}")

            if reset_buf.any():
                break

    if env.viewer:
        env.gym.destroy_viewer(env.viewer)

    print(f"Captured {frame_count} frames")

    # Encode video
    video_path = os.path.join(out_dir, f"{task}_{ckpt_name}.mp4")
    print(f"Encoding {video_path}...")
    ret = subprocess.run([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", video_path
    ], capture_output=True)
    if ret.returncode == 0:
        print(f"Done: {video_path} ({os.path.getsize(video_path) / 1024 / 1024:.1f} MB)")
        shutil.rmtree(frames_dir)
    else:
        print(f"ffmpeg error: {ret.stderr.decode()}")
        print(f"Frames kept in {frames_dir}")


if __name__ == "__main__":
    main()
