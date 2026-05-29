"""Evaluate / watch a trained SAC model.

By default it opens a WSLg window and plays episodes with the deterministic policy.

Examples:
    # watch the best model from a run, in a window
    python enjoy.py --run runs/HalfCheetah-v5_sac_0

    # headless quantitative eval over 20 episodes (no window)
    python enjoy.py --run runs/HalfCheetah-v5_sac_0 --no-render -n 20

    # record an mp4 instead of showing a window
    python enjoy.py --run runs/HalfCheetah-v5_sac_0 --video out.mp4

    # point straight at a model zip and override the env
    python enjoy.py --model runs/HalfCheetah-v5_sac_0/best_model/best_model.zip --env HalfCheetah-v5
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate / render a trained SAC model.")
    p.add_argument("--run", type=str, default=None,
                   help="Run directory (e.g. runs/HalfCheetah-v5_sac_0). "
                        "Uses best_model unless --final is given.")
    p.add_argument("--final", action="store_true",
                   help="Use final_model.zip instead of best_model.zip.")
    p.add_argument("--model", type=str, default=None,
                   help="Explicit path to a model .zip (overrides --run).")
    p.add_argument("--env", type=str, default=None,
                   help="Env id. Inferred from the run name if omitted.")
    p.add_argument("-n", "--n-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions instead of using the deterministic mean.")
    p.add_argument("--no-render", action="store_true",
                   help="Headless: no window, just print episode stats.")
    p.add_argument("--video", type=str, default=None,
                   help="Record to this mp4 path instead of opening a window.")
    return p.parse_args()


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    if not args.run:
        raise SystemExit("Provide --model or --run.")
    name = "final_model.zip" if args.final else os.path.join("best_model", "best_model.zip")
    path = os.path.join(args.run, name)
    if not os.path.exists(path):
        raise SystemExit(f"Model not found: {path}")
    return path


def infer_env_id(args: argparse.Namespace) -> str:
    if args.env:
        return args.env
    src = args.run or args.model or ""
    # run dirs look like "<EnvId>_sac_<seed>"; grab the part before "_sac"
    m = re.search(r"([A-Za-z0-9]+-v\d+)_sac", os.path.basename(os.path.normpath(src)))
    if not m:
        raise SystemExit("Could not infer env id from path; pass --env explicitly.")
    return m.group(1)


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args)
    env_id = infer_env_id(args)

    # render_mode: human -> WSLg window; rgb_array -> for video; None -> headless.
    if args.video:
        render_mode = "rgb_array"
    elif args.no_render:
        render_mode = None
    else:
        render_mode = "human"

    env = gym.make(env_id, render_mode=render_mode)
    if args.video:
        os.makedirs(os.path.dirname(os.path.abspath(args.video)) or ".", exist_ok=True)
        video_dir = os.path.dirname(os.path.abspath(args.video)) or "."
        name_prefix = os.path.splitext(os.path.basename(args.video))[0]
        env = gym.wrappers.RecordVideo(
            env, video_folder=video_dir, name_prefix=name_prefix,
            episode_trigger=lambda i: True,
        )

    model = SAC.load(model_path, device="cpu")  # inference is light; cpu avoids GPU transfer cost
    print(f"[info] model={model_path}  env={env_id}  deterministic={not args.stochastic}")

    returns, lengths = [], []
    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(ep_len)
        print(f"  episode {ep + 1:>2}: return={ep_ret:9.1f}  length={ep_len}")

    env.close()
    r = np.array(returns)
    print(f"[result] return  mean={r.mean():.1f}  std={r.std():.1f}  "
          f"min={r.min():.1f}  max={r.max():.1f}  over {args.n_episodes} episodes")
    if args.video:
        print(f"[info] video written under {os.path.dirname(os.path.abspath(args.video)) or '.'}")


if __name__ == "__main__":
    main()
