"""Train SAC on a Gymnasium MuJoCo environment (default: HalfCheetah-v5).

A "standard" training setup following rl-baselines3-zoo conventions:
  - separate train / eval environments, each wrapped in Monitor
  - EvalCallback   -> periodically evaluates and saves the best model
  - CheckpointCallback -> periodic snapshots so a crash doesn't lose progress
  - TensorBoard logging for reward / loss / entropy curves
  - reproducible seeding

Run:
    python train_sac.py --env HalfCheetah-v5 --total-timesteps 1000000 --seed 0
Then watch curves:
    tensorboard --logdir runs
Evaluate / render the best model:
    python enjoy.py --run runs/HalfCheetah-v5_sac_0
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import torch

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC on a MuJoCo env.")
    # --- experiment ---
    p.add_argument("--env", type=str, default="HalfCheetah-v5",
                   help="Gymnasium env id (e.g. HalfCheetah-v5, Hopper-v5, Ant-v5).")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' | 'cpu' | 'cuda'. For small MuJoCo nets, 'cpu' is often faster.")
    p.add_argument("--run-name", type=str, default=None,
                   help="Override the auto-generated run directory name.")
    p.add_argument("--log-root", type=str, default="runs")

    # --- SAC hyperparameters (zoo defaults that work well across MuJoCo) ---
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    p.add_argument("--ent-coef", type=str, default="auto",
                   help="'auto' tunes the entropy temperature automatically (recommended).")

    # --- evaluation / checkpointing cadence ---
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="Env steps between evaluations.")
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    return p.parse_args()


def make_env(env_id: str, seed: int, log_dir: str | None = None) -> gym.Env:
    """Single Monitor-wrapped env. Monitor records episode reward/length for logging."""
    env = gym.make(env_id)
    monitor_path = os.path.join(log_dir, "monitor.csv") if log_dir else None
    env = Monitor(env, filename=monitor_path)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def main() -> None:
    args = parse_args()

    run_name = args.run_name or f"{args.env}_sac_{args.seed}"
    run_dir = os.path.join(args.log_root, run_name)
    best_dir = os.path.join(run_dir, "best_model")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tb")
    for d in (run_dir, best_dir, ckpt_dir, tb_dir):
        os.makedirs(d, exist_ok=True)

    # Reproducibility: seed python/numpy/torch globally, then seed each env explicitly.
    set_random_seed(args.seed)

    # Train env (seed) and a *separate* eval env (seed + 1000 so eval != train rollouts).
    train_env = make_env(args.env, args.seed, log_dir=run_dir)
    eval_env = make_env(args.env, args.seed + 1000)

    policy_kwargs = dict(net_arch=list(args.net_arch))

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_dir,
        device=args.device,
        seed=args.seed,
        verbose=1,
    )

    print(f"[info] env={args.env}  device={model.device}  run_dir={run_dir}")

    # Save the best model (by mean eval reward) and periodic checkpoints.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=run_dir,             # writes evaluations.npz for offline plotting
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="sac",
        save_replay_buffer=False,     # replay buffers are huge; flip on if you need resume
    )

    started = datetime.now()
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_cb, ckpt_cb],
            tb_log_name="SAC",
            progress_bar=True,
        )
    finally:
        # Always save the final model, even on Ctrl-C, so the run isn't wasted.
        final_path = os.path.join(run_dir, "final_model")
        model.save(final_path)
        print(f"[info] final model saved to {final_path}.zip")
        print(f"[info] best model (by eval reward) in {best_dir}/best_model.zip")
        print(f"[info] wall time: {datetime.now() - started}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
