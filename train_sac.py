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
import tempfile
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
)
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

    # --- resume ---
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a saved .zip model to resume from (e.g. "
                        "runs/HalfCheetah-v5_sac_0/final_model.zip). Continues the "
                        "step counter. If a matching *_replay_buffer.pkl exists next "
                        "to it, the replay buffer is restored too (seamless resume); "
                        "otherwise it warm-starts from the weights and refills the buffer.")
    p.add_argument("--save-replay-buffer", action="store_true",
                   help="Also save the replay buffer in checkpoints / final model so a "
                        "future --resume can be fully seamless. Files are large (~GBs).")

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

    # --- early stopping (optional; off by default) ---
    p.add_argument("--stop-no-improve", type=int, default=None,
                   help="Stop if the best eval reward does not improve for this many "
                        "consecutive evals (env-agnostic, recommended). Use a generous "
                        "value (e.g. 15-20) for envs with early termination that "
                        "oscillate (Hopper/Walker2d/Ant), or they may stop in a dip.")
    p.add_argument("--stop-reward", type=float, default=None,
                   help="Stop as soon as the mean eval reward reaches this threshold "
                        "(needs a per-env target; checked only on a new best).")
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

    if args.resume:
        # Resume: load weights (and optimizer state) and keep training.
        model = SAC.load(
            args.resume,
            env=train_env,
            tensorboard_log=tb_dir,
            device=args.device,
        )
        # Try to restore the replay buffer for a seamless resume; SB3 saves it
        # next to the model as "<name>_replay_buffer.pkl".
        buf_path = args.resume[:-4] if args.resume.endswith(".zip") else args.resume
        buf_path += "_replay_buffer.pkl"
        if os.path.exists(buf_path):
            model.load_replay_buffer(buf_path)
            print(f"[info] resumed WITH replay buffer ({model.replay_buffer.size()} "
                  f"transitions) from {args.resume}")
        else:
            print(f"[info] resumed from weights only (no replay buffer at {buf_path}); "
                  f"buffer will refill — expect a brief dip before recovery.")
    else:
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

    print(f"[info] env={args.env}  device={model.device}  run_dir={run_dir}"
          f"{'  (resumed)' if args.resume else ''}")

    # Optional early stopping, wired into the eval callback.
    #   --stop-no-improve N -> StopTrainingOnNoModelImprovement (after each eval)
    #   --stop-reward R     -> StopTrainingOnRewardThreshold   (on each new best)
    on_new_best = None
    after_eval = None
    if args.stop_reward is not None:
        on_new_best = StopTrainingOnRewardThreshold(
            reward_threshold=args.stop_reward, verbose=1)
        print(f"[info] early stop: when eval reward >= {args.stop_reward}")
    if args.stop_no_improve is not None:
        after_eval = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=args.stop_no_improve,
            min_evals=args.stop_no_improve, verbose=1)
        print(f"[info] early stop: after {args.stop_no_improve} evals with no new best")

    # Save the best model (by mean eval reward) and periodic checkpoints.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=run_dir,             # writes evaluations.npz for offline plotting
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=on_new_best,
        callback_after_eval=after_eval,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=ckpt_dir,
        name_prefix="sac",
        save_replay_buffer=args.save_replay_buffer,  # large; enable for seamless resume
    )

    # --total-timesteps is always the *target total* step count. On resume we only
    # run the remaining steps and keep counting from where we left off.
    if args.resume:
        learn_steps = args.total_timesteps - model.num_timesteps
        if learn_steps <= 0:
            raise SystemExit(
                f"[error] already at {model.num_timesteps} steps >= target "
                f"{args.total_timesteps}; raise --total-timesteps to continue.")
        print(f"[info] resuming at {model.num_timesteps} steps; "
              f"running {learn_steps} more to reach {args.total_timesteps}.")
    else:
        learn_steps = args.total_timesteps

    started = datetime.now()
    try:
        model.learn(
            total_timesteps=learn_steps,
            callback=[eval_cb, ckpt_cb],
            tb_log_name="SAC",
            progress_bar=True,
            reset_num_timesteps=not bool(args.resume),
        )
    finally:
        # Always save the final model, even on Ctrl-C, so the run isn't wasted.
        final_path = os.path.join(run_dir, "final_model")
        model.save(final_path)
        if args.save_replay_buffer:
            model.save_replay_buffer(final_path + "_replay_buffer.pkl")
            print(f"[info] replay buffer saved to {final_path}_replay_buffer.pkl")
        print(f"[info] final model saved to {final_path}.zip")
        print(f"[info] best model (by eval reward) in {best_dir}/best_model.zip")
        print(f"[info] wall time: {datetime.now() - started}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
