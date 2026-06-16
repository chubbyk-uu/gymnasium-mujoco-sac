"""Fair CPU-vs-GPU throughput benchmark for this SAC setup.

Measures *training* FPS (env steps/sec while gradient updates happen), which is
the regime that actually dominates a real run. The first `learning_starts` steps
are pure sampling with no updates, so we warm up past them (and past CUDA init)
before timing.
"""
import argparse
import os
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


def bench(device: str, env_id: str, warmup: int, timed: int, seed: int = 0) -> float:
    env = Monitor(gym.make(env_id))
    env.reset(seed=seed)
    model = SAC(
        "MlpPolicy", env,
        learning_starts=500,          # start updating early so warmup is short
        train_freq=1, gradient_steps=1,
        batch_size=256, buffer_size=100_000,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device, seed=seed, verbose=0,
    )
    # Warmup: fill buffer past learning_starts + trigger CUDA/JIT init + cache warm.
    model.learn(total_timesteps=warmup, progress_bar=False)
    # Timed segment: continue training (updates active the whole time).
    t0 = time.perf_counter()
    model.learn(total_timesteps=timed, reset_num_timesteps=False, progress_bar=False)
    dt = time.perf_counter() - t0
    env.close()
    return timed / dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HalfCheetah-v5")
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--timed", type=int, default=10000)
    args = ap.parse_args()

    print(f"env={args.env}  warmup={args.warmup}  timed={args.timed} steps\n")
    results = {}
    for dev in ("cpu", "cuda"):
        fps = bench(dev, args.env, args.warmup, args.timed)
        results[dev] = fps
        print(f"  {dev:4s}: {fps:8.1f} env steps/sec   "
              f"(~{args.timed/fps:.1f}s for {args.timed} timed steps)")
    faster = max(results, key=results.get)
    ratio = results[faster] / results[min(results, key=results.get)]
    print(f"\n  => {faster.upper()} faster by {ratio:.2f}x for this net/env")
