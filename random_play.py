"""Play an env with RANDOM actions (no trained policy) in a WSLg window.

Useful to *see* an env's dynamics before/without training — e.g. to watch the
InvertedDoublePendulum fold at its middle hinge.

    python random_play.py --env InvertedDoublePendulum-v5 -n 3
"""

import argparse
import time
import gymnasium as gym


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="InvertedDoublePendulum-v5")
    p.add_argument("-n", "--n-episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold", action="store_true",
                   help="Keep stepping even after it falls (ignore termination), so you "
                        "can watch the linkage flail. Auto-resets only on truncation.")
    p.add_argument("--steps", type=int, default=400,
                   help="With --hold: how many steps to run per episode.")
    p.add_argument("--slow", type=float, default=0.0,
                   help="Seconds to sleep per step (e.g. 0.03) to slow it down for viewing.")
    args = p.parse_args()

    env = gym.make(args.env, render_mode="human")
    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_ret, ep_len = 0.0, 0
        while True:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())  # random action
            ep_ret += float(r)
            ep_len += 1
            if args.slow:
                time.sleep(args.slow)
            if args.hold:
                if trunc or ep_len >= args.steps:
                    break  # ignore `term` (falling) — keep flailing until step budget
            elif term or trunc:
                break
        print(f"episode {ep + 1}: return={ep_ret:.1f}  length={ep_len}")
    env.close()


if __name__ == "__main__":
    main()
