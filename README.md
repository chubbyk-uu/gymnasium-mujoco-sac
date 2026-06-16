# Gymnasium MuJoCo + SB3 SAC

学习 Gymnasium MuJoCo 连续控制的训练脚手架。算法用 SAC（连续控制首选：样本效率高、自动调熵、调参负担小）。

## 环境

- gymnasium 1.2.3 / mujoco 3.9.0 / stable-baselines3 2.7.1 / torch 2.10.0+cu128 / numpy 2.3.5
- 渲染走 WSLg 窗口（`DISPLAY=:0`）

## 训练

```bash
# HalfCheetah，100 万步（推荐入门，reward 容易上升）
python train_sac.py --env HalfCheetah-v5 --total-timesteps 1000000 --seed 0

# 换环境只改 --env：Hopper-v5 / Walker2d-v5 / Ant-v5 / Humanoid-v5
python train_sac.py --env Hopper-v5

# MuJoCo 网络小，CPU 有时比 GPU 快，可对比：
python train_sac.py --device cpu
```

产物放在 `runs/<env>_sac_<seed>/`：
- `best_model/best_model.zip` — 评估 reward 最高的模型
- `final_model.zip` — 训练结束（或 Ctrl-C）时的模型
- `checkpoints/` — 周期快照
- `tb/` — TensorBoard 日志，`evaluations.npz` — 评估曲线数据

## 看曲线

```bash
tensorboard --logdir runs
# 关注 rollout/ep_rew_mean、eval/mean_reward、train/ent_coef
```

## 评估 / 观看

```bash
# 开窗口播放 best model（自动从 run 名推断 env）
python enjoy.py --run runs/HalfCheetah-v5_sac_0

# 无窗口跑 20 个 episode 看分数
python enjoy.py --run runs/HalfCheetah-v5_sac_0 --no-render -n 20

# 录 mp4
python enjoy.py --run runs/HalfCheetah-v5_sac_0 --video out.mp4
```

## 参考分数（1M 步，SAC 默认超参）

| Env | 大致 reward |
|-----|------------|
| HalfCheetah-v5 | ~9000–12000 |
| Hopper-v5 | ~3000+ |
| Walker2d-v5 | ~4000+ |
| Ant-v5 | ~4000–6000 |

> v5 为推荐版本；v4 仅用于复现旧论文，v4/v5 reward 不可直接比较。
