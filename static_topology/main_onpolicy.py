"""
main_ddpg_online.py
On-policy deterministic policy-gradient (DDPG without replay).

• Logs every step reward.
• Saves only the data needed to reproduce the plot.
• Writes no model checkpoints or other artefacts.
"""

import argparse, os, time, random, json
import numpy as np, torch, matplotlib.pyplot as plt

import environment
import DDPG as rl   # your OnPolicyDDPG implementation

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def whiten(x): return (x - x.mean()) / (x.std() + 1e-12)

# ───────── CLI ───────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
# topology
p.add_argument("--num_sat_antennas", default=15, type=int)
p.add_argument("--num_bs_antennas",  default=16, type=int)
p.add_argument("--num_FSS",          default=5,  type=int)
p.add_argument("--num_users",        default=15, type=int)
# power & noise (dBW)
p.add_argument("--sat_power_dBW", default=23.01,  type=float)
p.add_argument("--bs_power_dBW",  default=-22.0,  type=float)
p.add_argument("--noise_s_dBW",   default=-126.47,type=float)
p.add_argument("--noise_p_dBW",   default=-121.52,type=float)
p.add_argument("--noise_e_dBW",   default=-121.52,type=float)
# RL hyper-params
p.add_argument("--num_eps",      default=20,     type=int)
p.add_argument("--steps_per_eps",default=10_000, type=int)
p.add_argument("--lr",  default=1e-3, type=float)
p.add_argument("--decay", default=1e-5, type=float)
p.add_argument("--discount", default=0.99, type=float)
p.add_argument("--tau", default=5e-3, type=float)
p.add_argument("--exploration_noise", default=0.1, type=float)
# reproducibility / I/O
p.add_argument("--seed", default=0, type=int)
p.add_argument("--gpu",  default="0")
p.add_argument("--plot_threshold", default=1e-3, type=float,
               help="Hide rewards with |r| ≤ threshold in the PNG")
args = p.parse_args()

# ───────── seeds ─────────────────────────────────────────────────────────
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# dBW → W
to_lin = lambda dbw: 10 ** (dbw / 10)
Ps, Pb = to_lin(args.sat_power_dBW), to_lin(args.bs_power_dBW)
sig_s, sig_p, sig_e = map(to_lin, (args.noise_s_dBW,
                                   args.noise_p_dBW,
                                   args.noise_e_dBW))

env = environment.ISTNEnv(
    N_s=args.num_sat_antennas, N_p=args.num_bs_antennas,
    N_f=args.num_FSS,          M=args.num_users,
    Ps_max=Ps, Pb_max=Pb,
    sigma_s=sig_s, sigma_p=sig_p, sigma_e=sig_e
)

agent = rl.OnPolicyDDPG(
    state_dim=env.state_dim, action_dim=env.action_dim,
    Ns=args.num_sat_antennas,
    Np=args.num_bs_antennas,
    Nf=args.num_FSS, M=args.num_users,
    Ps_max=Ps, Pb_max=Pb,
    actor_lr=args.lr, critic_lr=args.lr,
    actor_decay=args.decay, critic_decay=args.decay,
    max_action=1.0,
    discount=args.discount, tau=args.tau,
    device=device
)

# agent = OnPolicyDDPG(state_dim, action_dim,
#                      Ns=args.num_sat_antennas,
#                      Np=args.num_bs_antennas,
#                      Nf=args.num_FSS, M=args.num_users,
#                      Ps_max=Ps_lin, Pb_max=Pb_lin,
#                      actor_lr=args.lr, critic_lr=args.lr,
#                      actor_decay=args.decay, critic_decay=args.decay,
#                      discount=args.discount, tau=args.tau,
#                      max_action=1.0, device=device)


# ───────── run directory just for data & plot ────────────────────────────
run_tag = time.strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs_ddpg", run_tag)
os.makedirs(run_dir, exist_ok=True)

# ───────── training & logging ────────────────────────────────────────────
episode_log = []             # list of per-step reward traces
best_step, best_ep_idx = -np.inf, -1
global_step = 0

for ep in range(args.num_eps):
    s = whiten(env.reset())
    trace, ep_sum = [], 0.0

    for t in range(args.steps_per_eps):
        sigma = max(0.05, args.exploration_noise * (1 - global_step / 600_000))
        a = agent.select_action(s) + np.random.normal(0, sigma, env.action_dim)
        a = np.clip(a, -1.0, 1.0)

        s2, r_raw, _, _ = env.step(a)
        trace.append(r_raw); ep_sum += r_raw

        agent.train_step(s, a, r_raw / 5.0, s2, 0.0)   # single-transition update
        s = whiten(s2)

        if r_raw > best_step:
            best_step, best_ep_idx = r_raw, ep

        print(f"[Ep {ep+1:03d}|{t+1:05d}] R={r_raw:6.2f} best={best_step:6.2f}", end="\r")
        global_step += 1
    print()
    episode_log.append(trace)

# ───────── save raw data for reproducibility ─────────────────────────────
np.save(os.path.join(run_dir, "episode_rewards.npy"), episode_log)
meta = {"best_single_step": best_step,
        "best_episode_idx": best_ep_idx,
        "args": vars(args)}
with open(os.path.join(run_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

# ───────── plot best episode (mask small values) ─────────────────────────
best_trace = np.array(episode_log[best_ep_idx], dtype=float)
best_trace[np.abs(best_trace) <= args.plot_threshold] = np.nan

plt.figure(figsize=(8, 4))
plt.plot(best_trace, lw=1)
plt.title(f"Best episode #{best_ep_idx+1}  (best step = {best_step:.2f})")
plt.xlabel("Step"); plt.ylabel("Raw secrecy capacity (bit/s/Hz)")
plt.tight_layout()
png_path = os.path.join(run_dir, "best_episode.png")
plt.savefig(png_path)

print(f"\nTraining finished. Data saved in: {run_dir}")
print(f"Best single-step secrecy capacity = {best_step:.2f} bit/s/Hz")
print(f"Reproducible plot: {png_path}")


#
# import argparse, os, time, random, json
# import numpy as np, torch, matplotlib.pyplot as plt
#
# from environment  import ISTNEnv
# from DDPG import OnPolicyDDPG            # ← 关键导入
# # from sac_agent  import SAC                     # 如需 SAC 则把上行换掉
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
#
# # ─── CLI ────────────────────────────────────────────────────────────────
# p = argparse.ArgumentParser()
# #  网络尺寸
# p.add_argument("--num_sat_antennas", default=15, type=int)
# p.add_argument("--num_bs_antennas",  default=16, type=int)
# p.add_argument("--num_FSS",          default=5,  type=int)
# p.add_argument("--num_users",        default=15, type=int)
# #  功率 (dBW)
# p.add_argument("--sat_power_dBW", default=23.01,  type=float)
# p.add_argument("--bs_power_dBW",  default=-22.0,  type=float)
# #  RL 超参
# p.add_argument("--num_eps", default=20, type=int)
# p.add_argument("--steps_per_eps", default=10_0000, type=int)
# p.add_argument("--lr",    default=1e-3, type=float)
# p.add_argument("--decay", default=1e-5, type=float)
# p.add_argument("--discount", default=0.99, type=float)
# p.add_argument("--tau",   default=5e-3, type=float)
# p.add_argument("--noise", default=0.1,  type=float)      # exploration
# #  杂项
# p.add_argument("--seed", default=0, type=int)
# p.add_argument("--gpu",  default="0")
# args = p.parse_args()
#
# # ─── reproducibility ────────────────────────────────────────────────────
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark     = False
#
# device = torch.device(f"cuda:{args.gpu}"
#                       if torch.cuda.is_available() else "cpu")
#
# # ─── 功率上限转线性 ─────────────────────────────────────────────────────
# Ps_lin = 10 ** (args.sat_power_dBW / 10)          # ≈ 200.9 W
# Pb_lin = 10 ** (args.bs_power_dBW  / 10)          # ≈   6.3 mW
#
# # ─── 创建环境 ───────────────────────────────────────────────────────────
# env = ISTNEnv(N_s=args.num_sat_antennas, N_p=args.num_bs_antennas,
#               N_f=args.num_FSS,          M=args.num_users,
#               Ps_max=Ps_lin, Pb_max=Pb_lin)
#
# state_dim  = env.state_dim
# action_dim = env.action_dim
#
# # ─── 初始化 DDPG 智能体 ─────────────────────────────────────────────────
# agent = OnPolicyDDPG(state_dim, action_dim,
#                      Ns=args.num_sat_antennas,
#                      Np=args.num_bs_antennas,
#                      Nf=args.num_FSS, M=args.num_users,
#                      Ps_max=Ps_lin, Pb_max=Pb_lin,
#                      actor_lr=args.lr, critic_lr=args.lr,
#                      actor_decay=args.decay, critic_decay=args.decay,
#                      discount=args.discount, tau=args.tau,
#                      max_action=1.0, device=device)
#
# # ─── 日志缓存 ───────────────────────────────────────────────────────────
# os.makedirs("runs", exist_ok=True)
# tag = time.strftime("%Y%m%d_%H%M%S")
# run_dir = f"runs/{tag}"
# os.makedirs(run_dir, exist_ok=True)
#
# episode_rewards, best_step, best_trace, best_ep = [], -np.inf, None, -1
#
# # ─── 训练循环 ───────────────────────────────────────────────────────────
# for ep in range(args.num_eps):
#     s = env.reset()
#     trace = []
#     for t in range(args.steps_per_eps):
#         # exploration 噪声线性退火
#         sigma = max(0.05, args.noise * (1 - (ep*args.steps_per_eps+t) / 6e5))
#         a = agent.select_action(s) + np.random.normal(0, sigma, action_dim)
#         a = np.clip(a, -1.0, 1.0)
#
#         s2, r, _, _ = env.step(a)
#         agent.train_step(s, a, r, s2, 0.0)     # on-policy 无 done
#
#         trace.append(r)
#         if r > best_step:
#             best_step, best_trace, best_ep = r, trace.copy(), ep
#
#         s = s2
#
#     episode_rewards.append(trace)
#     print(f"[Ep {ep+1:02d}]  avg={np.mean(trace):.3f}  "
#           f"best_step={best_step:.2f}")
#
# # ─── 保存数据供复现 ────────────────────────────────────────────────────
# np.save(os.path.join(run_dir, "episode_rewards.npy"), episode_rewards)
# meta = {"best_single_step": best_step,
#         "best_episode_idx": best_ep,
#         "args": vars(args)}
# json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=2)
#
# # ─── 画出最佳 episode 曲线 ─────────────────────────────────────────────
# plt.figure(figsize=(8,4))
# plt.plot(best_trace, lw=1)
# plt.title(f"Best episode #{best_ep+1}  (peak = {best_step:.2f} bit/s/Hz)")
# plt.xlabel("Step"); plt.ylabel("Secrecy Capacity (bit/s/Hz)")
# plt.tight_layout()
# plt.savefig(os.path.join(run_dir, "best_episode.png"))
# plt.show()
#
# print(f"\nRun saved to  {run_dir}")
