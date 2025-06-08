#!/usr/bin/env python
# main_ddpg.py – on-policy DDPG for the dynamic-topology ISTN
import argparse, os, time, random, json
import numpy as np, torch, matplotlib.pyplot as plt

import environment as env_mod              # dynamic ISTNEnv
import DDPG as rl                          # OnPolicyDDPG class
import utils                               # (buffer shapes only)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------- helper ------------------------------------------------ #
def whiten(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)

# ---------------- CLI --------------------------------------------------- #
p = argparse.ArgumentParser()

# topology
p.add_argument("--num_sat_antennas", default=15, type=int)
p.add_argument("--num_bs_antennas",  default=16, type=int)
p.add_argument("--max_FSS",          default=8,  type=int)
p.add_argument("--max_PU",           default=20, type=int)
p.add_argument("--topo_hz",          default=1000, type=int)

# power & noise
p.add_argument("--sat_power_dBW", default=23.01,   type=float)
p.add_argument("--bs_power_dBW",  default=-22.0,   type=float)
p.add_argument("--noise_s_dBW",   default=-126.47, type=float)
p.add_argument("--noise_p_dBW",   default=-121.52, type=float)
p.add_argument("--noise_e_dBW",   default=-121.52, type=float)

# RL hyper-params
p.add_argument("--num_eps",        default=20,     type=int)
p.add_argument("--steps_per_eps",  default=10_000, type=int)
p.add_argument("--lr",             default=1e-3,   type=float)
p.add_argument("--decay",          default=1e-5,   type=float)
p.add_argument("--discount",       default=0.99,   type=float)
p.add_argument("--tau",            default=5e-3,   type=float)
p.add_argument("--exploration_noise", default=0.12, type=float)

# misc
p.add_argument("--seed", default=0, type=int)
p.add_argument("--gpu",  default="0")
p.add_argument("--save_model", action="store_true")
p.add_argument("--model_dir",  default="./ModelsOnline")
args = p.parse_args()

# ---------------- reproducibility -------------------------------------- #
random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# ---------------- env --------------------------------------------------- #
to_lin = lambda dbw: 10 ** (dbw / 10)
env = env_mod.ISTNEnv(
    Ns=args.num_sat_antennas,
    Np=args.num_bs_antennas,
    max_FSS=args.max_FSS,
    max_PU=args.max_PU,
    topo_hz=args.topo_hz,
    Ps_max=to_lin(args.sat_power_dBW),
    Pb_max=to_lin(args.bs_power_dBW),
    sigma_s=to_lin(args.noise_s_dBW),
    sigma_p=to_lin(args.noise_p_dBW),
    sigma_e=to_lin(args.noise_e_dBW)
)

# ---------------- agent ------------------------------------------------- #
agent = rl.OnPolicyDDPG(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    actor_lr=args.lr, critic_lr=args.lr,
    actor_decay=args.decay, critic_decay=args.decay,
    max_action=1.0,
    discount=args.discount, tau=args.tau,
    device=device
)

# ---------------- training --------------------------------------------- #
episode_log   = []            # list of per-episode traces
best_step     = -np.inf
best_ep_sum   = -np.inf
best_ep_idx   = -1
global_step   = 0
tag           = time.strftime("%Y%m%d_%H%M%S")

for ep in range(args.num_eps):
    s = whiten(env.reset())
    trace, sum_r = [], 0.0

    for t in range(args.steps_per_eps):
        sigma = max(0.05,
                    args.exploration_noise * (1 - global_step / 600_000))
        a = agent.select_action(s) + \
            np.random.normal(0, sigma, env.action_dim)
        a = np.clip(a, -1.0, 1.0)

        s2, r_raw, _, _ = env.step(a)
        trace.append(r_raw); sum_r += r_raw

        agent.train_step(s, a, r_raw / 5.0, s2, 0.0)   # on-policy update
        s = whiten(s2)

        # best single step
        if r_raw > best_step:
            best_step = r_raw
            if args.save_model:
                os.makedirs(args.model_dir, exist_ok=True)
                ck = f"{args.model_dir}/best_step_{tag}_{best_step:.2f}"
                agent.save(ck)
        print(f"[Ep {ep+1:03d}|{t+1:05d}] R={r_raw:7.3f} σ={sigma:4.2f} "
              f"best_step={best_step:7.3f}", end="\r", flush=True)
        global_step += 1
    print()

    episode_log.append(trace)
    print(f"Episode {ep+1:03d} — sum reward: {sum_r:8.2f}")

    if sum_r > best_ep_sum:
        best_ep_sum, best_ep_idx = sum_r, ep

# ---------------- save trace & meta ------------------------------------ #
best_trace = episode_log[best_ep_idx]
fname_base = f"ddpg_best_episode_{tag}"
np.save(f"{fname_base}_trace.npy", np.array(best_trace))
with open(f"{fname_base}_meta.json", "w") as f:
    json.dump({
        "episode_index": best_ep_idx + 1,
        "sum_reward": best_ep_sum,
        "best_single_step": best_step,
        "steps_per_episode": args.steps_per_eps,
        "seed": args.seed
    }, f, indent=2)

# ---------------- plot -------------------------------------------------- #
plt.figure(figsize=(8, 4))
plt.plot(best_trace)
plt.xlabel("Step"); plt.ylabel("Secrecy capacity (bit/s/Hz)")
plt.title(f"Best-sum episode #{best_ep_idx+1}")
png = f"{fname_base}.png"
plt.tight_layout(); plt.savefig(png)

print("\nFinished.")
print(f"Best single-step reward : {best_step:.2f} bit/s/Hz")
print(f"Best-sum episode        : #{best_ep_idx+1}  (sum = {best_ep_sum:.1f})")
print(f"Plot saved to           : {png}")
print(f"Trace saved to          : {fname_base}_trace.npy & meta.json")
