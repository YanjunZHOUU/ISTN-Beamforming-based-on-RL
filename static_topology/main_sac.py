"""
main_sac.py  –  Soft-Actor-Critic training on ISTNEnv
               • Logs every step reward
               • Saves them to disk for reproducible plotting
"""

import argparse, os, time, json, random, numpy as np, torch
import matplotlib.pyplot as plt

import environment, sac_agent as rl
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
p.add_argument("--buffer_size",  default=100_000, type=int)
p.add_argument("--batch_size",   default=256,    type=int)
p.add_argument("--num_eps",      default=20,     type=int)
p.add_argument("--steps_per_eps",default=10_000, type=int)
p.add_argument("--lr",           default=3e-4,   type=float)
p.add_argument("--decay",        default=1e-5,   type=float)
p.add_argument("--discount",     default=0.99,   type=float)
p.add_argument("--tau",          default=5e-3,   type=float)
p.add_argument("--exploration_noise", default=0.1, type=float)
# misc
p.add_argument("--seed", default=0, type=int)
p.add_argument("--gpu",  default="0")
p.add_argument("--plot_threshold", default=1e-3, type=float)
args = p.parse_args()

# ───────── reproducibility and device ────────────────────────────────────
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# ───────── convert dBW → W ----------------------------------------------
dbw2lin = lambda dbw: 10 ** (dbw / 10)
Ps, Pb = dbw2lin(args.sat_power_dBW), dbw2lin(args.bs_power_dBW)
sig_s, sig_p, sig_e = map(dbw2lin, (args.noise_s_dBW,
                                    args.noise_p_dBW,
                                    args.noise_e_dBW))

# ───────── environment ----------------------------------------------------
env = environment.ISTNEnv(
    N_s=args.num_sat_antennas, N_p=args.num_bs_antennas,
    N_f=args.num_FSS,          M=args.num_users,
    Ps_max=Ps, Pb_max=Pb,
    sigma_s=sig_s, sigma_p=sig_p, sigma_e=sig_e
)

# ───────── SAC agent ------------------------------------------------------
agent = rl.SAC(
    state_dim=env.state_dim, action_dim=env.action_dim,
    device=device, buffer_size=args.buffer_size,
    batch_size=args.batch_size,
    gamma=args.discount, tau=args.tau,
    actor_lr=args.lr, critic_lr=args.lr, alpha_lr=args.lr
)

# ───────── create run folder ---------------------------------------------
run_tag = time.strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs", run_tag)
os.makedirs(run_dir, exist_ok=True)

# ───────── logging containers --------------------------------------------
episode_log: list[list[float]] = []
best_step = -np.inf
best_step_ckpt = ""

# ───────── training loop --------------------------------------------------
global_step = 0
for ep in range(args.num_eps):
    s = whiten(env.reset())
    trace, ep_sum = [], 0.0

    for t in range(args.steps_per_eps):
        a = agent.act(s) + np.random.normal(0, args.exploration_noise, env.action_dim)
        a = np.clip(a, -1.0, 1.0)

        s2, r_raw, _, _ = env.step(a)
        agent.add(s, a, s2, r_raw / 5.0, 0.0)
        agent.update()
        s = whiten(s2)

        trace.append(r_raw); ep_sum += r_raw
        if r_raw > best_step:
            best_step = r_raw
            best_step_ckpt = os.path.join(run_dir, f"best_step_{best_step:.2f}")
            agent.save(best_step_ckpt)

        print(f"[Ep {ep+1:03d}|{t+1:05d}] R={r_raw:6.2f} best={best_step:6.2f}", end="\r")
        global_step += 1
    print()
    episode_log.append(trace)

# ───────── save raw data --------------------------------------------------
np.save(os.path.join(run_dir, "episode_rewards.npy"), episode_log)

meta = {
    "best_single_step": best_step,
    "best_ckpt": best_step_ckpt,
    "args": vars(args)
}
with open(os.path.join(run_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

# ───────── plot best episode (mask small values) --------------------------
best_idx = int(np.argmax([sum(tr) for tr in episode_log]))
best_trace = np.array(episode_log[best_idx], dtype=float)
best_trace[np.abs(best_trace) <= args.plot_threshold] = np.nan

plt.figure(figsize=(8, 4))
plt.plot(best_trace, lw=1)
plt.title(f"Best episode #{best_idx+1}\n(best step = {best_step:.2f} bit/s/Hz)")
plt.xlabel("Step"); plt.ylabel("Raw secrecy capacity (bit/s/Hz)")
plt.tight_layout()
png_path = os.path.join(run_dir, "best_episode.png")
plt.savefig(png_path)

print(f"\nFinished.  Data saved to {run_dir}")
print(f"Best single-step reward = {best_step:.2f} bit/s/Hz")
print(f"Best-episode plot: {png_path}")
if best_step_ckpt:
    print(f"Best-step model checkpoint: {best_step_ckpt}")
