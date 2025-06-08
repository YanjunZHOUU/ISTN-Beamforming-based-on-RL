# main.py
import argparse, os, time, random
import numpy as np, torch
import matplotlib.pyplot as plt

import environment as env_mod
import agent_td3 as rl
import utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------------------------------------------------------ #
def whiten(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)

# ---------------- CLI --------------------------------------------------- #
p = argparse.ArgumentParser()
p.add_argument("--num_sat_antennas", default=15, type=int)
p.add_argument("--num_bs_antennas",  default=16, type=int)

p.add_argument("--max_FSS",          default=8,   type=int)
p.add_argument("--max_PU",           default=20,  type=int)
p.add_argument("--topo_hz",          default=1000, type=int)

# power / noise in dBW
p.add_argument("--sat_power_dBW", default=23.01,   type=float)
p.add_argument("--bs_power_dBW",  default=-22.0,   type=float)
p.add_argument("--noise_s_dBW",   default=-126.47, type=float)
p.add_argument("--noise_p_dBW",   default=-121.52, type=float)
p.add_argument("--noise_e_dBW",   default=-121.52, type=float)

# replay / training
p.add_argument("--buffer_size",    default=100_000, type=int)
p.add_argument("--batch_size",     default=256,     type=int)
p.add_argument("--num_eps",        default=20,      type=int)
p.add_argument("--steps_per_eps",  default=10_000,  type=int)
p.add_argument("--lr",             default=1e-3,    type=float)
p.add_argument("--decay",          default=1e-5,    type=float)
p.add_argument("--discount",       default=0.99,    type=float)
p.add_argument("--tau",            default=5e-3,    type=float)
p.add_argument("--exploration_noise", default=0.1,  type=float)

# misc
p.add_argument("--seed", default=0, type=int)
p.add_argument("--gpu",  default="0")
args = p.parse_args()

# ---------------- reproducibility --------------------------------------- #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# ---------------- convert dBW → W --------------------------------------- #
to_lin = lambda dbw: 10 ** (dbw / 10)
Ps_lin, Pb_lin = to_lin(args.sat_power_dBW), to_lin(args.bs_power_dBW)
sig_s, sig_p, sig_e = map(to_lin, (args.noise_s_dBW,
                                   args.noise_p_dBW,
                                   args.noise_e_dBW))

# ---------------- environment ------------------------------------------- #
env = env_mod.ISTNEnv(
    Ns=args.num_sat_antennas,
    Np=args.num_bs_antennas,
    max_FSS=args.max_FSS,
    max_PU=args.max_PU,
    topo_hz=args.topo_hz,
    Ps_max=Ps_lin,
    Pb_max=Pb_lin,
    sigma_s=sig_s,
    sigma_p=sig_p,
    sigma_e=sig_e
)

# ---------------- agent & replay ---------------------------------------- #
agent = rl.TD3(
    env.state_dim,
    args.num_sat_antennas, args.max_FSS,
    args.num_bs_antennas,  args.max_PU,
    Ps_lin, Pb_lin, 1.0,
    args.lr, args.lr, args.decay, args.decay,
    device, discount=args.discount, tau=args.tau
)

replay = utils.ExperienceReplayBuffer(
    env.state_dim, env.action_dim, args.buffer_size)

# ---------------- logs -------------------------------------------------- #
episode_traces = []
best_sum, best_ep = -np.inf, -1
tag = time.strftime("%Y%m%d_%H%M%S")

global_step, warmup = 0, 5_000

# ---------------- training loop ----------------------------------------- #
for ep in range(args.num_eps):
    s = whiten(env.reset())
    ep_trace, ep_sum = [], 0.0

    for t in range(args.steps_per_eps):
        sigma = max(0.05, args.exploration_noise *
                    (1 - global_step / 600_000))
        a = agent.select_action(s) + \
            np.random.normal(0, sigma, env.action_dim)
        a = np.clip(a, -1.0, 1.0)

        s2, r_raw, _, _ = env.step(a)
        ep_trace.append(r_raw); ep_sum += r_raw

        replay.add(s, a, s2, r_raw / 5.0, 0.0)
        s = whiten(s2)

        if replay.size > warmup:
            agent.update(replay, args.batch_size)

        print(f"[Ep {ep+1:03d}|{t+1:05d}] "
              f"R={r_raw:6.2f} σ={sigma:4.2f}", end="\r", flush=True)
        global_step += 1
    print()  # newline after episode loop

    episode_traces.append(ep_trace)
    print(f"Episode {ep+1:03d} — sum reward: {ep_sum:8.2f}")

    if ep_sum > best_sum:
        best_sum, best_ep = ep_sum, ep

# ---------------- plot best-sum episode --------------------------------- #
best_trace = episode_traces[best_ep]
plt.figure(figsize=(8, 4))
plt.plot(best_trace)
plt.xlabel("Step")
plt.ylabel("Raw secrecy capacity (bit/s/Hz)")
plt.title(f"Best episode #{best_ep + 1}  (sum = {best_sum:.1f} bit/s/Hz)")
png_path = f"best_episode_{tag}.png"
plt.tight_layout()
plt.savefig(png_path)

print(f"\nTraining finished.")
print(f"Best episode: {best_ep + 1}  |  sum reward = {best_sum:.1f}")
print(f"Plot saved to: {png_path}")

