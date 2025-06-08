#!/usr/bin/env python
# main_sac.py – 20 episodes × 10 k steps, reproducible trace, and masked plot
import argparse, os, time, random, json
import numpy as np, torch, matplotlib.pyplot as plt

import environment as env_mod
from sac import SAC
import utils                                           # replay buffer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def whiten(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


# ───────── CLI ---------------------------------------------------------- #
p = argparse.ArgumentParser()
p.add_argument("--num_sat_antennas", default=15, type=int)
p.add_argument("--num_bs_antennas",  default=16, type=int)
p.add_argument("--max_FSS",          default=10, type=int)
p.add_argument("--max_PU",           default=25, type=int)
p.add_argument("--topo_hz",          default=800, type=int)

# SAC hyper-params
p.add_argument("--buffer_size", default=200_000, type=int)
p.add_argument("--batch_size",  default=512,     type=int)
p.add_argument("--gamma",       default=0.99,    type=float)
p.add_argument("--tau",         default=5e-3,    type=float)
p.add_argument("--actor_lr",    default=3e-4,    type=float)
p.add_argument("--critic_lr",   default=3e-4,    type=float)
p.add_argument("--alpha_lr",    default=3e-4,    type=float)
p.add_argument("--hidden_mul",  default=2,       type=int)

# runtime schedule
EPISODES, STEPS_PER_EP = 20, 10_000
p.add_argument("--update_every",      default=2, type=int)
p.add_argument("--updates_per_call",  default=1, type=int)

# plotting
p.add_argument("--plot_threshold",
               default=0.5, type=float,
               help="values below this are masked (not drawn) in the PNG")

# misc
p.add_argument("--seed", default=0, type=int)
p.add_argument("--gpu",  default="0")
args = p.parse_args()

# ───────── reproducibility --------------------------------------------- #
random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# ───────── build environment ------------------------------------------- #
to_lin = lambda dbw: 10 ** (dbw / 10)
env = env_mod.ISTNEnv(
    Ns=args.num_sat_antennas, Np=args.num_bs_antennas,
    max_FSS=args.max_FSS,     max_PU=args.max_PU,
    topo_hz=args.topo_hz,
    Ps_max=to_lin(23.01), Pb_max=to_lin(-22.0),
    sigma_s=to_lin(-126.47), sigma_p=to_lin(-121.52), sigma_e=to_lin(-121.52)
)

# ───────── agent & replay ---------------------------------------------- #
agent = SAC(
    state_dim=env.state_dim, action_dim=env.action_dim, device=device,
    hidden_mul=args.hidden_mul, gamma=args.gamma, tau=args.tau,
    actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr
)
replay = utils.ExperienceReplayBuffer(env.state_dim, env.action_dim,
                                      max_size=args.buffer_size)

# ───────── training loop ------------------------------------------------ #
warmup = 10_000
best_avg, best_ep_idx, best_trace = -np.inf, -1, []
total_steps = 0
tag = time.strftime("%Y%m%d_%H%M%S")

for ep in range(1, EPISODES + 1):
    state = whiten(env.reset())
    ep_sum, ep_trace = 0.0, []

    for t in range(1, STEPS_PER_EP + 1):
        # -- interact --------------------------------------------------- #
        action = agent.select_action(state, greedy=False)
        next_state, r_raw, _, _ = env.step(action)

        replay.add(state, action, next_state, r_raw, 1.0)
        ep_sum += r_raw
        ep_trace.append(r_raw)

        # per-step log
        print(f"[Ep {ep:02d}|{t:05d}] R={r_raw:8.3f}  α={agent.alpha:6.4f}", end="\r")

        # updates
        if replay.size > warmup and total_steps % args.update_every == 0:
            for _ in range(args.updates_per_call):
                batch = replay.sample(args.batch_size, device)
                agent.update(batch)

        state = whiten(next_state)
        total_steps += 1

    # episode summary
    ep_avg = ep_sum / STEPS_PER_EP
    print(f"\nEpisode {ep:02d} finished — average reward: {ep_avg:.3f} bit/s/Hz")
    if ep_avg > best_avg:
        best_avg, best_ep_idx, best_trace = ep_avg, ep, ep_trace.copy()

# ───────── save trace & meta for reproducibility ----------------------- #
fname_base = f"sac_best_episode_{tag}"
np.save(f"{fname_base}_trace.npy", np.array(best_trace))
with open(f"{fname_base}_meta.txt", "w") as f:
    json.dump({
        "episode_index": best_ep_idx,
        "average_reward": best_avg,
        "steps_per_episode": STEPS_PER_EP,
        "seed": args.seed,
        "plot_threshold": args.plot_threshold
    }, f, indent=2)

# ───────── masked plot -------------------------------------------------- #
trace_arr = np.array(best_trace)
masked = trace_arr.astype(float)
masked[masked < args.plot_threshold] = np.nan  # hide zeros / small values

plt.figure(figsize=(8, 4))
plt.plot(masked)
plt.xlabel("Step")
plt.ylabel("Raw secrecy capacity (bit/s/Hz)")
plt.title(f"SAC best episode #{best_ep_idx}  avg={best_avg:.2f}")
png = f"{fname_base}.png"
plt.tight_layout()
plt.savefig(png)

print(f"\nTraining complete. Best episode #{best_ep_idx} "
      f"avg = {best_avg:.2f} bit/s/Hz")
print(f"Plot saved to  {png}")
print(f"Trace saved to {fname_base}_trace.npy and meta.txt")

