#!/usr/bin/env python
# main_ppo.py – PPO on dynamic-topology ISTN with better defaults & per-step log
import argparse, os, time, random, math
import numpy as np, torch, matplotlib.pyplot as plt
import environment as env_mod
from ppo import PPO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def whiten(x): return (x - x.mean()) / (x.std() + 1e-12)

# ───────── CLI ---------------------------------------------------------- #
p = argparse.ArgumentParser()
p.add_argument("--num_sat_antennas", default=15, type=int)
p.add_argument("--num_bs_antennas",  default=16, type=int)
p.add_argument("--max_FSS",          default=10, type=int)
p.add_argument("--max_PU",           default=25, type=int)
p.add_argument("--topo_hz",          default=800, type=int)

# PPO knobs (already improved)
p.add_argument("--rollout_steps", default=8192, type=int)
p.add_argument("--lr",           default=1e-4, type=float)
p.add_argument("--epochs",       default=5,   type=int)
p.add_argument("--batch_size",   default=512, type=int)
p.add_argument("--hidden_mul",   default=2,   type=int)
p.add_argument("--ent_coef",     default=5e-3, type=float)

p.add_argument("--total_steps",  default=1_000_000, type=int)
p.add_argument("--seed", default=0, type=int)
p.add_argument("--gpu",  default="0")
args = p.parse_args()

# ───────── reproducibility --------------------------------------------- #
random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# ───────── env ---------------------------------------------------------- #
to_lin = lambda dbw: 10 ** (dbw / 10)
env = env_mod.ISTNEnv(
    Ns=args.num_sat_antennas, Np=args.num_bs_antennas,
    max_FSS=args.max_FSS,     max_PU=args.max_PU,
    topo_hz=args.topo_hz,
    Ps_max=to_lin(23.01), Pb_max=to_lin(-22.0),
    sigma_s=to_lin(-126.47), sigma_p=to_lin(-121.52), sigma_e=to_lin(-121.52)
)

# ───────── agent -------------------------------------------------------- #
agent = PPO(
    state_dim=env.state_dim, action_dim=env.action_dim,
    device=device,
    rollout_steps=args.rollout_steps,
    lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
    hidden_mul=args.hidden_mul, ent_coef=args.ent_coef
)

# ───────── training loop ----------------------------------------------- #
state = whiten(env.reset())
ep_sum, ep_trace, best_avg, best_idx = 0.0, [], -math.inf, -1
total_steps = 0; tag = time.strftime("%Y%m%d_%H%M%S")

while total_steps < args.total_steps:
    act, logp, val = agent.act(state)
    next_state, r_raw, _, _ = env.step(act)

    agent.store(state, act, logp, val, r_raw, 1.0)
    ep_sum += r_raw; ep_trace.append(r_raw)

    rollout_num = total_steps // args.rollout_steps + 1
    step_in_roll = total_steps % args.rollout_steps + 1
    print(f"[Ep {rollout_num:03d}|{step_in_roll:05d}] "
          f"R={r_raw:7.3f}", end="\r", flush=True)

    state = whiten(next_state)
    total_steps += 1

    # when rollout full -> update
    if agent.buf.full():
        _, _, boot_v = agent.act(state)
        agent.update(boot_v)

    # end-of-rollout logging
    if total_steps % args.rollout_steps == 0:
        avg = ep_sum / args.rollout_steps
        epi = total_steps // args.rollout_steps
        print(f"\nEpisode {epi:03d}  avg reward = {avg:.3f} bit/s/Hz")
        if avg > best_avg:
            best_avg, best_idx, best_trace = avg, epi, ep_trace.copy()
        ep_sum, ep_trace = 0.0, []

# ───────── plot best episode ------------------------------------------- #
plt.figure(figsize=(8,4))
plt.plot(best_trace)
plt.xlabel("Step"); plt.ylabel("Raw secrecy capacity (bit/s/Hz)")
plt.title(f"Best episode #{best_idx}  avg={best_avg:.2f}")
png = f"ppo_best_episode_{tag}.png"
plt.tight_layout(); plt.savefig(png)

print(f"\nTraining complete. Best avg = {best_avg:.2f} bit/s/Hz (episode {best_idx})")
print(f"Plot saved to {png}")
