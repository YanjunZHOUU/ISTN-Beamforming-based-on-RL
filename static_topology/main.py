# import argparse
# import os
#
# import numpy as np
# import torch
#
# import DDPG
# import utils
#
# import environment
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# def whiten(state):
#     return (state - np.mean(state)) / np.std(state)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # Choose the type of the experiment
#     parser.add_argument('--experiment_type', default='custom', choices=['custom', 'power', 'rsi_elements', 'learning_rate', 'decay'],
#                         help='Choose one of the experiment types to reproduce the learning curves given in the paper')
#
#     # Training-specific parameters
#     parser.add_argument("--policy", default="DDPG", help='Algorithm (default: DDPG)')
#     parser.add_argument("--env", default="RIS_MISO", help='OpenAI Gym environment name')
#     parser.add_argument("--seed", default=0, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
#     parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
#     parser.add_argument("--start_time_steps", default=0, type=int, metavar='N', help='Number of exploration time steps sampling random actions (default: 0)')
#     parser.add_argument("--buffer_size", default=100000, type=int, help='Size of the experience replay buffer (default: 100000)')
#     parser.add_argument("--batch_size", default=16, metavar='N', help='Batch size (default: 16)')
#     parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
#     parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')
#
#     # Environment-specific parameters
#     parser.add_argument("--num_sat_antennas", default=4, type=int, metavar='N', help='Number of antennas in the SAT')
#     parser.add_argument("--num_bs_antennas", default=4, type=int, metavar='N', help='Number of antennas in the BS')
#     # parser.add_argument("--num_RIS_elements", default=4, type=int, metavar='N', help='Number of RIS elements')
#     parser.add_argument("--num_FSS", default=4, type=int, metavar='N', help='Number of FSS')
#     parser.add_argument("--num_users", default=4, type=int, metavar='N', help='Number of users')
#     parser.add_argument("--power_t", default=30, type=float, metavar='N', help='Transmission power for the constrained optimization in dB')
#     parser.add_argument("--num_time_steps_per_eps", default=10000, type=int, metavar='N', help='Maximum number of steps per episode (default: 20000)')
#     parser.add_argument("--num_eps", default=10, type=int, metavar='N', help='Maximum number of episodes (default: 5000)')
#     parser.add_argument("--awgn_var", default=1e-2, type=float, metavar='G', help='Variance of the additive white Gaussian noise (default: 0.01)')
#     parser.add_argument("--channel_est_error", default=False, type=bool, help='Noisy channel estimate? (default: False)')
#
#     # Algorithm-specific parameters
#     parser.add_argument("--exploration_noise", default=0.0, metavar='G', help='Std of Gaussian exploration noise')
#     parser.add_argument("--discount", default=0.99, metavar='G', help='Discount factor for reward (default: 0.99)')
#     parser.add_argument("--tau", default=1e-3, type=float, metavar='G',  help='Learning rate in soft/hard updates of the target networks (default: 0.001)')
#     parser.add_argument("--lr", default=1e-3, type=float, metavar='G', help='Learning rate for the networks (default: 0.001)')
#     parser.add_argument("--decay", default=1e-5, type=float, metavar='G', help='Decay rate for the networks (default: 0.00001)')
#
#     args = parser.parse_args()
#
#     print("---------------------------------------")
#     print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
#     print("---------------------------------------")
#
#     file_name = f"{args.num_sat_antennas}_{args.num_bs_antennas}_{args.num_FSS}_{args.num_users}_{args.power_t}_{args.lr}_{args.decay}"
#
#     if not os.path.exists(f"./Learning Curves/{args.experiment_type}"):
#         os.makedirs(f"./Learning Curves/{args.experiment_type}")
#
#     if args.save_model and not os.path.exists("./Models"):
#         os.makedirs("./Models")
#
#     # env = environment.RIS_MISO(args.num_antennas, args.num_RIS_elements, args.num_users, AWGN_var=args.awgn_var)
#     env = environment.RIS_MISO(args.num_sat_antennas, args.num_bs_antennas,args.num_FSS, args.num_users)
#
#     # num_sat_antennas: int,  # N_s
#     # num_bs_antennas: int,  # N_p
#     # num_FSS: int,  # N_f
#     # num_PU: int,
#
#     # Set seeds
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#
#     state_dim = env.state_dim
#     action_dim = env.action_dim
#     max_action = 1
#
#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#
#     kwargs = {
#         "state_dim": state_dim,
#         "action_dim": action_dim,
#         "power_t": args.power_t,
#         "max_action": max_action,
#         "M": args.num_antennas,
#         "N": args.num_RIS_elements,
#         "K": args.num_users,
#         "actor_lr": args.lr,
#         "critic_lr": args.lr,
#         "actor_decay": args.decay,
#         "critic_decay": args.decay,
#         "device": device,
#         "discount": args.discount,
#         "tau": args.tau
#     }
#
#     # Initialize the algorithm
#     agent = DDPG.DDPG(**kwargs)
#
#     if args.load_model != "":
#         policy_file = file_name if args.load_model == "default" else args.load_model
#         agent.load(f"./models/{policy_file}")
#
#     replay_buffer = utils.ExperienceReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)
#
#     # Initialize the instant rewards recording array
#     instant_rewards = []
#
#     max_reward = 0
#     episode_num = 0
#
#     for eps in range(int(args.num_eps)):
#         state, done = env.reset(), False
#         episode_reward = 0
#
#         episode_time_steps = 0
#
#         state = whiten(state)
#
#         eps_rewards = []
#
#         for t in range(int(args.num_time_steps_per_eps)):
#             # Choose action from the policy
#             action = agent.select_action(np.array(state))
#
#             # Take the selected action
#             next_state, reward, done, _ = env.step(action)
#
#             done = 1.0 if t == args.num_time_steps_per_eps - 1 else float(done)
#
#             # Store data in the experience replay buffer
#             replay_buffer.add(state, action, next_state, reward, done)
#
#             state = next_state
#             episode_reward += reward
#
#             state = whiten(state)
#
#             if reward > max_reward:
#                 max_reward = reward
#
#             # Train the agent
#             agent.update_parameters(replay_buffer, args.batch_size)
#
#             print(f"Time step: {t + 1} Episode Num: {episode_num + 1} Reward: {reward:.3f}")
#
#             eps_rewards.append(reward)
#
#             episode_time_steps += 1
#
#             if done:
#                 print(f"\nTotal T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_time_steps} Max. Reward: {max_reward:.3f}\n")
#
#                 # Reset the environment
#                 state, done = env.reset(), False
#                 episode_reward = 0
#                 episode_time_steps = 0
#                 episode_num += 1
#
#                 state = whiten(state)
#
#                 instant_rewards.append(eps_rewards)
#
#                 np.save(f"./Learning Curves/{args.experiment_type}/{file_name}_episode_{episode_num + 1}", instant_rewards)
#
#     if args.save_model:
#         agent.save(f"./Models/{file_name}")







# import argparse, os, time, numpy as np, torch
#
# import environment
# import agent_td3 as rl
# import utils
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
#
# def whiten(x): return (x - x.mean()) / (x.std() + 1e-12)
#
#
# # ─────────── command-line flags ────────────────────────────────
# p = argparse.ArgumentParser()
#
# # topology
# p.add_argument("--num_sat_antennas", default=15, type=int)
# p.add_argument("--num_bs_antennas",  default=16, type=int)
# p.add_argument("--num_FSS",          default=5,  type=int)
# p.add_argument("--num_users",        default=15, type=int)
#
# # power in dBW (defaults from Table II)
# p.add_argument("--sat_power_dBW", default=23.01, type=float)
# p.add_argument("--bs_power_dBW",  default=-22.0, type=float)
#
# # noise power in dBW
# p.add_argument("--noise_s_dBW", default=-126.47, type=float)
# p.add_argument("--noise_p_dBW", default=-121.52, type=float)
# p.add_argument("--noise_e_dBW", default=-121.52, type=float)
#
# # RL hyper-params
# p.add_argument("--buffer_size",  default=100_000, type=int)
# p.add_argument("--batch_size",   default=256,    type=int)
# p.add_argument("--num_eps",      default=20,     type=int)
# p.add_argument("--steps_per_eps",default=10_000, type=int)
# p.add_argument("--lr",           default=1e-3,   type=float)
# p.add_argument("--decay",        default=1e-5,   type=float)
# p.add_argument("--discount",     default=0.99,   type=float)
# p.add_argument("--tau",          default=5e-3,   type=float)
#
# # exploration & misc
# p.add_argument("--exploration_noise", default=0.1, type=float)
# p.add_argument("--seed", default=0, type=int)
# p.add_argument("--gpu",  default="0")
# p.add_argument("--save_model", action="store_true")
# p.add_argument("--model_dir",  default="./Models")
#
# args = p.parse_args()
#
#
# # ─────────── setup ---------------------------------------------------------
# device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(args.seed); np.random.seed(args.seed)
#
# Ps_lin = 10 ** (args.sat_power_dBW / 10)
# Pp_lin = 10 ** (args.bs_power_dBW  / 10)
# sigma_s = 10 ** (args.noise_s_dBW / 10)
# sigma_p = 10 ** (args.noise_p_dBW / 10)
# sigma_e = 10 ** (args.noise_e_dBW / 10)
#
# env = environment.ISTNEnv(
#     N_s=args.num_sat_antennas,
#     N_p=args.num_bs_antennas,
#     N_f=args.num_FSS,
#     M  =args.num_users,
#     Ps_max=Ps_lin, Pb_max=Pp_lin,
#     sigma_s=sigma_s, sigma_p=sigma_p, sigma_e=sigma_e
# )
#
# agent = rl.TD3(
#     state_dim=env.state_dim,
#     Ns=args.num_sat_antennas, Nf=args.num_FSS,
#     Np=args.num_bs_antennas,  M=args.num_users,
#     Ps=Ps_lin, Pb=Pp_lin,
#     max_action=1.0,
#     actor_lr=args.lr, critic_lr=args.lr,
#     actor_decay=args.decay, critic_decay=args.decay,
#     device=device, discount=args.discount, tau=args.tau
# )
#
# replay = utils.ExperienceReplayBuffer(env.state_dim, env.action_dim, args.buffer_size)
#
# # bookkeeping
# best_raw = -np.inf
# if args.save_model:
#     os.makedirs(args.model_dir, exist_ok=True)
#     tag = time.strftime("%Y%m%d_%H%M%S")
#
# global_step, warmup = 0, 5000
#
# for ep in range(args.num_eps):
#     s = whiten(env.reset())
#     for t in range(args.steps_per_eps):
#         sigma = max(0.05, args.exploration_noise * (1 - global_step / 600_000))
#         a = agent.select_action(s) + np.random.normal(0, sigma, size=env.action_dim)
#         a = np.clip(a, -1.0, 1.0)
#
#         s2, r_raw, _, _ = env.step(a)
#         replay.add(s, a, s2, r_raw / 5.0, 0.0)
#         s = whiten(s2)
#
#         if replay.size > warmup:
#             agent.update(replay, args.batch_size)
#
#         if r_raw > best_raw:
#             best_raw = r_raw
#             if args.save_model:
#                 path = f"{args.model_dir}/ISTN_TD3_best_{tag}_{best_raw:.3f}"
#                 agent.save(path)
#                 print(f"\n>> new BEST {best_raw:.3f}  saved to {path}")
#
#         print(f"[Ep {ep+1:03d}|{t+1:05d}] R={r_raw:5.2f}  "
#               f"σ={sigma:4.2f}  best={best_raw:5.2f}", end="\r")
#         global_step += 1
#     print()
#
# print(f"\nFinished. Best single-step secrecy capacity: {best_raw:.3f} bit/s/Hz")





import argparse, os, time, random
import numpy as np, torch
import matplotlib.pyplot as plt

import environment, agent_td3 as rl, utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def whiten(x): return (x - x.mean()) / (x.std() + 1e-12)


# ---------------- CLI ------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--num_sat_antennas", default=15, type=int)
p.add_argument("--num_bs_antennas",  default=16, type=int)
p.add_argument("--num_FSS",          default=5,  type=int)
p.add_argument("--num_users",        default=15, type=int)
p.add_argument("--sat_power_dBW", default=23.01,  type=float)
p.add_argument("--bs_power_dBW",  default=-22.0,  type=float)
p.add_argument("--noise_s_dBW",   default=-126.47,type=float)
p.add_argument("--noise_p_dBW",   default=-121.52,type=float)
p.add_argument("--noise_e_dBW",   default=-121.52,type=float)
p.add_argument("--buffer_size",  default=100_000, type=int)
p.add_argument("--batch_size",   default=256,    type=int)
p.add_argument("--num_eps",      default=20,     type=int)
p.add_argument("--steps_per_eps",default=10_000, type=int)
p.add_argument("--lr",           default=1e-3,   type=float)
p.add_argument("--decay",        default=1e-5,   type=float)
p.add_argument("--discount",     default=0.99,   type=float)
p.add_argument("--tau",          default=5e-3,   type=float)
p.add_argument("--exploration_noise", default=0.1, type=float)
p.add_argument("--seed", default=0, type=int)
p.add_argument("--gpu",  default="0")
p.add_argument("--save_model", action="store_true")
p.add_argument("--model_dir",  default="./Models")
args = p.parse_args()

# ---------------- reproducibility -----------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# convert dBW → W
to_lin = lambda dbw: 10 ** (dbw / 10)
Ps_lin, Pb_lin = to_lin(args.sat_power_dBW), to_lin(args.bs_power_dBW)
sig_s, sig_p, sig_e = map(to_lin, (args.noise_s_dBW,
                                   args.noise_p_dBW,
                                   args.noise_e_dBW))

env = environment.ISTNEnv(
    N_s=args.num_sat_antennas, N_p=args.num_bs_antennas,
    N_f=args.num_FSS,          M=args.num_users,
    Ps_max=Ps_lin, Pb_max=Pb_lin,
    sigma_s=sig_s, sigma_p=sig_p, sigma_e=sig_e
)

agent = rl.TD3(
    env.state_dim,
    args.num_sat_antennas, args.num_FSS,
    args.num_bs_antennas,  args.num_users,
    Ps_lin, Pb_lin, 1.0,
    args.lr, args.lr, args.decay, args.decay,
    device, discount=args.discount, tau=args.tau
)

replay = utils.ExperienceReplayBuffer(env.state_dim, env.action_dim, args.buffer_size)

# logging containers
episode_log = []            # list of per-step lists
best_raw, best_ep = -np.inf, -1

if args.save_model:
    os.makedirs(args.model_dir, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")

global_step, warmup = 0, 5000

for ep in range(args.num_eps):
    s = whiten(env.reset())
    ep_trace, ep_sum = [], 0.0

    for t in range(args.steps_per_eps):
        sigma = max(0.05, args.exploration_noise * (1 - global_step / 600_000))
        a = agent.select_action(s) + np.random.normal(0, sigma, env.action_dim)
        a = np.clip(a, -1.0, 1.0)

        s2, r_raw, _, _ = env.step(a)
        ep_trace.append(r_raw); ep_sum += r_raw

        replay.add(s, a, s2, r_raw / 5.0, 0.0)
        s = whiten(s2)

        if replay.size > warmup:
            agent.update(replay, args.batch_size)

        if r_raw > best_raw:

            if args.save_model and last_ckpt_prefix is not None:
                for suff in ("_actor.pt", "_critic.pt"):
                    pth = f"{last_ckpt_prefix}{suff}"
                    if os.path.exists(pth):
                        os.remove(pth)


            best_raw = r_raw
            best_ep = ep
            if args.save_model:
                ckpt = f"{args.model_dir}/best_step_{tag}_{best_raw:.3f}"
                agent.save(ckpt)

        print(f"[Ep {ep+1:03d}|{t+1:05d}] R={r_raw:5.2f} σ={sigma:4.2f} best={best_raw:5.2f}", end="\r")
        global_step += 1
    print()

    episode_log.append(ep_trace)
    # if ep_sum > (sum(episode_log[best_ep]) if best_ep >= 0 else -np.inf):
    #     best_ep = ep

# --------------- plot best episode ---------------------------------------
best_trace = episode_log[best_ep]
plt.figure(figsize=(8, 4))
plt.plot(best_trace)
plt.xlabel("Step")
plt.ylabel("Raw secrecy capacity (bit/s/Hz)")
plt.title(f"Best episode #{best_ep+1}  (best = {best_raw:.1f} bit/s/Hz)")
png_path = f"best_episode_{tag}.png"
plt.tight_layout(); plt.savefig(png_path)
print(f"\nFinished. Best episode: {best_ep+1}  |  cumulative reward = {sum(best_trace):.1f}")
print(f"PNG saved to: {png_path}")
