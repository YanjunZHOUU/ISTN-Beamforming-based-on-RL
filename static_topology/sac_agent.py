"""
sac_agent.py
Soft-Actor-Critic implementation for the fixed-topology ISTN task.

Differences from the dynamic-topology version you pasted:
  • no topology parameters needed – state_dim & action_dim come from env
  • built-in replay buffer (ExperienceReplayBuffer) with uniform sampling
  • sample() returns torch tensors already on the chosen device
"""

from typing import Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


# ───────── replay buffer ─────────────────────────────────────────────── #
class Replay:
    def __init__(self, sdim: int, adim: int, size: int, device):
        self.ptr = 0
        self.size = 0
        self.max = size
        self.dev = device

        self.state      = np.zeros((size, sdim), dtype=np.float32)
        self.action     = np.zeros((size, adim), dtype=np.float32)
        self.next_state = np.zeros((size, sdim), dtype=np.float32)
        self.reward     = np.zeros((size, 1),   dtype=np.float32)
        self.not_done   = np.zeros((size, 1),   dtype=np.float32)

    def add(self, s, a, s2, r, done):
        self.state[self.ptr]      = s
        self.action[self.ptr]     = a
        self.next_state[self.ptr] = s2
        self.reward[self.ptr]     = r
        self.not_done[self.ptr]   = 1.0 - done
        self.ptr  = (self.ptr + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch):
        idx = np.random.randint(0, self.size, size=batch)
        to = lambda x: torch.as_tensor(x[idx]).to(self.dev)
        return (to(self.state), to(self.action),
                to(self.next_state), to(self.reward), to(self.not_done))


# ───────── helper MLP ---------------------------------------------------- #
def mlp(inp: int, out: int, hidden: int, layers: int = 2) -> nn.Sequential:
    net, d = [], inp
    for _ in range(layers):
        net += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        d = hidden
    net.append(nn.Linear(hidden, out))
    return nn.Sequential(*net)


# ───────── actor network ------------------------------------------------- #
class Actor(nn.Module):
    def __init__(self, sdim, adim):
        super().__init__()
        hid = 2 ** (sdim - 1).bit_length()
        self.back = mlp(sdim, hid, hid, 2)
        self.mu    = nn.Linear(hid, adim)
        self.log_s = nn.Linear(hid, adim)

    def forward(self, s, deterministic=False):
        h = self.back(s)
        mu, log_std = self.mu(h), torch.clamp(self.log_s(h), -5, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mu, std)
        u = mu if deterministic else dist.rsample()
        a = torch.tanh(u)

        logp = dist.log_prob(u).sum(1) - torch.log(1 - a.pow(2) + 1e-6).sum(1)
        return a, logp


# ───────── critic network (twin Q) -------------------------------------- #
class Critic(nn.Module):
    def __init__(self, sdim, adim):
        super().__init__()
        hid = 2 ** ((sdim + adim) - 1).bit_length()
        self.Q1 = mlp(sdim + adim, 1, hid, 2)
        self.Q2 = mlp(sdim + adim, 1, hid, 2)

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.Q1(sa), self.Q2(sa)


# ───────── SAC wrapper --------------------------------------------------- #
class SAC:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 device="cpu",
                 buffer_size=100_000,
                 batch_size=256,
                 gamma=0.99, tau=5e-3,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4):

        self.device = torch.device(device)
        self.batch  = batch_size
        self.gamma, self.tau = gamma, tau

        # networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_t = Critic(state_dim, action_dim).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -action_dim

        # replay buffer
        self.replay = Replay(state_dim, action_dim, buffer_size, self.device)

    # -------- property --------------------------------------------------- #
    @property
    def alpha(self): return self.log_alpha.exp()

    # -------- action ----------------------------------------------------- #
    @torch.no_grad()
    def act(self, s_np, greedy=False):
        s = torch.as_tensor(s_np.reshape(1, -1), dtype=torch.float32,
                            device=self.device)
        a, _ = self.actor(s, deterministic=greedy)
        return a.cpu().numpy().flatten()

    # -------- store transition ------------------------------------------ #
    def add(self, *args): self.replay.add(*args)

    # -------- update ----------------------------------------------------- #
    def update(self):
        if self.replay.size < self.batch:   # not enough data yet
            return

        s, a, s2, r, nd = self.replay.sample(self.batch)

        # --- critic ---
        with torch.no_grad():
            a2, logp2 = self.actor(s2)
            q1_t, q2_t = self.critic_t(s2, a2)
            q_min = torch.min(q1_t, q2_t) - self.alpha * logp2.unsqueeze(1)
            target_q = r + nd * self.gamma * q_min

        q1, q2 = self.critic(s, a)
        loss_c = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(); loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        # --- actor ---
        a_pi, logp_pi = self.actor(s)
        q1_pi, q2_pi = self.critic(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_a = (self.alpha * logp_pi - q_pi.squeeze(1)).mean()
        self.actor_opt.zero_grad(); loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        # --- temperature ---
        loss_alpha = -(self.log_alpha *
                       (logp_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); loss_alpha.backward(); self.alpha_opt.step()

        # --- target critic ---
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(),
                              self.critic_t.parameters()):
                p_t.mul_(1 - self.tau).add_(self.tau * p)

    # -------- save ------------------------------------------------------- #
    def save(self, prefix: str):
        torch.save(self.actor.state_dict(),  f"{prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{prefix}_critic.pt")
