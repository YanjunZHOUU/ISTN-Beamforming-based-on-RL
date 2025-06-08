"""
TD3 agent with stabilisers:
  • smaller policy noise (0.05,0.1)
  • gradient clipping on critic
"""

import copy, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np


def mlp(inp, out, hid, layers=2):
    seq, d = [], inp
    for _ in range(layers):
        seq += [nn.Linear(d, hid), nn.LayerNorm(hid), nn.GELU()]
        d = hid
    seq += [nn.Linear(hid, out)]
    return nn.Sequential(*seq)


class Actor(nn.Module):
    def __init__(self, sdim, Ns, Nf, Np, M, Ps, Pb, max_act=1.):
        super().__init__()
        self.Ns, self.Nf, self.Np, self.M, self.max_act = Ns, Nf, Np, M, max_act
        self.register_buffer("Ps_max", torch.tensor(Ps, dtype=torch.float32))
        self.register_buffer("Pb_max", torch.tensor(Pb, dtype=torch.float32))
        self.adim = 2 * (Ns * Nf + Np * M)
        hidden = 2 ** (sdim - 1).bit_length()
        self.net = mlp(sdim, self.adim, hidden, 2)

    @staticmethod
    def _pow(b):
        h = b.size(1) // 2
        return torch.sum(b[:, :h]**2 + b[:, h:]**2, 1, keepdim=True)

    def forward(self, s):
        x = torch.tanh(self.net(s)) * self.max_act
        slen = 2 * self.Ns * self.Nf
        sb, bb = x[:, :slen], x[:, slen:]
        sb = sb * torch.sqrt(self.Ps_max / (self._pow(sb) + 1e-12))
        bb = bb * torch.sqrt(self.Pb_max / (self._pow(bb) + 1e-12))
        return torch.cat([sb, bb], 1)


class Critic(nn.Module):
    def __init__(self, sdim, adim):
        super().__init__()
        h = 2 ** ((sdim + adim) - 1).bit_length()
        self.Q1 = mlp(sdim + adim, 1, h, 2)
        self.Q2 = mlp(sdim + adim, 1, h, 2)

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.Q1(sa), self.Q2(sa)


class TD3:
    def __init__(self, state_dim, Ns, Nf, Np, M, Ps, Pb, max_action,
                 actor_lr, critic_lr, actor_decay, critic_decay, device,
                 discount=0.99, tau=5e-3, policy_noise=0.05,
                 noise_clip=0.1, policy_delay=2):

        self.device, self.gamma, self.tau = device, discount, tau
        self.pn, self.nc, self.pd = policy_noise, noise_clip, policy_delay
        self.max_action, self.it = max_action, 0

        self.actor = Actor(state_dim, Ns, Nf, Np, M, Ps, Pb, max_action).to(device)
        self.actor_t = copy.deepcopy(self.actor)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_decay)

        adim = 2 * (Ns * Nf + Np * M)
        self.critic = Critic(state_dim, adim).to(device)
        self.critic_t = copy.deepcopy(self.critic)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_decay)

    def select_action(self, s):
        with torch.no_grad():
            return self.actor(torch.FloatTensor(s.reshape(1, -1)).to(self.device)).cpu().numpy().flatten()

    def update(self, buf, bs):
        self.it += 1
        s, a, s2, r, d = buf.sample(bs, self.device)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.pn).clamp(-self.nc, self.nc)
            a2 = (self.actor_t(s2) + noise).clamp(-self.max_action, self.max_action)
            tq1, tq2 = self.critic_t(s2, a2)
            tq = torch.min(tq1, tq2)
            target = r + d * self.gamma * tq

        q1, q2 = self.critic(s, a)
        loss_c = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.c_opt.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.c_opt.step()

        if self.it % self.pd == 0:
            loss_a = -self.critic.Q1(torch.cat([s, self.actor(s)], 1)).mean()
            self.a_opt.zero_grad()
            loss_a.backward()
            self.a_opt.step()

            # soft update
            for p, p_t in zip(self.actor.parameters(), self.actor_t.parameters()):
                p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, p_t in zip(self.critic.parameters(), self.critic_t.parameters()):
                p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)

    # save / load -------------------------------------------------
    def save(self, prefix):
        torch.save(self.actor.state_dict(),  f"{prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{prefix}_critic.pt")
