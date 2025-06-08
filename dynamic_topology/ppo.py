"""
ppo.py – Clipped-PPO with GAE for the ISTN environment.

Changes vs. minimal version
---------------------------
* hidden size can be scaled with `hidden_mul`
* entropy bonus (`ent_coef`) exposed
* supports reward normalisation via running mean / std
"""

from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────── small helpers ─────────────────────────────────────────────── #
class RunningMeanStd:
    """Track running mean / variance for reward normalisation."""
    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var  = 1.0
        self.count = eps

    def update(self, x: np.ndarray):
        batch_mean = x.mean()
        batch_var  = x.var()
        batch_count = x.size

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    @property
    def std(self): return np.sqrt(self.var + 1e-6)


def mlp(inp: int, out: int, hidden: int, layers: int) -> nn.Sequential:
    blocks, d = [], inp
    for _ in range(layers):
        blocks += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        d = hidden
    blocks.append(nn.Linear(hidden, out))
    return nn.Sequential(*blocks)


# ───────── rollout buffer ────────────────────────────────────────────── #
class RolloutBuffer:
    """Store N transitions + bootstrap slot for value, not_done."""
    def __init__(self, steps, sdim, adim, device):
        self.steps, self.device = steps, device
        self.reset(sdim, adim)

    def reset(self, sdim, adim):
        self.ptr = 0
        self.s  = torch.zeros((self.steps, sdim), device=self.device)
        self.a  = torch.zeros((self.steps, adim), device=self.device)
        self.logp = torch.zeros(self.steps, device=self.device)
        self.val  = torch.zeros(self.steps + 1, device=self.device)
        self.r    = torch.zeros(self.steps, device=self.device)
        self.nd   = torch.zeros(self.steps + 1, device=self.device)

    def add(self, s, a, logp, v, r, nd):
        i = self.ptr
        self.s[i], self.a[i] = s, a
        self.logp[i], self.val[i], self.r[i], self.nd[i] = logp, v, r, nd
        self.ptr += 1

    def full(self): return self.ptr == self.steps

    def advantages(self, γ, lam):
        adv = torch.zeros_like(self.r, device=self.device)
        gae = 0.0
        for t in reversed(range(self.steps)):
            delta = self.r[t] + γ * self.val[t+1] * self.nd[t] - self.val[t]
            gae = delta + γ * lam * self.nd[t] * gae
            adv[t] = gae
        ret = adv + self.val[:self.steps]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.adv, self.ret = adv, ret

    def minibatches(self, bsz):
        idx = np.random.permutation(self.steps)
        for i in range(0, self.steps, bsz):
            j = idx[i:i+bsz]
            yield self.s[j], self.a[j], self.logp[j], self.adv[j], self.ret[j]


# ───────── actor-critic ──────────────────────────────────────────────── #
class ActorCritic(nn.Module):
    def __init__(self, sdim, adim, hidden_mul=1):
        super().__init__()
        hidden = 2 ** ((sdim - 1).bit_length()) * hidden_mul
        self.actor  = mlp(sdim, adim, hidden, 3)
        self.critic = mlp(sdim, 1,   hidden, 3)
        self.log_std = nn.Parameter(torch.zeros(adim))

    def act(self, s: torch.Tensor):
        mu  = torch.tanh(self.actor(s))
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        a = dist.rsample()
        return a, dist.log_prob(a).sum(1), mu, std

    def evaluate(self, s, a):
        mu  = torch.tanh(self.actor(s))
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(a).sum(1)
        ent  = dist.entropy().sum(1)
        v = self.critic(s).squeeze(1)
        return logp, ent, v


# ───────── PPO agent ─────────────────────────────────────────────────── #
class PPO:
    def __init__(self,
                 state_dim, action_dim,
                 device="cpu",
                 rollout_steps=8192,
                 lr=1e-4,
                 gamma=0.99, lam=0.97,
                 clip_eps=0.1,
                 ent_coef=5e-3,
                 vf_coef=0.5,
                 hidden_mul=2,
                 epochs=5, batch_size=512):

        self.device = torch.device(device)
        self.net = ActorCritic(state_dim, action_dim, hidden_mul).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)

        self.buf = RolloutBuffer(rollout_steps, state_dim, action_dim, self.device)
        self.rms = RunningMeanStd()          # for reward normalisation
        self.steps = rollout_steps

        self.gam, self.lam = gamma, lam
        self.clip_eps = clip_eps
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.epochs, self.batch = epochs, batch_size

    # ---------- sampling ------------------------------------------------- #
    @torch.no_grad()
    def act(self, s_np):
        s = torch.as_tensor(s_np, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        a, logp, _, _ = self.net.act(s)
        v = self.net.critic(s).squeeze(1)
        return a.cpu().numpy().flatten(), logp.item(), v.item()

    def store(self, s, a, logp, v, r, nd):
        self.rms.update(np.array([r]))              # update running stats
        r_norm = (r - self.rms.mean) / self.rms.std
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        self.buf.add(s, a,
                     torch.tensor([logp], device=self.device),
                     torch.tensor([v],    device=self.device),
                     torch.tensor([r_norm], device=self.device),
                     torch.tensor([nd],   device=self.device))

    # ---------- update --------------------------------------------------- #
    def update(self, bootstrap_v):
        self.buf.val[self.steps] = bootstrap_v
        self.buf.nd [self.steps] = 0.0
        self.buf.advantages(self.gam, self.lam)

        for _ in range(self.epochs):
            for s, a, logp_old, adv, ret in self.buf.minibatches(self.batch):
                logp, ent, v = self.net.evaluate(s, a)
                ratio = torch.exp(logp - logp_old)

                pg = torch.min(ratio * adv,
                               torch.clamp(ratio, 1 - self.clip_eps,
                                           1 + self.clip_eps) * adv).mean()
                v_loss = F.mse_loss(v, ret)
                loss = -pg + self.vf_coef * v_loss - self.ent_coef * ent.mean()

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                self.opt.step()

        self.buf.reset(self.buf.s.shape[1], self.buf.a.shape[1])

