"""
sac.py  —  Soft-Actor-Critic for the dynamic-topology ISTN environment.

• Diagonal-Gaussian tanh-squashed policy
• Twin Qs + target Q
• Automatic temperature tuning (α)
"""

from typing import Union, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────── helper ------------------------------------------------------- #
def mlp(inp: int, out: int, hidden: int, layers: int = 2) -> nn.Sequential:
    net, d = [], inp
    for _ in range(layers):
        net += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        d = hidden
    net.append(nn.Linear(hidden, out))
    return nn.Sequential(*net)


# ───────── networks ----------------------------------------------------- #
class Actor(nn.Module):
    def __init__(self, sdim: int, adim: int, hidden_mul: int = 2):
        super().__init__()
        hidden = 2 ** ((sdim - 1).bit_length()) * hidden_mul
        self.backbone = mlp(sdim, hidden, hidden, 2)
        self.mu_head = nn.Linear(hidden, adim)
        self.logstd_head = nn.Linear(hidden, adim)

    def forward(self, s: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(s)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.logstd_head(h), -5, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mu, std)
        u = mu if deterministic else dist.rsample()
        a = torch.tanh(u)
        logp = dist.log_prob(u).sum(1) - torch.log(1 - a.pow(2) + 1e-6).sum(1)
        return a, logp


class Critic(nn.Module):
    def __init__(self, sdim: int, adim: int, hidden_mul: int = 2):
        super().__init__()
        hidden = 2 ** (((sdim + adim) - 1).bit_length()) * hidden_mul
        self.Q1 = mlp(sdim + adim, 1, hidden, 2)
        self.Q2 = mlp(sdim + adim, 1, hidden, 2)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        sa = torch.cat([s, a], 1)
        return self.Q1(sa), self.Q2(sa)


# ───────── SAC wrapper -------------------------------------------------- #
class SAC:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 device: Union[str, torch.device] = "cpu",
                 hidden_mul: int = 2,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 5e-3,
                 target_entropy: Optional[float] = None):

        self.device = torch.device(device)
        self.actor = Actor(state_dim, action_dim, hidden_mul).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_mul).to(self.device)
        self.critic_t = Critic(state_dim, action_dim, hidden_mul).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.gamma, self.tau = gamma, tau

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # automatic entropy tuning
        if target_entropy is None:
            target_entropy = -action_dim
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    # property for current α
    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    # ---------- act ------------------------------------------------------ #
    @torch.no_grad()
    def select_action(self, s_np: np.ndarray, greedy: bool = False):
        s = torch.as_tensor(s_np.reshape(1, -1), dtype=torch.float32,
                            device=self.device)
        a, _ = self.actor(s, deterministic=greedy)
        return a.cpu().numpy().flatten()

    # ---------- update --------------------------------------------------- #
    def update(self, batch, alpha_auto: bool = True):
        s, a, s2, r, nd = batch  # all tensors already on device

        # ----- critic ---------------------------------------------------- #
        with torch.no_grad():
            a2, logp2 = self.actor(s2)
            q1_t, q2_t = self.critic_t(s2, a2)
            q_min = torch.min(q1_t, q2_t) - self.alpha * logp2.unsqueeze(1)
            target_q = r + nd * self.gamma * q_min

        q1, q2 = self.critic(s, a)
        loss_c = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        # ----- actor ----------------------------------------------------- #
        a_pi, logp_pi = self.actor(s)
        q1_pi, q2_pi  = self.critic(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_a = (self.alpha * logp_pi - q_pi.squeeze(1)).mean()
        self.actor_opt.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        # ----- temperature ---------------------------------------------- #
        if alpha_auto:
            loss_alpha = -(self.log_alpha *
                           (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            loss_alpha.backward()
            self.alpha_opt.step()

        # ----- soft-update target critic -------------------------------- #
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(),
                              self.critic_t.parameters()):
                p_t.mul_(1 - self.tau).add_(self.tau * p)

