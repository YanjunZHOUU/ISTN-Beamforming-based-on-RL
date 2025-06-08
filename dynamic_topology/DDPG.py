# import copy
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# import numpy as np
#
#
# # Implementation of the Deep Deterministic Policy Gradient algorithm (DDPG)
# # Paper: https://arxiv.org/abs/1509.02971
#
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, M, N, K, power_t, device, max_action=1):
#         super(Actor, self).__init__()
#         hidden_dim = 1 if state_dim == 0 else 2 ** (state_dim - 1).bit_length()
#
#         self.device = device
#
#         self.M = M
#         self.N = N
#         self.K = K
#         self.power_t = power_t
#
#         self.l1 = nn.Linear(state_dim, hidden_dim)
#         self.l2 = nn.Linear(hidden_dim, hidden_dim)
#         self.l3 = nn.Linear(hidden_dim, action_dim)
#
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#
#         self.max_action = max_action
#
#     def compute_power(self, a):
#         # Normalize the power
#         G_real = a[:, :self.M ** 2].cpu().data.numpy()
#         G_imag = a[:, self.M ** 2:2 * self.M ** 2].cpu().data.numpy()
#
#         G = G_real.reshape(G_real.shape[0], self.M, self.K) + 1j * G_imag.reshape(G_imag.shape[0], self.M, self.K)
#
#         GG_H = np.matmul(G, np.transpose(G.conj(), (0, 2, 1)))
#
#         current_power_t = torch.sqrt(torch.from_numpy(np.real(np.trace(GG_H, axis1=1, axis2=2)))).reshape(-1, 1).to(self.device)
#
#         return current_power_t
#
#     def compute_phase(self, a):
#         # Normalize the phase matrix
#         Phi_real = a[:, -2 * self.N:-self.N].detach()
#         Phi_imag = a[:, -self.N:].detach()
#
#         return torch.sum(torch.abs(Phi_real), dim=1).reshape(-1, 1) * np.sqrt(2), torch.sum(torch.abs(Phi_imag), dim=1).reshape(-1, 1) * np.sqrt(2)
#
#     def forward(self, state):
#         a = torch.tanh(self.l1(state.float()))
#
#         # Apply batch normalization to the each hidden layer's input
#         a = self.bn1(a)
#         a = torch.tanh(self.l2(a))
#
#         a = self.bn2(a)
#         a = torch.tanh(self.l3(a))
#
#         # Normalize the transmission power and phase matrix
#         current_power_t = self.compute_power(a.detach()).expand(-1, 2 * self.M ** 2) / np.sqrt(self.power_t)
#
#         real_normal, imag_normal = self.compute_phase(a.detach())
#
#         real_normal = real_normal.expand(-1, self.N)
#         imag_normal = imag_normal.expand(-1, self.N)
#
#         division_term = torch.cat([current_power_t, real_normal, imag_normal], dim=1)
#
#         return self.max_action * a / division_term
#
#
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         hidden_dim = 1 if (state_dim + action_dim) == 0 else 2 ** ((state_dim + action_dim) - 1).bit_length()
#
#         self.l1 = nn.Linear(state_dim, hidden_dim)
#         self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
#         self.l3 = nn.Linear(hidden_dim, 1)
#
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#
#     def forward(self, state, action):
#         q = torch.tanh(self.l1(state.float()))
#
#         q = self.bn1(q)
#         q = torch.tanh(self.l2(torch.cat([q, action], 1)))
#
#         q = self.l3(q)
#
#         return q
#
#
# class DDPG(object):
#     def __init__(self, state_dim, action_dim, M, N, K, power_t, max_action, actor_lr, critic_lr, actor_decay, critic_decay, device, discount=0.99, tau=0.001):
#         self.device = device
#
#         powert_t_W = 10 ** (power_t / 10)
#
#         # Initialize actor networks and optimizer
#         self.actor = Actor(state_dim, action_dim, M, N, K, powert_t_W, max_action=max_action, device=device).to(self.device)
#         self.actor_target = copy.deepcopy(self.actor)
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_decay)
#
#         # Initialize critic networks and optimizer
#         self.critic = Critic(state_dim, action_dim).to(self.device)
#         self.critic_target = copy.deepcopy(self.critic)
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_decay)
#
#         # Initialize the discount and target update rated
#         self.discount = discount
#         self.tau = tau
#
#     def select_action(self, state):
#         self.actor.eval()
#
#         state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
#         action = self.actor(state).cpu().data.numpy().flatten().reshape(1, -1)
#
#         return action
#
#     def update_parameters(self, replay_buffer, batch_size=16):
#         self.actor.train()
#
#         # Sample from the experience replay buffer
#         state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
#
#         # Compute the target Q-value
#         target_Q = self.critic_target(next_state, self.actor_target(next_state))
#         target_Q = reward + (not_done * self.discount * target_Q).detach()
#
#         # Get the current Q-value estimate
#         current_Q = self.critic(state, action)
#
#         # Compute the critic loss
#         critic_loss = F.mse_loss(current_Q, target_Q)
#
#         # Optimize the critic
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()
#
#         # Compute the actor loss
#         actor_loss = -self.critic(state, self.actor(state)).mean()
#
#         # Optimize the actor
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()
#
#         # Soft update the target networks
#         for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#
#         for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#
#     # Save the model parameters
#     def save(self, file_name):
#         torch.save(self.critic.state_dict(), file_name + "_critic")
#         torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")
#
#         torch.save(self.actor.state_dict(), file_name + "_actor")
#         torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")
#
#     # Load the model parameters
#     def load(self, file_name):
#         self.critic.load_state_dict(torch.load(file_name + "_critic"))
#         self.critic_optimizer.load_state_dict(torch.load(file_name + "_critic_optimizer"))
#         self.critic_target = copy.deepcopy(self.critic)
#
#         self.actor.load_state_dict(torch.load(file_name + "_actor"))
#         self.actor_optimizer.load_state_dict(torch.load(file_name + "_actor_optimizer"))
#         self.actor_target = copy.deepcopy(self.actor)


"""
DDPG.py  —  On-policy deterministic actor-critic for the dynamic-topology ISTN.

Works with Python 3.7+ (no 3.10 `|` union types).
"""
import copy
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────── helper MLP ---------------------------------------------------- #
def mlp(inp: int, out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.LayerNorm(hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        nn.Linear(hidden, out)
    )

# ───────── actor --------------------------------------------------------- #
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        hidden = 2 ** ((state_dim - 1).bit_length())
        self.net = mlp(state_dim, action_dim, hidden)
        self.max_action = max_action

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(s)) * self.max_action

# ───────── critic -------------------------------------------------------- #
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hidden = 2 ** (((state_dim + action_dim) - 1).bit_length())
        self.net = mlp(state_dim + action_dim, 1, hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=1))

# ───────── on-policy DDPG wrapper --------------------------------------- #
class OnPolicyDDPG:
    """
    Updates immediately from each (s, a, r, s′) transition (no replay buffer).
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 actor_lr: float = 1e-3,
                 critic_lr: float = 1e-3,
                 actor_decay: float = 1e-5,
                 critic_decay: float = 1e-5,
                 max_action: float = 1.0,
                 discount: float = 0.99,
                 tau: float = 5e-3,
                 device: Union[str, torch.device] = "cpu"):

        self.device = torch.device(device)
        self.actor         = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic        = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.a_opt = torch.optim.Adam(self.actor.parameters(),
                                      lr=actor_lr, weight_decay=actor_decay)
        self.c_opt = torch.optim.Adam(self.critic.parameters(),
                                      lr=critic_lr, weight_decay=critic_decay)

        self.gamma = discount
        self.tau   = tau
        self.max_action = max_action

    # ---------- act (no grad) ------------------------------------------- #
    @torch.no_grad()
    def select_action(self, state_np: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(state_np.reshape(1, -1),
                            dtype=torch.float32, device=self.device)
        return self.actor(s).cpu().numpy().flatten()

    # ---------- single on-policy update --------------------------------- #
    def train_step(self,
                   s_np: np.ndarray,
                   a_np: np.ndarray,
                   r: float,
                   s2_np: np.ndarray,
                   done: float):

        s  = torch.as_tensor(s_np.reshape(1, -1), dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(a_np.reshape(1, -1), dtype=torch.float32, device=self.device)
        r  = torch.as_tensor([[r]], dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2_np.reshape(1, -1), dtype=torch.float32, device=self.device)
        not_done = torch.as_tensor([[1.0 - done]], dtype=torch.float32, device=self.device)

        # target Q
        with torch.no_grad():
            a2 = self.actor_target(s2)
            target_Q = r + not_done * self.gamma * self.critic_target(s2, a2)

        # critic update
        current_Q = self.critic(s, a)
        loss_c = F.mse_loss(current_Q, target_Q)
        self.c_opt.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.c_opt.step()

        # actor update
        loss_a = -self.critic(s, self.actor(s)).mean()
        self.a_opt.zero_grad()
        loss_a.backward()
        self.a_opt.step()

        # soft-update targets
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.mul_(1 - self.tau).add_(p * self.tau)
            for p, p_t in zip(self.actor.parameters(), self.actor_target.parameters()):
                p_t.mul_(1 - self.tau).add_(p * self.tau)

    # ---------- save ----------------------------------------------------- #
    def save(self, prefix: str):
        torch.save(self.actor.state_dict(),  f"{prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{prefix}_critic.pt")
