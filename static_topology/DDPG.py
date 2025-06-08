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


import copy, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np


# ───────── helper MLP ------------------------------------------------------
def mlp(inp, out, hidden):
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.LayerNorm(hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        nn.Linear(hidden, out)
    )


# ───────── actor -----------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, sdim, adim, max_action=1.0):
        super().__init__()
        hid = 2 ** (sdim - 1).bit_length()
        self.net = mlp(sdim, adim, hid)
        self.max_action = max_action

    def forward(self, s):
        return torch.tanh(self.net(s)) * self.max_action


# ───────── critic ----------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, sdim, adim):
        super().__init__()
        hid = 2 ** ((sdim + adim) - 1).bit_length()
        self.net = mlp(sdim + adim, 1, hid)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], 1))


# ───────── on-policy DDPG wrapper -----------------------------------------
class OnPolicyDDPG:
    """
    Updates actor & critic from the *current* transition only
    (no replay buffer).  Still uses target networks and soft τ-updates.
    """

    def __init__(self, state_dim, action_dim,
                 actor_lr=1e-3, critic_lr=1e-3,
                 actor_decay=1e-5, critic_decay=1e-5,
                 max_action=1.0, discount=0.99, tau=5e-3, device="cpu"):

        self.device = torch.device(device)
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.a_opt = torch.optim.Adam(self.actor.parameters(),
                                      lr=actor_lr, weight_decay=actor_decay)
        self.c_opt = torch.optim.Adam(self.critic.parameters(),
                                      lr=critic_lr, weight_decay=critic_decay)

        self.discount = discount
        self.tau = tau
        self.max_action = max_action

    # ------- act (no grad) -------------------------------------------------
    @torch.no_grad()
    def select_action(self, state_np):
        s = torch.FloatTensor(state_np.reshape(1, -1)).to(self.device)
        return self.actor(s).cpu().numpy().flatten()

    # ------- single on-policy update --------------------------------------
    def train_step(self, s, a, r, s2, done):
        """
        One gradient step using this transition only.
        Args are NumPy arrays / scalars.
        """
        s  = torch.FloatTensor(s.reshape(1, -1)).to(self.device)
        a  = torch.FloatTensor(a.reshape(1, -1)).to(self.device)
        r  = torch.FloatTensor([[r]]).to(self.device)
        s2 = torch.FloatTensor(s2.reshape(1, -1)).to(self.device)
        not_done = torch.FloatTensor([[1.0 - done]]).to(self.device)

        # target Q
        with torch.no_grad():
            a2 = self.actor_target(s2)
            target_Q = r + not_done * self.discount * self.critic_target(s2, a2)

        # critic loss
        current_Q = self.critic(s, a)
        c_loss = F.mse_loss(current_Q, target_Q)
        self.c_opt.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.c_opt.step()

        # actor (deterministic policy gradient)
        a_pred = self.actor(s)
        a_loss = -self.critic(s, a_pred).mean()
        self.a_opt.zero_grad()
        a_loss.backward()
        self.a_opt.step()

        # soft update
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, p_t in zip(self.actor.parameters(), self.actor_target.parameters()):
            p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)

    # ------- save / load ---------------------------------------------------
    def save(self, prefix):
        torch.save(self.actor.state_dict(),  f"{prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{prefix}_critic.pt")


#
# import copy, numpy as np
# import torch, torch.nn as nn, torch.nn.functional as F
#
#
# # ───── helper MLP ─────────────────────────────────────────────────────────
# def mlp(inp, out, hidden, depth=2):
#     layers, d = [], inp
#     for _ in range(depth):
#         layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
#         d = hidden
#     layers.append(nn.Linear(hidden, out))
#     return nn.Sequential(*layers)
#
#
# # ───── Actor ──────────────────────────────────────────────────────────────
# # ─── 修正版 _only_ Actor 类 (替换原文件同名类即可) ──────────────────
# class Actor(nn.Module):
#     def __init__(self, sdim, Ns, Np, Nf, M,
#                  Ps_max, Pb_max, max_action=1.0):
#         super().__init__()
#         hidden = 2 ** ((sdim - 1).bit_length())
#         self.Ns, self.Np, self.Nf, self.M = Ns, Np, Nf, M
#         self.Ps_max, self.Pb_max = Ps_max, Pb_max
#         self.adim = 2 * (Ns * Nf + Np * M + Ns)     # real+imag
#         self.net  = mlp(sdim, self.adim, hidden, 2)
#         self.max_a = max_action
#
#     # ---------- forward (all-real) -------------------------------------
#     def forward(self, s):
#         """
#         输入  s : (B, state_dim)   ─→   输出 a : (B, action_dim)  ∈ (-1,1)
#         已内部功率归一化；**全程 real32**，彻底避开复数 CUDA kernel
#         """
#         B   = s.size(0)
#         raw = torch.tanh(self.net(s)) * self.max_a     # (B, adim)
#
#         Ns, Np, Nf, M = self.Ns, self.Np, self.Nf, self.M
#         ptr = 0
#
#         # ------- Ws ----------------------------------------------------
#         sz = Ns * Nf
#         Ws_r = raw[:, ptr:ptr+sz].view(B, Ns, Nf); ptr += sz
#         Ws_i = raw[:, ptr:ptr+sz].view(B, Ns, Nf); ptr += sz
#
#         # ------- Wb ----------------------------------------------------
#         sz = Np * M
#         Wb_r = raw[:, ptr:ptr+sz].view(B, Np, M); ptr += sz
#         Wb_i = raw[:, ptr:ptr+sz].view(B, Np, M); ptr += sz
#
#         # ------- v  ----------------------------------------------------
#         sz = Ns
#         v_r = raw[:, ptr:ptr+sz].view(B, Ns, 1)
#         v_i = raw[:, ptr+sz:ptr+2*sz].view(B, Ns, 1)
#
#         p_sat = (Ws_r.pow(2) + Ws_i.pow(2)).sum((1, 2)) + (v_r.pow(2) + v_i.pow(2)).sum((1, 2))
#
#         # 1) 卫星功率裁剪
#         scale_sat = torch.sqrt((p_sat / self.Ps_max).clamp(min=1.0)).view(-1, 1, 1)
#         Ws_r = Ws_r / scale_sat  # ← no in-place
#         Ws_i = Ws_i / scale_sat
#         v_r = v_r / scale_sat
#         v_i = v_i / scale_sat
#
#         # 2) BS 列功率裁剪
#         p_bs = (Wb_r.pow(2) + Wb_i.pow(2)).sum(1, keepdim=True)
#         scale_bs = torch.sqrt((p_bs / self.Pb_max).clamp(min=1.0))
#         Wb_r = Wb_r / scale_bs  # ← no in-place
#         Wb_i = Wb_i / scale_bs
#
#         # ==== 3) 拼回动作向量 (仍纯 real) =============================
#         a_out = torch.cat([
#             Ws_r.reshape(B,-1), Ws_i.reshape(B,-1),
#             Wb_r.reshape(B,-1), Wb_i.reshape(B,-1),
#             v_r.reshape(B,-1),  v_i.reshape(B,-1)
#         ], dim=1)
#         return a_out
#
# # ───── Critic (单 Q) ─────────────────────────────────────────────────────
# class Critic(nn.Module):
#     def __init__(self, sdim, adim):
#         super().__init__()
#         hidden = 2 ** (((sdim + adim) - 1).bit_length())
#         self.net = mlp(sdim + adim, 1, hidden, 2)
#
#     def forward(self, s, a):
#         return self.net(torch.cat([s, a], dim=1))
#
#
# # ───── On-policy DDPG Wrapper ────────────────────────────────────────────
# class OnPolicyDDPG:
#     """
#     无重放池；每交互一步立即用该 transition 更新一次网络。
#     """
#
#     def __init__(self, state_dim, action_dim,
#                  Ns, Np, Nf, M, Ps_max, Pb_max,
#                  max_action=1.0,
#                  actor_lr=1e-3, critic_lr=1e-3,
#                  actor_decay=1e-5, critic_decay=1e-5,
#                  discount=0.99, tau=5e-3,
#                  device="cpu"):
#
#         self.device = torch.device(device)
#
#         # ---- networks ---------------------------------------------------
#         self.actor = Actor(state_dim, Ns, Np, Nf, M,
#                            Ps_max, Pb_max, max_action).to(self.device)
#         self.actor_t = copy.deepcopy(self.actor)
#
#         self.critic = Critic(state_dim, action_dim).to(self.device)
#         self.critic_t = copy.deepcopy(self.critic)
#
#         # ---- optimizers -------------------------------------------------
#         self.a_opt = torch.optim.Adam(self.actor.parameters(),
#                                       lr=actor_lr, weight_decay=actor_decay)
#         self.c_opt = torch.optim.Adam(self.critic.parameters(),
#                                       lr=critic_lr, weight_decay=critic_decay)
#
#         # ---- hyper-params ----------------------------------------------
#         self.discount = discount
#         self.tau      = tau
#         self.action_dim = action_dim
#
#     # ---------- act ------------------------------------------------------
#     @torch.no_grad()
#     def select_action(self, s_np):
#         s = torch.as_tensor(s_np, dtype=torch.float32,
#                             device=self.device).unsqueeze(0)
#         a = self.actor(s)
#         return a.cpu().numpy().flatten()
#
#     # ---------- single-step update --------------------------------------
#     def train_step(self, s, a, r, s2, done):
#         """
#         参数均为 NumPy array / 标量
#         """
#         s   = torch.as_tensor(s.reshape(1, -1), dtype=torch.float32,
#                               device=self.device)
#         a   = torch.as_tensor(a.reshape(1, -1), dtype=torch.float32,
#                               device=self.device)
#         r   = torch.as_tensor([[r]], dtype=torch.float32,
#                               device=self.device)
#         s2  = torch.as_tensor(s2.reshape(1, -1), dtype=torch.float32,
#                               device=self.device)
#         nd  = torch.as_tensor([[1.0 - done]], dtype=torch.float32,
#                               device=self.device)
#
#         # ----- critic ---------------------------------------------------
#         with torch.no_grad():
#             a2 = self.actor_t(s2)
#             y  = r + nd * self.discount * self.critic_t(s2, a2)   # TD 目标
#         q = self.critic(s, a)
#         loss_q = F.mse_loss(q, y)
#         self.c_opt.zero_grad(); loss_q.backward()
#         torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
#         self.c_opt.step()
#
#         # ----- actor ----------------------------------------------------
#         loss_pi = -self.critic(s, self.actor(s)).mean()
#         self.a_opt.zero_grad(); loss_pi.backward()
#         self.a_opt.step()
#
#         # ----- soft update ---------------------------------------------
#         with torch.no_grad():
#             for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
#                 pt.mul_(1 - self.tau).add_(self.tau * p)
#             for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
#                 pt.mul_(1 - self.tau).add_(self.tau * p)
#
#     # ---------- checkpoint ----------------------------------------------
#     def save(self, prefix):
#         torch.save(self.actor.state_dict(),  f"{prefix}_actor.pt")
#         torch.save(self.critic.state_dict(), f"{prefix}_critic.pt")
#
#     def load(self, prefix):
#         self.actor.load_state_dict(torch.load(f"{prefix}_actor.pt"))
#         self.critic.load_state_dict(torch.load(f"{prefix}_critic.pt"))
#         self.actor_t = copy.deepcopy(self.actor)
#         self.critic_t = copy.deepcopy(self.critic)
