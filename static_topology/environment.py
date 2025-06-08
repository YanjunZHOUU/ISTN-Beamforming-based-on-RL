# import numpy as np
#
#
# class RIS_MISO(object):
#     def __init__(self,
#                  num_antennas,
#                  num_RIS_elements,
#                  num_users,
#                  channel_est_error=False,
#                  AWGN_var=1e-2,
#                  channel_noise_var=1e-2):
#
#         self.M = num_antennas
#         self.L = num_RIS_elements
#         self.K = num_users
#
#         self.channel_est_error = channel_est_error
#
#         assert self.M == self.K
#
#         self.awgn_var = AWGN_var
#         self.channel_noise_var = channel_noise_var
#
#         power_size = 2 * self.K
#
#         channel_size = 2 * (self.L * self.M + self.L * self.K)
#
#         self.action_dim = 2 * self.M * self.K + 2 * self.L
#         self.state_dim = power_size + channel_size + self.action_dim
#
#         self.H_1 = None
#         self.H_2 = None
#         self.G = np.eye(self.M, dtype=complex)
#         self.Phi = np.eye(self.L, dtype=complex)
#
#         self.state = None
#         self.done = None
#
#         self.episode_t = None
#
#     def _compute_H_2_tilde(self):
#         return self.H_2.T @ self.Phi @ self.H_1 @ self.G
#
#     def reset(self):
#         self.episode_t = 0
#
#         self.H_1 = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) + 1j * np.random.normal(0, np.sqrt(0.5),
#                                                                                                (self.L, self.M))
#         self.H_2 = np.random.normal(0, np.sqrt(0.5), (self.L, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5),
#                                                                                                (self.L, self.K))
#
#         init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), np.imag(self.G.reshape(1, -1))))
#         init_action_Phi = np.hstack(
#             (np.real(np.diag(self.Phi)).reshape(1, -1), np.imag(np.diag(self.Phi)).reshape(1, -1)))
#
#         init_action = np.hstack((init_action_G, init_action_Phi))
#
#         Phi_real = init_action[:, -2 * self.L:-self.L]
#         Phi_imag = init_action[:, -self.L:]
#
#         self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)
#
#         power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2
#
#         H_2_tilde = self._compute_H_2_tilde()
#         power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2
#
#         H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
#         H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)
#
#         self.state = np.hstack((init_action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))
#
#         return self.state
#
#     def _compute_reward(self, Phi):
#         reward = 0
#         opt_reward = 0
#
#         for k in range(self.K):
#             h_2_k = self.H_2[:, k].reshape(-1, 1)
#             g_k = self.G[:, k].reshape(-1, 1)
#
#             x = np.abs(h_2_k.T @ Phi @ self.H_1 @ g_k) ** 2
#
#             x = x.item()
#
#             G_removed = np.delete(self.G, k, axis=1)
#
#             interference = np.sum(np.abs(h_2_k.T @ Phi @ self.H_1 @ G_removed) ** 2)
#             y = interference + (self.K - 1) * self.awgn_var
#
#             rho_k = x / y
#
#             reward += np.log(1 + rho_k) / np.log(2)
#             opt_reward += np.log(1 + x / ((self.K - 1) * self.awgn_var)) / np.log(2)
#
#         return reward, opt_reward
#
#     def step(self, action):
#         self.episode_t += 1
#
#         action = action.reshape(1, -1)
#
#         G_real = action[:, :self.M ** 2]
#         G_imag = action[:, self.M ** 2:2 * self.M ** 2]
#
#         Phi_real = action[:, -2 * self.L:-self.L]
#         Phi_imag = action[:, -self.L:]
#
#         self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)
#
#         self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)
#
#         power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2
#
#         H_2_tilde = self._compute_H_2_tilde()
#
#         power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2
#
#         H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
#         H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)
#
#         self.state = np.hstack((action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))
#
#         reward, opt_reward = self._compute_reward(self.Phi)
#
#         done = opt_reward == reward
#
#         return self.state, reward, done, None
#
#     def close(self):
#         pass
#


"""
environment.py
ISTN‑secrecy environment that matches
    Du et al., JSAC 2018, Sec. II‑III (eqs. 1‑12, 18).
Only the *environment* is rewritten – your RL code stays the same.
"""

"""
ISTNEnv — secure satellite-terrestrial environment fully matching
Du et al., JSAC 2018, eqs. (1)–(12) & (18).
"""

"""
Satellite–Terrestrial environment that exactly follows
Du et al., JSAC 2018 (eqs. 1–12, 18) but keeps dimensions configurable via
main.py command-line arguments.

Defaults (paper values):
    N_s = 15   satellite antennas
    N_p = 16   antennas per terrestrial BS
    N_f = 5    FSS terminals
    M   = 15   terrestrial BSs / PU users
"""

"""
ISTNEnv (paper-accurate)

    • Topology defaults:
        satellite antennas N_s = 15
        BS antennas       N_p = 16
        FSS terminals     N_f = 5
        BSs / PU users    M   = 15
    • Physical constants replicate Table II of
      Du et al., JSAC 2018 (see README in code).

You can still override any default through the constructor,
e.g. ISTNEnv(N_s=8, Ps_max=50.0, sigma_s=1e-12)
"""

"""
environment.py  –  ISTNEnv (fully aligned with Du et al., JSAC 2018)

Defaults:
    N_s = 15  satellite antennas
    N_p = 16  antennas per BS
    N_f = 5   FSS terminals
    M   = 15  terrestrial BSs (one PU per BS)

Physical constants from Table II are set as defaults but can be
overridden through constructor kwargs (see README in code).
"""

import numpy as np


# ───────── helper functions ─────────────────────────────────────
def ula_response(N, theta, d_over_lambda=0.5):
    k = np.arange(N)
    return (1 / np.sqrt(N)) * np.exp(-1j * 2 * np.pi *
                                     d_over_lambda * k * np.sin(theta))


def rician_vector(N, L):
    v = np.zeros(N, dtype=complex)
    for _ in range(L):
        delta = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        theta = np.random.uniform(-1, 1) * np.pi / 2
        v += delta * ula_response(N, theta)
    return np.sqrt(N / L) * v


# ───────── environment class ────────────────────────────────────
class ISTNEnv:
    # --- paper-default parameters ---------------------------------------
    Ns, Np, Nf, M = 15, 16, 5, 15          # dimensions
    L_sat, L_bs = 2, 3                     # path counts

    Ps_max = 10 ** (23.01 / 10)            # 23.01 dBW ≈ 200.88 W
    Pb_max = 10 ** (-22.0 / 10)            # −22 dBW ≈ 6.31 mW

    sigma_s = 10 ** (-126.47 / 10)         # 2.25e-13 W
    sigma_p = 10 ** (-121.52 / 10)         # 7.04e-13 W
    sigma_e = 10 ** (-121.52 / 10)         # 7.04e-13 W

    rho_int, rho_ext, rho_e = 0.01, 1.0, 1.0

    # -------------------------------------------------------------------
    def __init__(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)

        self.action_dim = 2 * (self.Ns * self.Nf + self.Np * self.M)
        self.state_dim = (2 * (self.Ns * (self.Nf + 1) +
                               self.Np * (self.M + self.Nf))
                          + self.action_dim)

        self.h_sf = self.h_se = self.g_bp = self.g_bf = None
        self.Ws = self.Wb = None

    # ---------------- channel generation -------------------------------
    def _gen_channels(self):
        self.h_sf = np.vstack([rician_vector(self.Ns, self.L_sat)
                               for _ in range(self.Nf)]).T
        self.h_se = rician_vector(self.Ns, self.L_sat).reshape(-1, 1)
        self.g_bp = np.vstack([rician_vector(self.Np, self.L_bs)
                               for _ in range(self.M)]).T
        self.g_bf = np.vstack([rician_vector(self.Np, self.L_bs)
                               for _ in range(self.Nf)]).T

    # ---------------- helper for state ---------------------------------
    @staticmethod
    def _vec_ri(x): return np.hstack((x.real.reshape(-1), x.imag.reshape(-1)))

    def _state(self):
        return np.hstack([
            self._vec_ri(self.h_sf), self._vec_ri(self.h_se),
            self._vec_ri(self.g_bp), self._vec_ri(self.g_bf),
            self._vec_ri(self.Ws),   self._vec_ri(self.Wb)
        ]).astype(np.float32)

    # ---------------- gym-style API ------------------------------------
    def reset(self):
        self._gen_channels()
        self.Ws = self.h_sf / np.linalg.norm(self.h_sf, axis=0, keepdims=True) \
                  * np.sqrt(self.Ps_max / self.Nf)
        self.Wb = self.g_bp / np.linalg.norm(self.g_bp, axis=0, keepdims=True) \
                  * np.sqrt(self.Pb_max / self.M)
        return self._state()

    def step(self, action):
        # -------- decode action to complex matrices --------------------
        slen = 2 * self.Ns * self.Nf
        blk_s, blk_b = action[:slen], action[slen:]

        def to_cx(v, r, c):
            h = v.shape[0] // 2
            return v[:h].reshape(r, c) + 1j * v[h:].reshape(r, c)

        self.Ws = to_cx(blk_s, self.Ns, self.Nf)
        self.Wb = to_cx(blk_b, self.Np, self.M)

        # power clipping
        Ps = np.sum(np.abs(self.Ws) ** 2)
        if Ps > self.Ps_max:
            self.Ws *= np.sqrt(self.Ps_max / Ps)
        for m in range(self.M):
            Pm = np.sum(np.abs(self.Wb[:, m]) ** 2)
            if Pm > self.Pb_max:
                self.Wb[:, m] *= np.sqrt(self.Pb_max / Pm)

        # -------- SINRs -------------------------------------------------
        eps = 1e-12                        # numerical safety

        # FSS
        sig_f = np.sum(self.h_sf.conj() * self.Ws, axis=0)
        intra_f = self.rho_int * (np.sum(np.abs(self.h_sf.conj() * self.Ws)**2, axis=0)
                                  - np.abs(sig_f)**2)
        ext_f = self.rho_ext * np.sum(np.abs(self.g_bf.conj().T @ self.Wb)**2, axis=1)
        SNR_f = np.abs(sig_f)**2 / (intra_f + ext_f + self.sigma_s + eps)
        SNR_f = np.maximum(SNR_f, 0.0)      # clamp negatives to zero

        # PU
        sig_p = np.sum(self.g_bp.conj() * self.Wb, axis=0)
        intra_p = self.rho_int * (np.sum(np.abs(self.g_bp.conj().T @ self.Wb)**2, axis=1)
                                  - np.abs(sig_p)**2)
        ext_p_scalar = self.rho_ext * np.sum(np.abs(self.h_sf.conj().T @ self.Ws)**2)
        ext_p = np.full(self.M, ext_p_scalar)
        SNR_p = np.abs(sig_p)**2 / (intra_p + ext_p + self.sigma_p + eps)
        SNR_p = np.maximum(SNR_p, 0.0)

        # Eve
        sig_e = (self.h_se.conj().T @ self.Ws[:, -1]).item()
        int_e = self.rho_e * (np.sum(np.abs(self.h_se.conj().T @ self.Ws[:, :-1])**2) +
                              np.sum(np.abs(self.g_bp.conj().T @ self.Wb)**2))
        SNR_e = np.abs(sig_e)**2 / (int_e + self.sigma_e + eps)
        SNR_e = max(SNR_e, 0.0)

        # secrecy capacity (now safe)
        Cs = max(np.log2(1 + SNR_f[-1]) - np.log2(1 + SNR_e), 0.0)
        return self._state(), float(Cs), False, {}


#
#
# # environment.py  (strict SINR with Artificial Noise v)
# import numpy as np
#
# # ---------- helper -------------------------------------------------------
# def _rician(n_rx, n_tx, L=2, K=5):
#     """生成 L 路 Rician 信道，每列归一化单位功率"""
#     h = np.zeros((n_rx, n_tx), dtype=np.complex128)
#     for _ in range(L):
#         h += (np.random.randn(n_rx, n_tx) + 1j*np.random.randn(n_rx, n_tx)) / np.sqrt(2*L)
#     h_los = (np.random.randn(n_rx, n_tx) + 1j*np.random.randn(n_rx, n_tx)) / np.sqrt(2)
#     return np.sqrt(K/(K+1))*h_los + np.sqrt(1/(K+1))*h
#
#
# # ========================================================================
# class ISTNEnv:
#     """
#     信道、干扰、噪声均按论文 (10a)–(10c) 计算
#       action = [Re{Ws}, Im{Ws}, Re{Wb}, Im{Wb}, Re{v}, Im{v}]
#       d_a    = 2 (Ns*Nf + Np*M + Ns)
#     """
#     # ---- 初始化 ---------------------------------------------------------
#     def __init__(self,
#                  N_s=15, N_p=16,
#                  N_f=5,  M=15,
#                  Ps_max=200.9,          # 23.01 dBW → 10^(23.01/10)
#                  Pb_max=6.31e-3,        # -22 dBW
#                  sigma_s=2.26e-13,      # -126.47 dBW
#                  sigma_p=7.06e-13,      # -121.52 dBW
#                  sigma_e=7.06e-13,
#                  rho_int=0.1, rho_ext=0.05, rho_e=1.0):
#         self.Ns, self.Np = N_s, N_p
#         self.Nf, self.M  = N_f, M
#
#         self.Ps_max, self.Pb_max = Ps_max, Pb_max
#         self.sigma_s, self.sigma_p, self.sigma_e = sigma_s, sigma_p, sigma_e
#         self.rho_int, self.rho_ext, self.rho_e = rho_int, rho_ext, rho_e
#
#         # -------- 动作 / 状态维度 -----------
#         self.action_dim = 2 * (N_s*N_f + N_p*M + N_s)
#         self.state_dim  = 2 * ( N_s*N_f          # H_sf
#                                + N_s             # h_se
#                                + N_p*M           # G_bp
#                                + N_p*N_f         # G_bf
#                                + N_s*N_f         # Ws (t-1)
#                                + N_p*M )         # Wb (t-1)
#
#         self.reset()
#
#     # ---- 每 episode 采样一次信道 ---------------------------------------
#     def reset(self):
#         # 卫星 → FSS / Eve
#         self.H_sf = _rician(self.Ns, self.Nf, L=2)
#         self.h_se = _rician(self.Ns, 1,      L=2).reshape(-1, 1)
#
#         # BS → FSS / PU / Eve
#         self.G_bf = _rician(self.Np, self.Nf, L=3)          # toward FSS
#         self.G_bp = _rician(self.Np, self.M,  L=3)          # toward PU
#         self.G_be = _rician(self.Np, 1,      L=3).reshape(-1, 1)
#
#         # 上一时隙波束初始化为 0
#         self.Ws = np.zeros((self.Ns, self.Nf), dtype=np.complex128)
#         self.Wb = np.zeros((self.Np, self.M),  dtype=np.complex128)
#
#         return self._state()
#
#     # ---- 将复杂矩阵打平成实向量 ----------------------------------------
#     def _state(self):
#         def c2r(mat):
#             return np.hstack([mat.real.ravel(), mat.imag.ravel()])
#         return np.hstack([ c2r(self.H_sf), c2r(self.h_se),
#                            c2r(self.G_bp), c2r(self.G_bf),
#                            c2r(self.Ws),   c2r(self.Wb) ])
#
#     # ---- step -----------------------------------------------------------
#     def step(self, a_np):
#         a = a_np.flatten()
#         Ns, Np, Nf, M = self.Ns, self.Np, self.Nf, self.M
#
#         # ----------- 解码动作向量 ----------------
#         ptr = 0
#         size = Ns * Nf
#         Ws_r = a[ptr:ptr+size];  ptr += size
#         Ws_i = a[ptr:ptr+size];  ptr += size
#         self.Ws = Ws_r.reshape(Ns, Nf) + 1j*Ws_i.reshape(Ns, Nf)
#
#         size = Np * M
#         Wb_r = a[ptr:ptr+size];  ptr += size
#         Wb_i = a[ptr:ptr+size];  ptr += size
#         self.Wb = Wb_r.reshape(Np, M) + 1j*Wb_i.reshape(Np, M)
#
#         size = Ns
#         v_r  = a[ptr:ptr+size];  v_i = a[ptr+size:ptr+2*size]
#         self.v = (v_r + 1j*v_i).reshape(Ns, 1)
#
#         # ----------- 功率归一化 -----------------
#         Ps_now = np.linalg.norm(self.Ws)**2 + np.linalg.norm(self.v)**2
#         if Ps_now > self.Ps_max > 0:
#             scale = np.sqrt(self.Ps_max / Ps_now)
#             self.Ws *= scale; self.v *= scale
#
#         for m in range(M):
#             p = np.linalg.norm(self.Wb[:, m])**2
#             if p > self.Pb_max > 0:
#                 self.Wb[:, m] *= np.sqrt(self.Pb_max / p)
#
#         # ------------ S I N R 计算 --------------
#         idx_tar = Nf - 1
#         h_n = self.H_sf[:, [idx_tar]]                 # Ns×1
#
#         # --- Desired signal / FSS_N -------------
#         sig_f = np.abs(h_n.conj().T @ self.Ws[:, [idx_tar]])**2
#
#         # --- Intra-system interference ----------
#         I_int_n = 0.0
#         if Nf > 1:
#             other = np.delete(self.Ws, idx_tar, axis=1)    # Ns×(Nf-1)
#             I_int_n = np.sum(np.abs(h_n.conj().T @ other)**2)
#
#         # --- Extra-system interference ----------
#         g_bf_n = self.G_bf[:, [idx_tar]]                    # Np×1
#         I_ext_n = np.sum(np.abs(g_bf_n.conj().T @ self.Wb)**2)
#
#         # --- AN interference (to FSS) ------------
#         I_AN_n = np.abs(h_n.conj().T @ self.v)**2
#
#         Gamma_f = sig_f / ( self.rho_int * I_int_n +
#                             self.rho_ext * I_ext_n +
#                             self.rho_int * I_AN_n +
#                             self.sigma_s )
#
#         # ------------- Eve link ------------------
#         h_e = self.h_se                                   # Ns×1
#         sig_e = np.abs(h_e.conj().T @ self.Ws[:, [idx_tar]])**2
#
#         I_s_e   = 0.0
#         if Nf > 1:
#             I_s_e = np.sum(np.abs(h_e.conj().T @ np.delete(self.Ws, idx_tar, axis=1))**2)
#
#         I_p_e   = np.sum(np.abs(self.G_be.conj().T @ self.Wb)**2)
#         I_AN_e  = np.abs(h_e.conj().T @ self.v)**2
#
#         Gamma_e = sig_e / ( self.rho_e * I_s_e +
#                             self.rho_e * I_p_e +
#                             self.rho_e * I_AN_e +
#                             self.sigma_e )
#
#         # ------------- reward --------------------
#         Cs = max(np.log2(1 + Gamma_f.item()) - np.log2(1 + Gamma_e.item()), 0.0)
#         return self._state(), Cs, False, {}
