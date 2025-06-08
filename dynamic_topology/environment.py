# environment.py  —  Dynamic ISTN with gentle topology changes
import numpy as np

# ───────── helper functions ───────────────────────────────────────────── #
def ula_response(N: int, theta: float, d_over_lambda: float = 0.5) -> np.ndarray:
    k = np.arange(N)
    return (1 / np.sqrt(N)) * np.exp(-1j * 2 * np.pi *
                                     d_over_lambda * k * np.sin(theta))

def rician_vector(N: int, L: int) -> np.ndarray:
    v = np.zeros(N, dtype=complex)
    for _ in range(L):
        delta = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        theta = np.random.uniform(-1, 1) * np.pi / 2
        v += delta * ula_response(N, theta)
    return np.sqrt(N / L) * v


# ───────── environment class ──────────────────────────────────────────── #
class ISTNEnv:
    # antenna counts
    Ns, Np = 15, 16
    # Rician path counts
    L_sat, L_bs = 2, 3

    # power & noise (W)
    Ps_max = 10 ** (23.01 / 10)
    Pb_max = 10 ** (-22.0 / 10)
    sigma_s = 10 ** (-126.47 / 10)
    sigma_p = 10 ** (-121.52 / 10)
    sigma_e = 10 ** (-121.52 / 10)

    rho_int, rho_ext, rho_e = 0.01, 1.0, 1.0

    # padded topology limits
    max_FSS, max_PU = 10, 25
    min_FSS, min_PU = 1, 1

    # dynamics knobs
    topo_hz = 800                   # when add/drop is *considered*
    p_add_fss = p_drop_fss = 0.05   # gentler than 0.5
    p_add_pu  = p_drop_pu  = 0.05

    # tiny OU drift (≈ 0.05° rms)
    angle_mu            = 0.0
    angle_theta         = 0.2        # mean-reversion
    angle_sigma         = np.deg2rad(0.05)

    # ------------------------------------------------------------------ #
    def __init__(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)

        self.action_dim = 2 * (self.Ns * self.max_FSS + self.Np * self.max_PU)
        self.state_dim  = (2 * (self.Ns * (self.max_FSS + 1) +
                                self.Np * (self.max_PU + self.max_FSS))
                           + self.action_dim + self.max_FSS + self.max_PU)

        # internal
        self._step = 0
        self.Nf = self.M = None
        self.h_sf = self.h_se = self.g_bp = self.g_bf = None
        self.Ws = self.Wb = None
        self._theta_f = None
        self._ou_state = None

    # ---------------- channel generation -------------------------------- #
    def _gen_channels(self):
        self.h_sf[:, :self.Nf] = np.vstack([rician_vector(self.Ns, self.L_sat)
                                            for _ in range(self.Nf)]).T
        self.h_se = rician_vector(self.Ns, self.L_sat).reshape(-1, 1)

        self.g_bp[:, :self.M] = np.vstack([rician_vector(self.Np, self.L_bs)
                                           for _ in range(self.M)]).T
        self.g_bf[:, :self.Nf] = np.vstack([rician_vector(self.Np, self.L_bs)
                                            for _ in range(self.Nf)]).T
        # initial steering angles
        self._theta_f  = np.random.uniform(-np.pi/2, np.pi/2, size=self.Nf)
        self._ou_state = self._theta_f.copy()

    # ---------------- helper -------------------------------------------- #
    @staticmethod
    def _vec_ri(x): return np.hstack((x.real.ravel(), x.imag.ravel()))

    def _state(self):
        mask_f = np.concatenate((np.ones(self.Nf), np.zeros(self.max_FSS - self.Nf)))
        mask_p = np.concatenate((np.ones(self.M),  np.zeros(self.max_PU  - self.M)))
        return np.hstack([
            self._vec_ri(self.h_sf), self._vec_ri(self.h_se),
            self._vec_ri(self.g_bp), self._vec_ri(self.g_bf),
            self._vec_ri(self.Ws),   self._vec_ri(self.Wb),
            mask_f, mask_p
        ]).astype(np.float32)

    # ---------------- gentle angle OU drift ----------------------------- #
    def _drift_angles(self):
        noise = np.random.randn(self.Nf) * self.angle_sigma
        self._ou_state += self.angle_theta * (self.angle_mu - self._ou_state) + noise
        self._theta_f = self._ou_state.copy()
        for k in range(self.Nf):
            self.h_sf[:, k] = ula_response(self.Ns, self._theta_f[k])

    # ---------------- gentle add/drop ----------------------------------- #
    def _add_drop_nodes(self):
        # at most ±1 node each event, low probability
        if np.random.rand() < self.p_add_fss and self.Nf < self.max_FSS:
            k = self.Nf
            self.h_sf[:, k] = rician_vector(self.Ns, self.L_sat)
            self.g_bf[:, k] = rician_vector(self.Np, self.L_bs)
            self.Ws[:, k]   = self.h_sf[:, k] / np.linalg.norm(self.h_sf[:, k]) * np.sqrt(self.Ps_max)
            self._theta_f   = np.append(self._theta_f, np.random.uniform(-np.pi/2, np.pi/2))
            self._ou_state  = np.append(self._ou_state, 0.0)
            self.Nf += 1
        elif np.random.rand() < self.p_drop_fss and self.Nf > self.min_FSS:
            drop = np.random.randint(self.Nf)
            keep = [i for i in range(self.Nf) if i != drop]
            self.h_sf[:, :self.Nf-1] = self.h_sf[:, keep]
            self.g_bf[:, :self.Nf-1] = self.g_bf[:, keep]
            self.Ws[:,  :self.Nf-1]  = self.Ws[:,  keep]
            self._theta_f            = self._theta_f[keep]
            self._ou_state           = self._ou_state[keep]
            self.Nf -= 1
            self.h_sf[:, self.Nf:] = self.g_bf[:, self.Nf:] = self.Ws[:, self.Nf:] = 0.0

        # PU side
        if np.random.rand() < self.p_add_pu and self.M < self.max_PU:
            k = self.M
            self.g_bp[:, k] = rician_vector(self.Np, self.L_bs)
            self.Wb[:, k]   = self.g_bp[:, k] / np.linalg.norm(self.g_bp[:, k]) * np.sqrt(self.Pb_max)
            self.M += 1
        elif np.random.rand() < self.p_drop_pu and self.M > self.min_PU:
            drop = np.random.randint(self.M)
            keep = [i for i in range(self.M) if i != drop]
            self.g_bp[:, :self.M-1] = self.g_bp[:, keep]
            self.Wb[:,  :self.M-1]  = self.Wb[:,  keep]
            self.M -= 1
            self.g_bp[:, self.M:] = self.Wb[:, self.M:] = 0.0

    # ---------------- gym-style API ------------------------------------- #
    def reset(self):
        self._step = 0
        self.Nf = np.random.randint(self.min_FSS, self.max_FSS + 1)
        self.M  = np.random.randint(self.min_PU,  self.max_PU  + 1)

        # zero-init padded arrays
        self.h_sf = np.zeros((self.Ns, self.max_FSS), dtype=complex)
        self.g_bp = np.zeros((self.Np, self.max_PU),  dtype=complex)
        self.g_bf = np.zeros((self.Np, self.max_FSS), dtype=complex)
        self.Ws   = np.zeros((self.Ns, self.max_FSS), dtype=complex)
        self.Wb   = np.zeros((self.Np, self.max_PU),  dtype=complex)

        self._gen_channels()

        # MRT initial beams
        self.Ws[:, :self.Nf] = (self.h_sf[:, :self.Nf] /
                                np.linalg.norm(self.h_sf[:, :self.Nf], axis=0, keepdims=True)
                               ) * np.sqrt(self.Ps_max / self.Nf)
        self.Wb[:, :self.M]  = (self.g_bp[:, :self.M]  /
                                np.linalg.norm(self.g_bp[:, :self.M],  axis=0, keepdims=True)
                               ) * np.sqrt(self.Pb_max / self.M)
        return self._state()

    def step(self, action: np.ndarray):
        self._step += 1
        slen = 2 * self.Ns * self.max_FSS

        def to_cx(v, r, c):
            h = v.shape[0] // 2
            return v[:h].reshape(r, c) + 1j * v[h:].reshape(r, c)
        self.Ws = to_cx(action[:slen], self.Ns, self.max_FSS)
        self.Wb = to_cx(action[slen:], self.Np, self.max_PU)
        self.Ws[:, self.Nf:] = 0.0
        self.Wb[:, self.M:]  = 0.0

        # power clipping
        Ps = np.sum(np.abs(self.Ws[:, :self.Nf]) ** 2)
        if Ps > self.Ps_max:
            self.Ws[:, :self.Nf] *= np.sqrt(self.Ps_max / Ps)
        for m in range(self.M):
            Pm = np.sum(np.abs(self.Wb[:, m]) ** 2)
            if Pm > self.Pb_max:
                self.Wb[:, m] *= np.sqrt(self.Pb_max / Pm)

        # dynamics -------------------------------------------------------
        self._drift_angles()
        if self._step % self.topo_hz == 0:
            self._add_drop_nodes()

        # SINR calculations (active nodes only) -------------------------
        eps = 1e-12
        sig_f = np.sum(self.h_sf[:, :self.Nf].conj() * self.Ws[:, :self.Nf], axis=0)
        intra_f = self.rho_int * (np.sum(np.abs(self.h_sf[:, :self.Nf].conj() *
                                                self.Ws[:, :self.Nf]) ** 2, axis=0)
                                  - np.abs(sig_f) ** 2)
        ext_f = self.rho_ext * np.sum(np.abs(self.g_bf[:, :self.Nf].conj().T @
                                             self.Wb[:, :self.M]) ** 2, axis=1)
        SNR_f = np.maximum(np.abs(sig_f)**2 / (intra_f + ext_f + self.sigma_s + eps), 0.0)

        sig_p = np.sum(self.g_bp[:, :self.M].conj() * self.Wb[:, :self.M], axis=0)
        intra_p = self.rho_int * (np.sum(np.abs(self.g_bp[:, :self.M].conj().T @
                                                self.Wb[:, :self.M]) ** 2, axis=1)
                                  - np.abs(sig_p) ** 2)
        ext_p_s = self.rho_ext * np.sum(np.abs(self.h_sf[:, :self.Nf].conj().T @
                                               self.Ws[:, :self.Nf]) ** 2)
        SNR_p = np.maximum(np.abs(sig_p)**2 / (intra_p + ext_p_s + self.sigma_p + eps), 0.0)

        sig_e = (self.h_se.conj().T @ self.Ws[:, max(self.Nf - 1, 0)]).item()
        int_e = self.rho_e * (np.sum(np.abs(self.h_se.conj().T @
                                            self.Ws[:, :max(self.Nf - 1, 1)]) ** 2) +
                              np.sum(np.abs(self.g_bp[:, :self.M].conj().T @
                                            self.Wb[:, :self.M]) ** 2))
        SNR_e = max(np.abs(sig_e) ** 2 / (int_e + self.sigma_e + eps), 0.0)

        Cs = max(np.log2(1 + (SNR_f[-1] if self.Nf else 0.0)) -
                 np.log2(1 + SNR_e), 0.0)
        return self._state(), float(Cs), False, {}
