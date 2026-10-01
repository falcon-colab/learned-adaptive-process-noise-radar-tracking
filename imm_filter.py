"""
imm_filter.py  --  RESEARCH EXTENSION: classical Interacting Multiple
Model (IMM) baseline, following the standard formulation in Bar-Shalom,
Li & Kirubarajan, "Estimation with Applications to Tracking and
Navigation" (Wiley, 2001) -- already cited as [7] in the paper.

This is the real classical answer to maneuvering-target tracking (a
single fixed-Q filter, our earlier "classical" baseline, was never
designed to handle maneuvers at all). IMM runs several motion-model
hypotheses in parallel and probabilistically switches between them.
Comparing our learned correction against IMM, not just against a
filter that was never built for this problem, is the fair comparison.

Two models here, both constant-velocity (same F, H as the rest of this
project), differing ONLY in process noise scale:
  Model 1 ("non-maneuvering"): q_low  = cfg.Q_SCALE (0.5), matches the
    classical baseline exactly.
  Model 2 ("maneuvering"):     q_high, selected via a systematic grid
    search on the maneuvering scenarios (see select_q_high.py) rather
    than a hand-picked value.

Standard textbook defaults are used for everything else: a symmetric
Markov transition matrix with 0.95 probability of remaining in the
same mode per scan (a very commonly used default in the IMM
literature), and uniform initial mode probabilities. Neither was tuned
to favor any particular outcome.
"""

import numpy as np
import config as cfg

T = cfg.T
F = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Q_UNIT = np.array([
    [T**3/3, 0,      T**2/2, 0],
    [0,      T**3/3, 0,      T**2/2],
    [T**2/2, 0,      T,      0],
    [0,      T**2/2, 0,      T],
])

DEFAULT_TRANSITION = np.array([[0.95, 0.05],
                                [0.05, 0.95]])  # standard textbook default


def gaussian_likelihood(innovation, S):
    """N(innovation; 0, S) for a 2-D innovation vector."""
    d = innovation.shape[0]
    det_S = np.linalg.det(S)
    if det_S <= 0:
        det_S = 1e-12
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * det_S)
    exponent = -0.5 * innovation @ np.linalg.solve(S, innovation)
    return norm_const * np.exp(exponent)


class IMMFilter:
    def __init__(self, q_low=None, q_high=None, transition=None):
        self.q_low = cfg.Q_SCALE if q_low is None else q_low
        self.q_high = 10.0 if q_high is None else q_high  # placeholder; see select_q_high.py
        self.Pi = DEFAULT_TRANSITION if transition is None else transition
        self.Q = [self.q_low * Q_UNIT, self.q_high * Q_UNIT]
        self.n_models = 2

    def initialize(self, x0, y0):
        x_init = np.array([x0, y0, 0.0, 0.0])
        P_init = np.diag([cfg.P0_POS_STD**2, cfg.P0_POS_STD**2,
                           cfg.P0_VEL_STD**2, cfg.P0_VEL_STD**2])
        self.x = [x_init.copy(), x_init.copy()]
        self.P = [P_init.copy(), P_init.copy()]
        self.mu = np.array([0.5, 0.5])  # uniform initial mode probabilities

    def step(self, z=None, R=None):
        # ---- Step 1-2: predicted mode probs + mixing weights ----
        c = self.Pi.T @ self.mu  # c[j] = sum_i Pi[i,j] mu[i]
        mix_w = np.zeros((self.n_models, self.n_models))  # mix_w[i,j] = mu_{i|j}
        for j in range(self.n_models):
            for i in range(self.n_models):
                mix_w[i, j] = self.Pi[i, j] * self.mu[i] / c[j]

        # ---- Step 3: mixed initial conditions per model ----
        x0j, P0j = [], []
        for j in range(self.n_models):
            xm = sum(mix_w[i, j] * self.x[i] for i in range(self.n_models))
            Pm = np.zeros((4, 4))
            for i in range(self.n_models):
                diff = (self.x[i] - xm).reshape(-1, 1)
                Pm += mix_w[i, j] * (self.P[i] + diff @ diff.T)
            x0j.append(xm)
            P0j.append(Pm)

        # ---- Step 4: model-conditioned filtering ----
        x_new, P_new, likelihoods = [], [], []
        for j in range(self.n_models):
            x_pred = F @ x0j[j]
            P_pred = F @ P0j[j] @ F.T + self.Q[j]
            if z is not None:
                innovation = z - H @ x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                xj = x_pred + K @ innovation
                Pj = (np.eye(4) - K @ H) @ P_pred
                lam = gaussian_likelihood(innovation, S)
            else:
                xj, Pj, lam = x_pred, P_pred, 1.0  # no info this scan
            x_new.append(xj); P_new.append(Pj); likelihoods.append(lam)

        # ---- Step 5: mode probability update ----
        num = np.array(likelihoods) * c
        self.mu = num / num.sum()

        # ---- Step 6: combination ----
        x_comb = sum(self.mu[j] * x_new[j] for j in range(self.n_models))
        P_comb = np.zeros((4, 4))
        for j in range(self.n_models):
            diff = (x_new[j] - x_comb).reshape(-1, 1)
            P_comb += self.mu[j] * (P_new[j] + diff @ diff.T)

        self.x, self.P = x_new, P_new
        return x_comb, P_comb, self.mu.copy()


def run_imm(x_target_meas, y_target_meas, detected, r_meas, theta_meas,
            measurement_covariance_fn, q_low=None, q_high=None, transition=None):
    imm = IMMFilter(q_low=q_low, q_high=q_high, transition=transition)
    first_idx = int(np.argmax(detected))
    imm.initialize(x_target_meas[first_idx], y_target_meas[first_idx])

    x_history = np.zeros((cfg.N, 4))
    P_history = np.zeros((cfg.N, 4, 4))
    mu_history = np.zeros((cfg.N, 2))
    for i in range(cfg.N):
        if detected[i]:
            z = np.array([x_target_meas[i], y_target_meas[i]])
            R = measurement_covariance_fn(r_meas[i], theta_meas[i])
            x_history[i], P_history[i], mu_history[i] = imm.step(z, R)
        else:
            x_history[i], P_history[i], mu_history[i] = imm.step(None, None)
    return x_history, P_history, mu_history
