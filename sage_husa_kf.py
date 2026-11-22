"""
sage_husa_kf.py  --  RESEARCH EXTENSION (camera-ready addition):
classical adaptive-Q baseline via the Sage-Husa recursive noise
estimator (Sage & Husa, 1969 -- already cited in the paper's Related
Work), to sit alongside IMM as a second classical answer to the same
problem the learned correction addresses.

Unlike IMM (which switches between a small number of pre-specified
discrete process-noise levels), Sage-Husa continuously re-estimates a
single process-noise covariance matrix online from the innovation/
state-correction sequence, using a forgetting factor b. No training
data or neural network is involved -- it adapts within a single run,
scan by scan, exactly as a deployed classical filter would.

Standard formulation (same one a reader would find in Sage & Husa's
original note, and in the many adaptive-KF surveys that followed it;
only the a single scalar/matrix process-noise channel is adapted
here, matching the scope of this paper's own learned correction,
which likewise adapts only Q and leaves F, H, R untouched):

    d_k   = (1 - b) / (1 - b^(k+1))          # time-varying, higher early on
    Q_hat <- (1 - d_k) * Q_hat + d_k * (K y y^T K^T + P_est - F P_prev F^T)

On a missed-detection scan there is no innovation to learn from, so
the estimate is simply carried forward unchanged, matching how every
other filter in this paper handles missed detections.
"""

import numpy as np
import config as cfg

T = cfg.T
F_MAT = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
H_MAT = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Q_UNIT = np.array([
    [T**3/3, 0,      T**2/2, 0],
    [0,      T**3/3, 0,      T**2/2],
    [T**2/2, 0,      T,      0],
    [0,      T**2/2, 0,      T],
])


class SageHusaKF:
    def __init__(self, b=0.98, q0_scale=None):
        self.b = b
        self.q0_scale = cfg.Q_SCALE if q0_scale is None else q0_scale
        self.F = F_MAT
        self.H = H_MAT
        self.Q = self.q0_scale * Q_UNIT  # initial process-noise estimate
        self.k = 0

    def initialize(self, x0, y0):
        self.x_est = np.array([x0, y0, 0.0, 0.0])
        self.P_est = np.diag([cfg.P0_POS_STD**2, cfg.P0_POS_STD**2,
                               cfg.P0_VEL_STD**2, cfg.P0_VEL_STD**2])
        self.k = 0

    def _clip_psd(self, M):
        """Keep the adapted Q numerically well-behaved: symmetrize and
        floor tiny/negative eigenvalues (adaptation can otherwise drift
        slightly indefinite after many scans of noisy updates)."""
        M = 0.5 * (M + M.T)
        w, v = np.linalg.eigh(M)
        w = np.clip(w, 1e-6, None)
        return (v * w) @ v.T

    def step(self, z=None, R=None):
        P_prev = self.P_est
        x_pred = self.F @ self.x_est
        P_pred = self.F @ P_prev @ self.F.T + self.Q

        if z is not None:
            innovation = z - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + R
            K = P_pred @ self.H.T @ np.linalg.inv(S)

            self.x_est = x_pred + K @ innovation
            self.P_est = (np.eye(4) - K @ self.H) @ P_pred

            self.k += 1
            d_k = (1 - self.b) / (1 - self.b ** (self.k + 1))
            correction = K @ np.outer(innovation, innovation) @ K.T
            innovation_term = correction + self.P_est - self.F @ P_prev @ self.F.T
            self.Q = self._clip_psd((1 - d_k) * self.Q + d_k * innovation_term)
        else:
            self.x_est = x_pred
            self.P_est = P_pred
            # no innovation available this scan; Q estimate carried forward

        return self.x_est.copy()


def run_sage_husa(x_target_meas, y_target_meas, detected, r_meas, theta_meas,
                   measurement_covariance_fn, b=0.98):
    kf = SageHusaKF(b=b)
    first_idx = int(np.argmax(detected))
    kf.initialize(x_target_meas[first_idx], y_target_meas[first_idx])

    x_history = np.zeros((cfg.N, 4))
    P_history = np.zeros((cfg.N, 4, 4))
    for i in range(cfg.N):
        if detected[i]:
            z = np.array([x_target_meas[i], y_target_meas[i]])
            R = measurement_covariance_fn(r_meas[i], theta_meas[i])
            x_history[i] = kf.step(z, R)
        else:
            x_history[i] = kf.step(None, None)
        P_history[i] = kf.P_est
    return x_history, P_history
