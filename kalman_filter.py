"""
kalman_filter.py  --  Task 7: Basic Kalman filter for the constant-
velocity target.

State:        x = [x, y, vx, vy]^T
Transition:    x_k = F x_{k-1} + w_k,   w_k ~ N(0, Q)
Measurement:   z_k = H x_k + v_k,       v_k ~ N(0, R_k)   (R varies per
                                                            scan -- see
                                                            measurement_model.py)

Standard five-equation KF:
  Predict:  x_pred = F x_est ;  P_pred = F P_est F^T + Q
  Innovation:  y = z - H x_pred
  Gain:     K = P_pred H^T (H P_pred H^T + R)^-1
  Update:   x_est = x_pred + K y ;  P_est = (I - K H) P_pred

On a missed-detection scan, the update step is skipped -- the filter
simply carries the prediction forward (this is the correct Bayesian
thing to do when there's no new information).
"""

import numpy as np
import config as cfg


class ConstantVelocityKF:
    def __init__(self):
        self.F = np.array([[1, 0, cfg.T, 0],
                            [0, 1, 0, cfg.T],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        q = cfg.Q_SCALE
        T = cfg.T
        self.Q = q * np.array([
            [T**3/3, 0,      T**2/2, 0],
            [0,      T**3/3, 0,      T**2/2],
            [T**2/2, 0,      T,      0],
            [0,      T**2/2, 0,      T],
        ])
        self.x_est = None
        self.P_est = None

    def initialize(self, x0, y0):
        self.x_est = np.array([x0, y0, 0.0, 0.0])
        self.P_est = np.diag([cfg.P0_POS_STD**2, cfg.P0_POS_STD**2,
                               cfg.P0_VEL_STD**2, cfg.P0_VEL_STD**2])

    def step(self, z=None, R=None):
        """Advance the filter by one scan.
        z: measurement [x, y], or None if this scan was missed.
        R: 2x2 measurement covariance for this scan (required if z given).
        Returns the updated state estimate (length-4 array).
        """
        x_pred = self.F @ self.x_est
        P_pred = self.F @ self.P_est @ self.F.T + self.Q

        if z is not None:
            innovation = z - self.H @ x_pred
            S = self.H @ P_pred @ self.H.T + R
            K = P_pred @ self.H.T @ np.linalg.inv(S)

            self.x_est = x_pred + K @ innovation
            self.P_est = (np.eye(4) - K @ self.H) @ P_pred
        else:
            self.x_est = x_pred
            self.P_est = P_pred

        return self.x_est.copy()


def run_filter(x_target_meas, y_target_meas, detected, r_meas, theta_meas,
                measurement_covariance_fn):
    """Runs the KF over a full scenario. Returns arrays of estimated
    (x, y, vx, vy) for every scan."""
    kf = ConstantVelocityKF()
    first_idx = int(np.argmax(detected))  # first True index
    kf.initialize(x_target_meas[first_idx], y_target_meas[first_idx])

    history = np.zeros((cfg.N, 4))
    for i in range(cfg.N):
        if detected[i]:
            z = np.array([x_target_meas[i], y_target_meas[i]])
            R = measurement_covariance_fn(r_meas[i], theta_meas[i])
            history[i] = kf.step(z, R)
        else:
            history[i] = kf.step(None, None)
    return history


# =========================================================================
# RESEARCH EXTENSION (ICCI paper): covariance-tracking variant of
# run_filter, needed for NEES consistency analysis (see metrics.py).
# The original run_filter above is untouched -- this is purely additive,
# so the original EE-891 deliverable stays reproducible exactly as-is.
# =========================================================================

def run_filter_with_covariance(x_target_meas, y_target_meas, detected,
                                 r_meas, theta_meas, measurement_covariance_fn,
                                 q_scale=None):
    """Same as run_filter, but also returns the full P history, and
    optionally overrides Q_SCALE for this run only (used later by the
    adaptive-Q filter comparison, and by the q-mismatch experiments) --
    without touching the original ConstantVelocityKF default behavior.
    Returns (x_history, P_history) where P_history has shape (N, 4, 4).
    """
    kf = ConstantVelocityKF()
    if q_scale is not None:
        q = q_scale
        T = cfg.T
        kf.Q = q * np.array([
            [T**3/3, 0,      T**2/2, 0],
            [0,      T**3/3, 0,      T**2/2],
            [T**2/2, 0,      T,      0],
            [0,      T**2/2, 0,      T],
        ])
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
