"""
adaptive_kf.py  --  RESEARCH EXTENSION, Phase 2: the proposed method.

AdaptiveKF wraps the SAME linear constant-velocity F/H used by the
classical ConstantVelocityKF (kalman_filter.py) -- no EKF/UKF, by
design (see paper rationale: a lightweight correction within a plain
linear KF, not a switch to nonlinear-filter machinery). The only
difference from the classical filter is that the process-noise scale
q_t is no longer a fixed constant -- it's predicted at every scan by a
tiny GRUCell reading the CAUSAL history of innovations (only past
information, so this is a valid online/real-time filter, not a
lookahead-cheating one).

Trained end-to-end (KalmanNet-style): gradients flow from a state-MSE
loss, backward through every predict/update step of the filter
recursion, into the GRUCell's weights. This requires the KF math to be
implemented in differentiable PyTorch ops (torch.linalg.solve etc.)
rather than plain numpy.
"""

import torch
import torch.nn as nn
import numpy as np
import config as cfg

T = cfg.T
F_np = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
H_np = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Q_UNIT_np = np.array([
    [T**3/3, 0,      T**2/2, 0],
    [0,      T**3/3, 0,      T**2/2],
    [T**2/2, 0,      T,      0],
    [0,      T**2/2, 0,      T],
])  # same white-noise-acceleration structure as the classical filter;
    # q_t (learned) multiplies this each step, exactly analogous to the
    # classical filter's fixed cfg.Q_SCALE -- this keeps the comparison
    # apples-to-apples: same Q structure, only the scale becomes adaptive.


class AdaptiveKF(nn.Module):
    def __init__(self, hidden_size=16, base_q=cfg.Q_SCALE, normalize_innovation=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.base_q = base_q
        # normalize_innovation=False (default) reproduces every checkpoint
        # trained earlier in this project unchanged: the GRU sees the raw,
        # unnormalized innovation in meters. normalize_innovation=True
        # instead whitens the innovation elementwise by sqrt(diag(S)),
        # the innovation covariance already computed each step for the
        # Kalman update -- a scale-invariant feature that should, in
        # principle, transfer more readily to noise regimes the network
        # was not trained under. This is a diagonal whitening (dividing
        # by each component's own standard deviation), not full
        # Mahalanobis whitening via S^{-1/2}, to stay lightweight and
        # keep the two variants differing in exactly one respect.
        self.normalize_innovation = normalize_innovation
        # Input per step: [innovation_x, innovation_y, detected_flag]
        # (raw or whitened, per normalize_innovation).
        self.gru = nn.GRUCell(input_size=3, hidden_size=hidden_size)
        self.head = nn.Linear(hidden_size, 1)
        # Initialize the head near zero so exp(0)=1 -> q_scale starts
        # at base_q, i.e. the untrained network behaves like the
        # classical filter -- a standard, important init trick for
        # training stability.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.F = torch.tensor(F_np, dtype=torch.float32)
        self.H = torch.tensor(H_np, dtype=torch.float32)
        self.Q_unit = torch.tensor(Q_UNIT_np, dtype=torch.float32)
        self.I4 = torch.eye(4)

    def forward(self, traj):
        """traj: dict of numpy arrays from dataset.build_trajectory.
        Returns (x_est_history, q_scale_history) as torch tensors,
        shapes (N, 4) and (N,). Fully differentiable w.r.t. self's
        parameters.
        """
        z = torch.tensor(traj["z"], dtype=torch.float32)
        R = torch.tensor(traj["R"], dtype=torch.float32)
        detected = traj["detected"]
        N = z.shape[0]

        x_est = torch.zeros(4)
        first_idx = int(np.argmax(detected))
        x_est = torch.tensor(
            [traj["z"][first_idx, 0], traj["z"][first_idx, 1], 0.0, 0.0],
            dtype=torch.float32,
        )
        P_est = torch.diag(torch.tensor(
            [cfg.P0_POS_STD**2, cfg.P0_POS_STD**2, cfg.P0_VEL_STD**2, cfg.P0_VEL_STD**2],
            dtype=torch.float32,
        ))
        h = torch.zeros(self.hidden_size)

        x_history = []
        q_history = []
        P_history = []

        for k in range(N):
            # --- q_t predicted from PAST hidden state only (causal) ---
            raw = self.head(h).squeeze(0) if h.dim() > 1 else self.head(h)
            q_scale = self.base_q * torch.exp(raw.squeeze())
            q_history.append(q_scale)

            Q_k = q_scale * self.Q_unit

            # --- predict ---
            x_pred = self.F @ x_est
            P_pred = self.F @ P_est @ self.F.T + Q_k

            if detected[k]:
                innovation = z[k] - self.H @ x_pred
                S = self.H @ P_pred @ self.H.T + R[k]
                K = P_pred @ self.H.T @ torch.linalg.inv(S)
                x_est = x_pred + K @ innovation
                P_est = (self.I4 - K @ self.H) @ P_pred
                if self.normalize_innovation:
                    innov_feature = innovation / torch.sqrt(torch.diagonal(S) + 1e-8)
                else:
                    innov_feature = innovation
                gru_input = torch.cat([innov_feature, torch.tensor([1.0])])
            else:
                x_est = x_pred
                P_est = P_pred
                gru_input = torch.zeros(3)

            x_history.append(x_est)
            P_history.append(P_est)

            # --- update hidden state for NEXT step's q prediction ---
            h = self.gru(gru_input.unsqueeze(0), h.unsqueeze(0) if h.dim() == 1 else h).squeeze(0)

        return torch.stack(x_history), torch.stack(q_history), torch.stack(P_history)


def run_adaptive_filter_numpy(model, traj):
    """Inference-mode helper: runs the trained model on one trajectory
    and returns plain numpy arrays, for reuse with the existing
    (numpy-based) metrics.py / plotting code. Now also returns P
    history so the adaptive filter's own NEES/consistency can be
    computed on equal footing with the classical filter."""
    model.eval()
    with torch.no_grad():
        x_hist, q_hist, P_hist = model(traj)
    return x_hist.numpy(), q_hist.numpy(), P_hist.numpy()
