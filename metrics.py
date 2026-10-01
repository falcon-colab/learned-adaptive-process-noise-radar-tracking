"""
metrics.py  --  RESEARCH EXTENSION (not part of the original EE-891
deliverable).

NEES (Normalized Estimation Error Squared) checks filter CONSISTENCY,
not just accuracy: it asks whether the filter's own reported
uncertainty (P) actually matches its real error. RMSE alone can't
catch this -- a filter can have low RMSE by luck while being wildly
overconfident (P too small) or underconfident (P too large). This is
exactly the failure mode a fixed-Q classical KF is expected to show
during a maneuver: it doesn't just get less accurate, it gets
CONFIDENTLY wrong, because P shrinks based on an assumption (constant
velocity) the target is actively violating.

NEES at scan k:  epsilon_k = (x_true_k - x_est_k)^T P_k^-1 (x_true_k - x_est_k)

For a consistent linear-Gaussian filter, E[epsilon_k] = n (the state
dimension). Averaged over N_mc independent runs, N_mc * nees_bar_k
follows a chi-squared distribution with N_mc * n degrees of freedom,
which gives a standard two-sided 95% consistency envelope to plot
alongside the average NEES curve.
"""

import numpy as np
from scipy import stats


def compute_nees(x_true_seq, x_est_seq, P_seq, state_dims=None):
    """NEES at every scan for one run.
    x_true_seq, x_est_seq: (N, n) arrays. P_seq: (N, n, n) array.
    state_dims: optional list of indices to evaluate NEES over (e.g.
        [0, 1] for position-only consistency instead of the full
        4-D state). Defaults to the full state.
    Returns an (N,) array of per-scan NEES values.
    """
    N = x_true_seq.shape[0]
    if state_dims is None:
        state_dims = list(range(x_true_seq.shape[1]))
    nees = np.zeros(N)
    for k in range(N):
        err = (x_true_seq[k, state_dims] - x_est_seq[k, state_dims])
        P_sub = P_seq[k][np.ix_(state_dims, state_dims)]
        nees[k] = err @ np.linalg.solve(P_sub, err)
    return nees


def nees_consistency_bounds(n_mc, n_state, alpha=0.05):
    """Two-sided (1-alpha) chi-squared consistency envelope for the
    MEAN NEES averaged over n_mc runs, given a state dimension
    n_state. Returns (lower, upper) bounds that the averaged NEES
    curve should stay within if the filter is consistent."""
    dof = n_mc * n_state
    lower = stats.chi2.ppf(alpha / 2, dof) / n_mc
    upper = stats.chi2.ppf(1 - alpha / 2, dof) / n_mc
    return lower, upper


def track_loss_rate(pos_err_seq_all_runs, threshold_m):
    """Fraction of Monte Carlo runs where position error ever exceeds
    threshold_m -- a simple, interpretable 'did the filter lose the
    target' metric to report alongside RMSE and NEES.
    pos_err_seq_all_runs: (n_runs, N) array of per-scan position errors.
    """
    lost = np.any(pos_err_seq_all_runs > threshold_m, axis=1)
    return lost.mean()
