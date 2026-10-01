"""
select_q_high.py  --  Systematic selection of the IMM's high-Q
(maneuvering-mode) process-noise value, via grid search over candidate
values on the two maneuvering scenarios. This replaces a hand-picked
value with a documented, reproducible procedure: the reported q_high
is whichever candidate minimizes mean RMSE across mild_turn and
sharp_turn, averaged over independent Monte Carlo trials.

This mirrors the same methodology already used and validated for the
classical filter's q in the original EE-891 project's q_sweep.py,
now applied to the maneuvering scenarios instead of cv_only.
"""

import numpy as np
import config as cfg

from maneuver_truth_model import generate_maneuvering_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections
from imm_filter import run_imm


def rmse_for_q_high(q_high, n_trials=40):
    """Mean RMSE across mild_turn and sharp_turn for a given q_high,
    averaged over n_trials independent Monte Carlo runs per scenario."""
    all_rmse = []
    for scenario in ["mild_turn", "sharp_turn"]:
        errs = []
        for trial in range(n_trials):
            seed = cfg.BASE_SEED + trial
            rng = np.random.default_rng(seed)
            x_true, y_true, vx_true, vy_true, man = generate_maneuvering_truth(scenario)
            x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
                generate_noisy_measurements(x_true, y_true, rng)
            x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)
            x_hist, _, _ = run_imm(x_target_meas, y_target_meas, detected, r_meas, theta_meas,
                                    measurement_covariance, q_high=q_high)
            err = np.sqrt((x_hist[:, 0] - x_true)**2 + (x_hist[:, 1] - y_true)**2)
            errs.append(err)
        errs = np.array(errs)
        all_rmse.append(np.sqrt(np.mean(errs**2)))
    return np.mean(all_rmse)  # combined mild_turn + sharp_turn RMSE


if __name__ == "__main__":
    candidates = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    results = {}
    for q in candidates:
        rmse = rmse_for_q_high(q, n_trials=40)
        results[q] = rmse
        print(f"q_high={q:6.0f}  combined mild+sharp RMSE={rmse:8.2f} m")

    best_q = min(results, key=results.get)
    print(f"\nSelected q_high = {best_q} (minimizes combined RMSE = {results[best_q]:.2f} m)")

    import json
    with open("outputs/q_high_selection.json", "w") as f:
        json.dump({"candidates": results, "selected": best_q}, f, indent=2)
