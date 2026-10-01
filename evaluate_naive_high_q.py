"""
evaluate_naive_high_q.py  --  RESEARCH EXTENSION: the cheapest possible
classical response to a maneuvering target -- permanently set the
process noise to the high value already selected for IMM's
maneuvering mode, with no mode-switching, no learning, and no tuning
beyond that single value.

This answers a question IMM and the adaptive filter's added
complexity should have to justify itself against: does a trivially
simple fix already do the job during maneuvers? (Yes, roughly.) Where
does that simple fix actually lose to the more sophisticated
approaches? (The non-maneuvering steady state, where a permanently
elevated Q pays a real, consistent accuracy cost.)

Usage:
    python evaluate_naive_high_q.py --q_high 500 --n_runs 300
"""

import argparse
import pickle

import numpy as np

import config as cfg
from maneuver_truth_model import generate_maneuvering_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections
from kalman_filter import run_filter_with_covariance
from metrics import compute_nees


def evaluate(scenario, q_high, n_runs):
    errs, nees_all = [], []
    for run in range(n_runs):
        seed = cfg.BASE_SEED + run
        rng = np.random.default_rng(seed)
        x_true, y_true, vx_true, vy_true, man = generate_maneuvering_truth(scenario)
        x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
            generate_noisy_measurements(x_true, y_true, rng)
        x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)
        x_true_full = np.stack([x_true, y_true, vx_true, vy_true], axis=1)

        x_est, P_hist = run_filter_with_covariance(
            x_target_meas, y_target_meas, detected, r_meas, theta_meas,
            measurement_covariance, q_scale=q_high)
        errs.append(np.sqrt((x_est[:, 0] - x_true) ** 2 + (x_est[:, 1] - y_true) ** 2))
        nees_all.append(compute_nees(x_true_full, x_est, P_hist, state_dims=[0, 1]))

    errs, nees_all = np.array(errs), np.array(nees_all)
    return errs, nees_all


def main(q_high=500, n_runs=300, results_path="outputs/naive_high_q_results_300.pkl"):
    results = {}
    for scenario in ["cv_only", "mild_turn", "sharp_turn", "s_turn"]:
        errs, nees_all = evaluate(scenario, q_high, n_runs)
        rmse = np.sqrt(np.mean(errs ** 2))
        nees_mean = nees_all.mean()
        results[scenario] = {"err": errs, "nees": nees_all, "rmse": rmse, "nees_mean": nees_mean}
        print(f"{scenario:<12} naive-high-Q (q={q_high}):  RMSE={rmse:8.2f}m  NEES={nees_mean:7.2f}")

    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--q_high", type=float, default=500,
                    help="Fixed process-noise scale, same value already "
                         "selected for IMM's high-Q mode via select_q_high.py")
    p.add_argument("--n_runs", type=int, default=300)
    p.add_argument("--results_path", type=str, default="outputs/naive_high_q_results_300.pkl")
    args = p.parse_args()
    main(args.q_high, args.n_runs, args.results_path)
