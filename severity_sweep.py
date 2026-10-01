"""
severity_sweep.py  --  RESEARCH EXTENSION, final phase: sensitivity to
detection probability and clutter rate.

Two distinct experiments, for a reason worth stating explicitly: our
single-target filter never consumes clutter in its update step (see
System Model, Section III-C), so clutter rate should have NO effect on
RMSE/NEES by construction. Detection probability, by contrast, directly
changes how often the filter must coast on prediction alone, so it is
the axis where a real effect is expected.

Classical, adaptive, AND IMM are all evaluated. The classical and
adaptive filters are the ALREADY-TRAINED checkpoints from the main
experiment (trained only at the baseline Pd=0.92, Lambda=1.5).
Evaluating them at OTHER severities they never saw during training is a
genuine generalization test, not a repeat of the headline result. IMM
is not "trained" in that sense -- its mode probabilities are Bayesian
updates from measurement likelihood at whatever Pd is actually present,
same as the naive filter's unconditional generalization across Pd by
construction -- but it IS evaluated at the same swept conditions here
for a fair, complete three-way comparison, using the same q_high=500
selected via the systematic grid search in select_q_high.py (not
re-tuned for this sweep).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import config as cfg

from maneuver_truth_model import generate_maneuvering_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections, generate_false_alarms
from kalman_filter import run_filter_with_covariance
from adaptive_kf import AdaptiveKF, run_adaptive_filter_numpy
from imm_filter import run_imm
from metrics import compute_nees


def run_trial(scenario, seed, pd_override=None, q_high=500):
    rng = np.random.default_rng(seed)
    x_true, y_true, vx_true, vy_true, maneuvering = generate_maneuvering_truth(scenario)
    x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
        generate_noisy_measurements(x_true, y_true, rng)
    x_target_meas, y_target_meas, detected = apply_missed_detections(
        x_meas, y_meas, rng, pd=pd_override)

    x_true_full = np.stack([x_true, y_true, vx_true, vy_true], axis=1)

    # classical
    x_est_c, P_hist_c = run_filter_with_covariance(
        x_target_meas, y_target_meas, detected, r_meas, theta_meas, measurement_covariance)
    err_c = np.sqrt((x_est_c[:, 0] - x_true)**2 + (x_est_c[:, 1] - y_true)**2)
    nees_c = compute_nees(x_true_full, x_est_c, P_hist_c, state_dims=[0, 1])

    # adaptive
    R_seq = np.zeros((cfg.N, 2, 2))
    for k in range(cfg.N):
        if detected[k]:
            R_seq[k] = measurement_covariance(r_meas[k], theta_meas[k])
    z_seq = np.stack([np.where(detected, x_target_meas, 0.0),
                       np.where(detected, y_target_meas, 0.0)], axis=1)
    traj = {"x_true": x_true_full, "z": z_seq, "R": R_seq, "detected": detected}
    x_est_a, _, P_hist_a = run_adaptive_filter_numpy(adaptive_model, traj)
    err_a = np.sqrt((x_est_a[:, 0] - x_true)**2 + (x_est_a[:, 1] - y_true)**2)
    nees_a = compute_nees(x_true_full, x_est_a, P_hist_a, state_dims=[0, 1])

    # IMM
    x_est_imm, P_hist_imm, _ = run_imm(
        x_target_meas, y_target_meas, detected, r_meas, theta_meas,
        measurement_covariance, q_high=q_high)
    err_imm = np.sqrt((x_est_imm[:, 0] - x_true)**2 + (x_est_imm[:, 1] - y_true)**2)
    nees_imm = compute_nees(x_true_full, x_est_imm, P_hist_imm, state_dims=[0, 1])

    return err_c, nees_c, err_a, nees_a, err_imm, nees_imm


def sweep_pd(pd_values, scenarios, n_runs=150, q_high=500):
    results = {}
    for scenario in scenarios:
        for pd in pd_values:
            errs_c, nees_c_all, errs_a, nees_a_all, errs_imm, nees_imm_all = [], [], [], [], [], []
            for run in range(n_runs):
                seed = cfg.BASE_SEED + run
                ec, nc, ea, na, ei, ni = run_trial(scenario, seed, pd_override=pd, q_high=q_high)
                errs_c.append(ec); nees_c_all.append(nc)
                errs_a.append(ea); nees_a_all.append(na)
                errs_imm.append(ei); nees_imm_all.append(ni)
            errs_c, errs_a, errs_imm = np.array(errs_c), np.array(errs_a), np.array(errs_imm)
            rmse_c = np.sqrt(np.mean(errs_c**2))
            rmse_a = np.sqrt(np.mean(errs_a**2))
            rmse_imm = np.sqrt(np.mean(errs_imm**2))
            results[(scenario, pd)] = {
                "rmse_c": rmse_c, "rmse_a": rmse_a, "rmse_imm": rmse_imm,
                "nees_c": np.mean(nees_c_all), "nees_a": np.mean(nees_a_all),
                "nees_imm": np.mean(nees_imm_all),
                "improvement": (1 - rmse_a / rmse_c) * 100,
                "improvement_imm": (1 - rmse_imm / rmse_c) * 100,
            }
            print(f"{scenario:<10} Pd={pd:.2f}  classical={rmse_c:7.1f}m  "
                  f"adaptive={rmse_a:7.1f}m  IMM={rmse_imm:7.1f}m  "
                  f"NEES(c/a/imm)={np.mean(nees_c_all):7.1f}/{np.mean(nees_a_all):5.2f}/{np.mean(nees_imm_all):5.2f}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="outputs/adaptive_kf_standardized_1400.pt",
                    help="Headline checkpoint to evaluate. Defaults to the standardized-"
                         "protocol headline model (Table I's primary reported result). "
                         "The original ad-hoc-protocol checkpoint (outputs/adaptive_kf_weights.pt, "
                         "if you have it from an earlier run) can be passed instead for "
                         "comparison, but is NOT produced by a fresh reproduce_all.py run.")
    p.add_argument("--n_runs", type=int, default=150)
    p.add_argument("--q_high", type=float, default=500,
                    help="IMM high-Q mode value, same one selected via select_q_high.py's "
                         "systematic grid search and used everywhere else IMM appears in "
                         "this project -- not re-tuned for this sweep.")
    p.add_argument("--results_path", type=str, default="outputs/severity_sweep_pd_results.pkl")
    args = p.parse_args()

    adaptive_model = AdaptiveKF(hidden_size=16)
    adaptive_model.load_state_dict(torch.load(args.checkpoint))
    adaptive_model.eval()

    print(f"=== Detection-probability sweep (checkpoint: {args.checkpoint}, trained only at Pd=0.92) ===")
    pd_values = [0.92, 0.75, 0.5, 0.3]
    scenarios = ["cv_only", "mild_turn", "sharp_turn"]
    pd_results = sweep_pd(pd_values, scenarios, n_runs=args.n_runs, q_high=args.q_high)

    import pickle
    with open(args.results_path, "wb") as f:
        pickle.dump(pd_results, f)
    print(f"Saved {args.results_path}")
