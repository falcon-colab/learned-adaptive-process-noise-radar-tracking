"""
evaluate_s_turn.py  --  RESEARCH EXTENSION: evaluates classical, IMM,
and the adaptive filter on the held-out S-turn scenario (config.py's
"s_turn" entry): two coordinated turns in opposite directions in
sequence. This scenario is structurally different from anything either
the adaptive filter's training data or IMM's high-Q selection ever
saw (both used only single-direction turns), so it is a genuine test
of generalization to a novel maneuver PATTERN, distinct from the
Pd-generalization sweep in severity_sweep.py, which tests robustness
to a different SEVERITY of the same, already-seen turn pattern.

Usage:
    python evaluate_s_turn.py --checkpoint outputs/adaptive_kf_standardized_1400.pt
"""

import argparse
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

import config as cfg
from adaptive_kf import AdaptiveKF, run_adaptive_filter_numpy
from maneuver_truth_model import generate_maneuvering_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections
from kalman_filter import run_filter_with_covariance
from imm_filter import run_imm
from metrics import compute_nees


def run_trial(model, seed, q_high=500):
    rng = np.random.default_rng(seed)
    x_true, y_true, vx_true, vy_true, man = generate_maneuvering_truth("s_turn")
    x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
        generate_noisy_measurements(x_true, y_true, rng)
    x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)
    x_true_full = np.stack([x_true, y_true, vx_true, vy_true], axis=1)

    x_est_c, P_hist_c = run_filter_with_covariance(
        x_target_meas, y_target_meas, detected, r_meas, theta_meas, measurement_covariance)
    err_c = np.sqrt((x_est_c[:, 0] - x_true) ** 2 + (x_est_c[:, 1] - y_true) ** 2)
    nees_c = compute_nees(x_true_full, x_est_c, P_hist_c, state_dims=[0, 1])

    x_est_imm, P_hist_imm, _ = run_imm(
        x_target_meas, y_target_meas, detected, r_meas, theta_meas,
        measurement_covariance, q_high=q_high)
    err_imm = np.sqrt((x_est_imm[:, 0] - x_true) ** 2 + (x_est_imm[:, 1] - y_true) ** 2)
    nees_imm = compute_nees(x_true_full, x_est_imm, P_hist_imm, state_dims=[0, 1])

    R_seq = np.zeros((cfg.N, 2, 2))
    for k in range(cfg.N):
        if detected[k]:
            R_seq[k] = measurement_covariance(r_meas[k], theta_meas[k])
    z_seq = np.stack([np.where(detected, x_target_meas, 0.0),
                       np.where(detected, y_target_meas, 0.0)], axis=1)
    traj = {"x_true": x_true_full, "z": z_seq, "R": R_seq, "detected": detected}
    x_est_a, _, P_hist_a = run_adaptive_filter_numpy(model, traj)
    err_a = np.sqrt((x_est_a[:, 0] - x_true) ** 2 + (x_est_a[:, 1] - y_true) ** 2)
    nees_a = compute_nees(x_true_full, x_est_a, P_hist_a, state_dims=[0, 1])

    return err_c, err_imm, err_a, nees_c, nees_imm, nees_a


def main(checkpoint, n_runs=150, hidden_size=16, q_high=500):
    model = AdaptiveKF(hidden_size=hidden_size)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    errs_c, errs_imm, errs_a = [], [], []
    nees_c, nees_imm, nees_a = [], [], []
    for run in range(n_runs):
        seed = cfg.BASE_SEED + run
        ec, ei, ea, nc, ni, na = run_trial(model, seed, q_high=q_high)
        errs_c.append(ec); errs_imm.append(ei); errs_a.append(ea)
        nees_c.append(nc); nees_imm.append(ni); nees_a.append(na)

    errs_c, errs_imm, errs_a = np.array(errs_c), np.array(errs_imm), np.array(errs_a)
    nees_c, nees_imm, nees_a = np.array(nees_c), np.array(nees_imm), np.array(nees_a)

    print(f"=== S-turn (held-out, structurally novel maneuver), {n_runs} MC runs ===")
    print(f"Classical: RMSE={np.sqrt(np.mean(errs_c**2)):.1f}m  NEES={nees_c.mean():.2f}")
    print(f"IMM:       RMSE={np.sqrt(np.mean(errs_imm**2)):.1f}m  NEES={nees_imm.mean():.2f}")
    print(f"Adaptive:  RMSE={np.sqrt(np.mean(errs_a**2)):.1f}m  NEES={nees_a.mean():.2f}")

    with open("outputs/s_turn_results.pkl", "wb") as f:
        pickle.dump({
            "classical_err": errs_c, "imm_err": errs_imm, "adaptive_err": errs_a,
            "classical_nees": nees_c, "imm_nees": nees_imm, "adaptive_nees": nees_a,
        }, f)

    k = np.arange(cfg.N)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k, errs_c.mean(axis=0), "r-", alpha=0.6,
            label=f"Classical (RMSE={np.sqrt(np.mean(errs_c**2)):.0f}m)")
    ax.plot(k, errs_imm.mean(axis=0), color="purple", linewidth=1.8,
            label=f"IMM (RMSE={np.sqrt(np.mean(errs_imm**2)):.0f}m)")
    ax.plot(k, errs_a.mean(axis=0), "b-", linewidth=1.8,
            label=f"Adaptive (RMSE={np.sqrt(np.mean(errs_a**2)):.0f}m)")
    ax.set_xlabel("Scan number"); ax.set_ylabel("Mean position error (m)")
    ax.set_title("Held-out S-turn scenario (never seen by either filter during training/tuning)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/16_s_turn_comparison.png", dpi=120)
    print("Saved outputs/16_s_turn_comparison.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="outputs/adaptive_kf_standardized_1400.pt")
    p.add_argument("--n_runs", type=int, default=150)
    p.add_argument("--hidden_size", type=int, default=16)
    p.add_argument("--q_high", type=float, default=500)
    args = p.parse_args()
    main(args.checkpoint, args.n_runs, args.hidden_size, args.q_high)
