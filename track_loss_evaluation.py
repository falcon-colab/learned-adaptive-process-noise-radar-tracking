"""
track_loss_evaluation.py  --  Computes track-loss rate (fraction of Monte
Carlo runs where position error ever exceeds a threshold) for the naive,
IMM, and adaptive filters across the three headline scenarios (Table 1).

Reuses the exact same evaluation machinery that generated Table 1's
RMSE/NEES numbers (evaluate_on_fixed_scenarios for naive/adaptive,
run_imm for IMM, same seeds, same n_runs), so results are guaranteed
consistent with the already-published table rather than a fresh,
possibly-different re-evaluation.

Threshold: 1000 m, matching the convention already established in
phase1_checkpoint.py for the original EE-891 baseline.

Added in response to an external review noting the paper's Experimental
Setup section claimed track-loss rate was "all reported" alongside
RMSE and NEES, but no table in an earlier draft actually contained it --
a real, valid gap, fixed here by building the infrastructure properly
rather than removing the claim.
"""
import pickle

import numpy as np
import torch

import config as cfg
from adaptive_kf import AdaptiveKF
from train_phase2 import evaluate_on_fixed_scenarios
from maneuver_truth_model import generate_maneuvering_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections
from imm_filter import run_imm
from metrics import track_loss_rate

THRESHOLD_M = 1000.0


def evaluate_imm_errs(scenario, n_runs, q_high=500):
    """Returns an (n_runs, N) array of per-scan position errors for IMM,
    using the identical seed formula as evaluate_on_fixed_scenarios."""
    errs = np.zeros((n_runs, cfg.N))
    for run in range(n_runs):
        seed = cfg.BASE_SEED + run
        rng = np.random.default_rng(seed)
        x_true, y_true, vx_true, vy_true, man = generate_maneuvering_truth(scenario)
        x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
            generate_noisy_measurements(x_true, y_true, rng)
        x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)
        x_est_imm, P_hist_imm, _ = run_imm(
            x_target_meas, y_target_meas, detected, r_meas, theta_meas,
            measurement_covariance, q_high=q_high)
        errs[run] = np.sqrt((x_est_imm[:, 0] - x_true) ** 2 + (x_est_imm[:, 1] - y_true) ** 2)
    return errs


def main(checkpoint="outputs/adaptive_kf_standardized_1400.pt", n_runs=300):
    model = AdaptiveKF(hidden_size=16)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    print(f"=== Track-loss rate (threshold={THRESHOLD_M:.0f} m), {n_runs} runs per scenario ===")
    results = {}
    r = evaluate_on_fixed_scenarios(model, n_runs=n_runs)
    for scenario in ["cv_only", "mild_turn", "sharp_turn"]:
        naive_rate = track_loss_rate(r[scenario]["classical_err"], THRESHOLD_M)
        adaptive_rate = track_loss_rate(r[scenario]["adaptive_err"], THRESHOLD_M)
        imm_errs = evaluate_imm_errs(scenario, n_runs)
        imm_rate = track_loss_rate(imm_errs, THRESHOLD_M)
        results[scenario] = {"naive": naive_rate, "imm": imm_rate, "adaptive": adaptive_rate}
        print(f"{scenario:<12} naive={naive_rate*100:5.1f}%  IMM={imm_rate*100:5.1f}%  "
              f"adaptive={adaptive_rate*100:5.1f}%")

    with open("outputs/track_loss_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved outputs/track_loss_results.pkl")


if __name__ == "__main__":
    main()
