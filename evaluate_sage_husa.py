"""
evaluate_sage_husa.py  --  Select the Sage-Husa forgetting factor b by
the same systematic grid-search methodology already used for IMM's
q_high (select_q_high.py: minimize combined mild_turn+sharp_turn RMSE
over 40 trials), then run the full 300-run evaluation across all three
fixed scenarios, computing RMSE/NEES/track-loss exactly as for every
other baseline in the paper (same seeds, same metrics.py functions).
"""

import json
import pickle

import numpy as np

import config as cfg
from maneuver_truth_model import generate_maneuvering_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections
from sage_husa_kf import run_sage_husa
from metrics import compute_nees, nees_consistency_bounds, track_loss_rate


def rmse_for_b(b, n_trials=40):
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
            x_hist, _ = run_sage_husa(x_target_meas, y_target_meas, detected, r_meas, theta_meas,
                                       measurement_covariance, b=b)
            err = np.sqrt((x_hist[:, 0] - x_true) ** 2 + (x_hist[:, 1] - y_true) ** 2)
            errs.append(err)
        errs = np.array(errs)
        all_rmse.append(np.sqrt(np.mean(errs ** 2)))
    return float(np.mean(all_rmse))


def select_b():
    candidates = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.999]
    results = {}
    for b in candidates:
        rmse = rmse_for_b(b, n_trials=40)
        results[b] = rmse
        print(f"b={b:6.3f}  combined mild+sharp RMSE={rmse:8.2f} m")
    best_b = min(results, key=results.get)
    print(f"\nSelected b = {best_b} (minimizes combined RMSE = {results[best_b]:.2f} m)")
    with open("outputs/sage_husa_b_selection.json", "w") as f:
        json.dump({"candidates": results, "selected": best_b}, f, indent=2)
    return best_b


def evaluate_full(b, n_runs=300):
    results = {}
    for scenario in cfg.MANEUVER_SCENARIOS:
        err = np.zeros((n_runs, cfg.N))
        nees = np.zeros((n_runs, cfg.N))
        for run in range(n_runs):
            seed = cfg.BASE_SEED + run  # SAME eval seeds as every other table
            rng = np.random.default_rng(seed)
            x_true, y_true, vx_true, vy_true, man = generate_maneuvering_truth(scenario)
            x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
                generate_noisy_measurements(x_true, y_true, rng)
            x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)
            x_true_full = np.stack([x_true, y_true, vx_true, vy_true], axis=1)

            x_hist, P_hist = run_sage_husa(x_target_meas, y_target_meas, detected,
                                            r_meas, theta_meas, measurement_covariance, b=b)
            err[run] = np.sqrt((x_hist[:, 0] - x_true) ** 2 + (x_hist[:, 1] - y_true) ** 2)
            nees[run] = compute_nees(x_true_full, x_hist, P_hist, state_dims=[0, 1])

        results[scenario] = {"err": err, "nees": nees}
    return results


def print_table(results, n_runs):
    lower, upper = nees_consistency_bounds(n_runs, n_state=2)
    print(f"\n=== Sage-Husa adaptive-Q baseline, {n_runs} MC runs/scenario ===")
    print(f"95% NEES consistency envelope (2 DOF): [{lower:.2f}, {upper:.2f}]")
    print(f"{'Scenario':<12} {'RMSE':>10} {'NEES':>8} {'TL(1000m)':>10}")
    for scenario, r in results.items():
        rmse = np.sqrt(np.mean(r["err"] ** 2))
        nees_mean = r["nees"].mean()
        tl = track_loss_rate(r["err"], 1000.0) * 100
        print(f"{scenario:<12} {rmse:>9.2f}m {nees_mean:>8.2f} {tl:>9.1f}%")


if __name__ == "__main__":
    best_b = select_b()
    results = evaluate_full(best_b, n_runs=300)
    print_table(results, 300)
    with open("outputs/sage_husa_results_300.pkl", "wb") as f:
        pickle.dump({"b": best_b, "results": results}, f)
    print("\nSaved outputs/sage_husa_results_300.pkl")
