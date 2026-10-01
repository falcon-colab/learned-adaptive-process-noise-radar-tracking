"""
phase1_checkpoint.py  --  RESEARCH EXTENSION, Phase 1 deliverable.

Goal: confirm, with the EXISTING classical fixed-Q Kalman filter (no
new ML yet), that:
  (a) it performs fine on the CV-only scenario (as already shown in the
      EE-891 deliverable),
  (b) it visibly degrades in RMSE during turn segments, and
  (c) that degradation shows up as a genuine CONSISTENCY failure (NEES
      exceeding its expected envelope), not just bigger error bars --
      i.e. the filter becomes confidently wrong, which is precisely the
      failure mode an adaptive-Q correction is meant to fix.

This uses maneuver_truth_model.py (new) together with the ORIGINAL,
unmodified measurement_model.py and detection_model.py -- proving those
two modules generalize to non-CV truth without any changes, exactly as
designed.
"""

import numpy as np
import matplotlib.pyplot as plt
import config as cfg

from maneuver_truth_model import generate_maneuvering_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections, generate_false_alarms
from kalman_filter import run_filter_with_covariance
from metrics import compute_nees, nees_consistency_bounds, track_loss_rate


def run_one_trial(scenario_name, seed):
    rng = np.random.default_rng(seed)
    x_true, y_true, vx_true, vy_true, maneuvering = generate_maneuvering_truth(scenario_name)

    x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
        generate_noisy_measurements(x_true, y_true, rng)
    x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)

    x_est, P_hist = run_filter_with_covariance(
        x_target_meas, y_target_meas, detected, r_meas, theta_meas,
        measurement_covariance
    )

    x_true_full = np.stack([x_true, y_true, vx_true, vy_true], axis=1)
    pos_err = np.sqrt((x_est[:, 0] - x_true)**2 + (x_est[:, 1] - y_true)**2)
    nees_pos = compute_nees(x_true_full, x_est, P_hist, state_dims=[0, 1])

    return pos_err, nees_pos, maneuvering


def run_monte_carlo_by_scenario(n_runs=100):
    results = {}
    for scenario in cfg.MANEUVER_SCENARIOS:
        all_err = np.zeros((n_runs, cfg.N))
        all_nees = np.zeros((n_runs, cfg.N))
        maneuvering = None
        for run in range(n_runs):
            seed = cfg.BASE_SEED + run
            pos_err, nees_pos, man = run_one_trial(scenario, seed)
            all_err[run] = pos_err
            all_nees[run] = nees_pos
            maneuvering = man
        results[scenario] = {
            "pos_err": all_err,
            "nees": all_nees,
            "maneuvering": maneuvering,
        }
    return results


def print_summary(results, n_runs):
    print(f"=== Phase 1 checkpoint: classical fixed-Q KF across scenarios ({n_runs} MC runs each) ===")
    lower, upper = nees_consistency_bounds(n_runs, n_state=2)
    print(f"95% NEES consistency envelope (position-only, 2 DOF): [{lower:.2f}, {upper:.2f}]")
    print(f"{'Scenario':<12} {'RMSE (m)':>10} {'Mean NEES':>10} {'Mean NEES (maneuver only)':>28} {'Track-loss@1km':>16}")
    for scenario, r in results.items():
        rmse = np.sqrt(np.mean(r["pos_err"]**2))
        mean_nees_all = r["nees"].mean()
        man_mask = r["maneuvering"]
        mean_nees_man = r["nees"][:, man_mask].mean() if man_mask.any() else float("nan")
        loss_rate = track_loss_rate(r["pos_err"], threshold_m=1000.0)
        print(f"{scenario:<12} {rmse:>10.2f} {mean_nees_all:>10.2f} {mean_nees_man:>28.2f} {loss_rate*100:>15.1f}%")


def plot_checkpoint(results, n_runs, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    k = np.arange(cfg.N)
    colors = {"cv_only": "g", "mild_turn": "b", "sharp_turn": "r"}

    ax = axes[0]
    for scenario, r in results.items():
        mean_err = r["pos_err"].mean(axis=0)
        ax.plot(k, mean_err, color=colors.get(scenario, "k"), label=scenario, linewidth=1.8)
        man = r["maneuvering"]
        if man.any():
            ax.fill_between(k, 0, mean_err.max() * 1.1, where=man,
                             color=colors.get(scenario, "k"), alpha=0.08)
    ax.set_ylabel("Mean position error (m)")
    ax.set_title(f"Classical fixed-Q KF: RMSE degrades during maneuvers ({n_runs} MC runs/scenario)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    lower, upper = nees_consistency_bounds(n_runs, n_state=2)
    for scenario, r in results.items():
        mean_nees = r["nees"].mean(axis=0)
        ax.plot(k, mean_nees, color=colors.get(scenario, "k"), label=scenario, linewidth=1.8)
    ax.axhline(2, color="gray", linestyle="--", linewidth=1, label="Expected (consistent) NEES = 2")
    ax.axhspan(lower, upper, color="gray", alpha=0.15, label="95% consistency envelope")
    ax.set_xlabel("Scan number"); ax.set_ylabel("Mean NEES (position, 2 DOF)")
    ax.set_title("Filter consistency: NEES exceeding the envelope means the filter is confidently wrong")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved {save_path}")
    plt.close()


if __name__ == "__main__":
    N_RUNS = 100
    results = run_monte_carlo_by_scenario(n_runs=N_RUNS)
    print_summary(results, N_RUNS)
    plot_checkpoint(results, N_RUNS, save_path="outputs/08_phase1_checkpoint.png")
