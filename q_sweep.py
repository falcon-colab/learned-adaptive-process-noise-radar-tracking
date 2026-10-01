"""
q_sweep.py  --  Optional add-on: sensitivity of Kalman filter performance
to the process-noise scale q.

q controls how much the filter TRUSTS its own constant-velocity motion
model vs. the incoming measurements:
  - q too small  -> filter becomes overconfident in the motion model,
                    responds sluggishly to real measurement information,
                    P shrinks and the filter can "lock in" on a biased
                    estimate.
  - q too large  -> filter barely trusts its own prediction, ends up
                    tracking the noisy measurements almost directly
                    (defeats the purpose of filtering).
  - There's a sweet spot in between that minimizes RMSE -- this script
    finds and plots it, run over multiple Monte Carlo trials per q value
    so the curve isn't just noise.

This directly supports a stronger Task 7/8 discussion: instead of just
asserting q=0.5 "worked", you can show WHY it's a reasonable choice.
"""

import numpy as np
import matplotlib.pyplot as plt
import config as cfg

from truth_model import generate_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections, generate_false_alarms
from kalman_filter import run_filter


def rmse_for_q(q_value, n_trials=50, base_seed=cfg.BASE_SEED):
    """Runs n_trials independent scenarios with Q_SCALE overridden to
    q_value, returns the mean RMSE of the KF estimate across trials."""
    # Temporarily override the module-level Q_SCALE that kalman_filter.py
    # reads at ConstantVelocityKF construction time.
    original_q = cfg.Q_SCALE
    cfg.Q_SCALE = q_value

    rmse_list = np.zeros(n_trials)
    x_true, y_true, _, _ = generate_truth()

    for trial in range(n_trials):
        rng = np.random.default_rng(base_seed + trial)
        x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
            generate_noisy_measurements(x_true, y_true, rng)
        x_target_meas, y_target_meas, detected = \
            apply_missed_detections(x_meas, y_meas, rng)

        kf_history = run_filter(x_target_meas, y_target_meas, detected,
                                 r_meas, theta_meas, measurement_covariance)
        x_kf, y_kf = kf_history[:, 0], kf_history[:, 1]
        err = np.sqrt((x_kf - x_true)**2 + (y_kf - y_true)**2)
        rmse_list[trial] = np.sqrt(np.mean(err**2))

    cfg.Q_SCALE = original_q  # restore, so this function has no side effects
    return rmse_list.mean(), rmse_list.std()


def run_q_sweep(q_values=None, n_trials=50):
    if q_values is None:
        q_values = np.logspace(-3, 3, 25)  # 0.001 to 1000, log-spaced

    mean_rmse = np.zeros(len(q_values))
    std_rmse = np.zeros(len(q_values))

    for i, q in enumerate(q_values):
        mean_rmse[i], std_rmse[i] = rmse_for_q(q, n_trials=n_trials)
        print(f"q={q:10.4f}  ->  mean RMSE={mean_rmse[i]:8.2f} m  "
              f"(std across trials={std_rmse[i]:.2f})")

    return q_values, mean_rmse, std_rmse


def plot_q_sweep(q_values, mean_rmse, std_rmse, save_path=None):
    best_idx = np.argmin(mean_rmse)

    plt.figure(figsize=(8, 6))
    plt.errorbar(q_values, mean_rmse, yerr=std_rmse, fmt='o-', capsize=3,
                 color='b', ecolor='b', alpha=0.7, markersize=4)
    plt.axvline(q_values[best_idx], color='green', linestyle='--',
                label=f'Best q ≈ {q_values[best_idx]:.3g} '
                      f'(RMSE={mean_rmse[best_idx]:.1f} m)')
    plt.axvline(cfg.Q_SCALE, color='red', linestyle=':',
                label=f'Project default q={cfg.Q_SCALE} '
                      f'(RMSE={mean_rmse[np.argmin(np.abs(q_values - cfg.Q_SCALE))]:.1f} m)')
    plt.xscale('log')
    plt.xlabel('Process noise scale q (log scale)')
    plt.ylabel('Mean KF RMSE over trials (m)')
    plt.title('Kalman filter sensitivity to process-noise tuning (q)')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved {save_path}")
    plt.close()

    print(f"\nBest-performing q in sweep: {q_values[best_idx]:.4g} "
          f"(mean RMSE {mean_rmse[best_idx]:.2f} m)")
    print(f"Project's chosen default q={cfg.Q_SCALE} gives mean RMSE "
          f"{mean_rmse[np.argmin(np.abs(q_values - cfg.Q_SCALE))]:.2f} m")


if __name__ == "__main__":
    print("Running q-sweep (this takes a little longer -- multiple trials per q value)...")
    q_values, mean_rmse, std_rmse = run_q_sweep()
    plot_q_sweep(q_values, mean_rmse, std_rmse,
                 save_path="outputs/06_q_sweep_sensitivity.png")
