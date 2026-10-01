"""
run_single_scenario.py  --  Task 5 (construct radar measurements) +
runs the KF on one full scenario, for demo purposes: this is the script
to run live during your demonstration, since it produces the clearest
"before vs after" picture in one shot.
"""

import numpy as np
import matplotlib.pyplot as plt
import config as cfg

from truth_model import generate_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections, generate_false_alarms
from kalman_filter import run_filter


def run_scenario(seed=cfg.BASE_SEED):
    """Runs one complete scenario end-to-end. Returns a dict with every
    intermediate signal, so callers (demo script or Monte Carlo loop)
    can use whichever pieces they need."""
    rng = np.random.default_rng(seed)

    x_true, y_true, vx_true, vy_true = generate_truth()

    x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
        generate_noisy_measurements(x_true, y_true, rng)

    x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)
    false_alarms = generate_false_alarms(x_true, y_true, rng)

    kf_history = run_filter(x_target_meas, y_target_meas, detected,
                             r_meas, theta_meas, measurement_covariance)

    return {
        "x_true": x_true, "y_true": y_true,
        "vx_true": vx_true, "vy_true": vy_true,
        "x_meas": x_meas, "y_meas": y_meas,
        "r_meas": r_meas, "theta_meas": theta_meas,
        "r_true": r_true, "theta_true": theta_true,
        "x_target_meas": x_target_meas, "y_target_meas": y_target_meas,
        "detected": detected,
        "false_alarms": false_alarms,
        "kf_history": kf_history,
    }


def compute_rmse(result):
    x_true, y_true = result["x_true"], result["y_true"]
    x_meas, y_meas = result["x_meas"], result["y_meas"]
    x_kf, y_kf = result["kf_history"][:, 0], result["kf_history"][:, 1]

    meas_err = np.sqrt((x_meas - x_true)**2 + (y_meas - y_true)**2)
    kf_err = np.sqrt((x_kf - x_true)**2 + (y_kf - y_true)**2)

    rmse_meas = np.sqrt(np.mean(meas_err**2))
    rmse_kf = np.sqrt(np.mean(kf_err**2))
    return meas_err, kf_err, rmse_meas, rmse_kf


def plot_full_result(result, meas_err, kf_err, rmse_meas, rmse_kf, save_path=None):
    x_true, y_true = result["x_true"], result["y_true"]
    x_meas, y_meas = result["x_meas"], result["y_meas"]
    x_kf, y_kf = result["kf_history"][:, 0], result["kf_history"][:, 1]
    false_alarms = result["false_alarms"]
    k = np.arange(cfg.N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(x_true, y_true, 'g-', linewidth=2, label='True trajectory')
    ax.scatter(x_meas, y_meas, s=15, c='r', alpha=0.4, label='Noisy measurements')
    if false_alarms:
        fa_x = [f[1] for f in false_alarms]
        fa_y = [f[2] for f in false_alarms]
        ax.scatter(fa_x, fa_y, s=8, c='orange', alpha=0.3, marker='x',
                   label='False alarms')
    ax.plot(x_kf, y_kf, 'b-', linewidth=1.8, label='Kalman estimate')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('Task 5/7/8: True vs. measured vs. filtered trajectory')
    ax.legend(loc='upper left', fontsize=9); ax.axis('equal'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(k, meas_err, 'r-', alpha=0.6, label=f'Measurement error (RMSE={rmse_meas:.1f} m)')
    ax.plot(k, kf_err, 'b-', linewidth=1.5, label=f'KF estimate error (RMSE={rmse_kf:.1f} m)')
    ax.set_xlabel('Scan number'); ax.set_ylabel('Position error (m)')
    ax.set_title('Task 8: Error vs. scan')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved {save_path}")
    plt.close()


if __name__ == "__main__":
    result = run_scenario()
    meas_err, kf_err, rmse_meas, rmse_kf = compute_rmse(result)

    print(f"Detected {result['detected'].sum()} / {cfg.N} scans")
    print(f"False alarms generated: {len(result['false_alarms'])}")
    print(f"RMSE (raw measurements vs truth): {rmse_meas:.2f} m")
    print(f"RMSE (Kalman estimate vs truth):  {rmse_kf:.2f} m")
    print(f"Improvement: {(1 - rmse_kf/rmse_meas)*100:.1f}%")

    plot_full_result(result, meas_err, kf_err, rmse_meas, rmse_kf,
                      save_path="outputs/04_single_run_full_pipeline.png")
