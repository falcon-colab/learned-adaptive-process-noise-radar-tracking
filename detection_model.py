"""
detection_model.py  --  Task 3 (false alarms) and Task 4 (missed detections).

Missed detections: Bernoulli(PD) per scan. If not detected, the scan
contributes no true-target measurement.

False alarms: Poisson-distributed COUNT per scan (equivalently,
exponential inter-arrival times between clutter events), scattered
uniformly across a surveillance region around the trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from truth_model import generate_truth
from measurement_model import generate_noisy_measurements


def apply_missed_detections(x_meas, y_meas, rng, pd=None):
    """Returns (x_target_meas, y_target_meas, detected_mask). Missed
    scans are set to NaN in the returned arrays.
    pd: optional override for the detection probability (defaults to
    cfg.PD). Used by the severity sweep to test the already-trained
    filter at Pd values it never saw during training.
    """
    pd = cfg.PD if pd is None else pd
    detected = rng.random(cfg.N) < pd
    x_target_meas = np.where(detected, x_meas, np.nan)
    y_target_meas = np.where(detected, y_meas, np.nan)
    return x_target_meas, y_target_meas, detected


def generate_false_alarms(x_true, y_true, rng, lam=None):
    """Returns a list of (scan_index, x, y) false-alarm detections.
    lam: optional override for the clutter rate (defaults to cfg.LAMBDA_FA).
    """
    lam = cfg.LAMBDA_FA if lam is None else lam
    n_fa_per_scan = rng.poisson(lam=lam, size=cfg.N)

    xmin, xmax = x_true.min() - cfg.FA_BOX_MARGIN, x_true.max() + cfg.FA_BOX_MARGIN
    ymin, ymax = y_true.min() - cfg.FA_BOX_MARGIN, y_true.max() + cfg.FA_BOX_MARGIN

    false_alarms = []
    for i in range(cfg.N):
        for _ in range(n_fa_per_scan[i]):
            fx = rng.uniform(xmin, xmax)
            fy = rng.uniform(ymin, ymax)
            false_alarms.append((i, fx, fy))
    return false_alarms


def plot_detections(x_true, y_true, x_target_meas, y_target_meas,
                     false_alarms, save_path=None):
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5.5))
    fa_x = [f[1] for f in false_alarms] if false_alarms else []
    fa_y = [f[2] for f in false_alarms] if false_alarms else []

    for ax in (ax_full, ax_zoom):
        ax.plot(x_true, y_true, 'g-', linewidth=2, label='True trajectory', zorder=3)
        ax.scatter(x_target_meas, y_target_meas, s=20, c='b', alpha=0.7,
                   label='True-target detections', zorder=4)
        if false_alarms:
            ax.scatter(fa_x, fa_y, s=10, c='orange', alpha=0.4, marker='x',
                       label='False alarms (clutter)', zorder=2)
        ax.grid(True, alpha=0.3)

    ax_full.scatter([0], [0], marker='^', c='k', s=100, label='Radar', zorder=5)
    ax_full.set_xlabel('x (m)'); ax_full.set_ylabel('y (m)')
    ax_full.set_title('Full geometry (radar to target)')
    ax_full.legend(loc='lower right', fontsize=8)
    ax_full.set_aspect('equal')

    margin = 0.08 * (x_true.max() - x_true.min())
    all_x = np.concatenate([[x_true.min(), x_true.max()], fa_x]) if fa_x else x_true
    all_y = np.concatenate([[y_true.min(), y_true.max()], fa_y]) if fa_y else y_true
    ax_zoom.set_xlim(x_true.min() - margin, x_true.max() + margin)
    ax_zoom.set_ylim(y_true.min() - margin, y_true.max() + margin)
    ax_zoom.set_xlabel('x (m)'); ax_zoom.set_ylabel('y (m)')
    ax_zoom.set_title('Zoomed: detections near trajectory')

    fig.suptitle('Tasks 3 & 4: Target detections (with misses) + false alarm clutter', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved {save_path}")
    plt.close()


if __name__ == "__main__":
    rng = np.random.default_rng(cfg.BASE_SEED)
    x_true, y_true, _, _ = generate_truth()
    x_meas, y_meas, *_ = generate_noisy_measurements(x_true, y_true, rng)

    x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)
    false_alarms = generate_false_alarms(x_true, y_true, rng)

    plot_detections(x_true, y_true, x_target_meas, y_target_meas, false_alarms,
                     save_path="outputs/03_detections_and_clutter.png")

    print(f"Detected {detected.sum()} / {cfg.N} scans "
          f"({detected.sum()/cfg.N*100:.1f}% -- target {cfg.PD*100:.0f}%)")
    print(f"Generated {len(false_alarms)} false alarms "
          f"(mean {len(false_alarms)/cfg.N:.2f}/scan -- target {cfg.LAMBDA_FA})")
