"""
measurement_model.py  --  Task 2: Gaussian measurement noise (+ support for Task 6/7).

A radar measures RANGE and AZIMUTH, not (x, y). So noise is added in the
polar domain and then converted to Cartesian for use downstream.

Also provides `measurement_covariance()`, the Jacobian-linearized
Cartesian covariance for a given noisy (range, azimuth) pair -- this is
what feeds the Kalman filter's R matrix in kalman_filter.py, since a
fixed diagonal R would be physically wrong (Cartesian noise from a polar
sensor is NOT isotropic -- it stretches along the azimuth direction more
at long range).
"""

import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from truth_model import generate_truth


def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def generate_noisy_measurements(x_true, y_true, rng):
    """Add Gaussian noise in polar domain, return noisy (x, y) plus the
    underlying noisy (r, theta) and true (r, theta) for diagnostics."""
    r_true, theta_true = cartesian_to_polar(x_true, y_true)

    r_meas = r_true + rng.normal(0, cfg.SIGMA_R, cfg.N)
    theta_meas = theta_true + rng.normal(0, cfg.SIGMA_THETA, cfg.N)

    x_meas, y_meas = polar_to_cartesian(r_meas, theta_meas)
    return x_meas, y_meas, r_meas, theta_meas, r_true, theta_true


def measurement_covariance(r, theta):
    """Converted-measurement covariance R via first-order (Jacobian)
    linearization of the polar -> Cartesian transform, evaluated at the
    (noisy) measured range/azimuth."""
    J = np.array([[np.cos(theta), -r * np.sin(theta)],
                  [np.sin(theta),  r * np.cos(theta)]])
    polar_cov = np.diag([cfg.SIGMA_R**2, cfg.SIGMA_THETA**2])
    return J @ polar_cov @ J.T


def plot_measurements(x_true, y_true, x_meas, y_meas, save_path=None):
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 5.5))

    for ax in (ax_full, ax_zoom):
        ax.plot(x_true, y_true, 'g-', linewidth=2, label='True trajectory')
        ax.scatter(x_meas, y_meas, s=15, c='r', alpha=0.6, label='Noisy measurements')
        ax.grid(True, alpha=0.3)

    ax_full.scatter([0], [0], marker='^', c='k', s=100, label='Radar')
    ax_full.set_xlabel('x (m)'); ax_full.set_ylabel('y (m)')
    ax_full.set_title('Full geometry (radar to target)')
    ax_full.legend(loc='lower right', fontsize=9)
    ax_full.set_aspect('equal')

    margin = 0.08 * (x_true.max() - x_true.min())
    ax_zoom.set_xlim(x_true.min() - margin, x_true.max() + margin)
    ax_zoom.set_ylim(min(y_true.min(), y_meas.min()) - margin,
                      max(y_true.max(), y_meas.max()) + margin)
    ax_zoom.set_xlabel('x (m)'); ax_zoom.set_ylabel('y (m)')
    ax_zoom.set_title('Zoomed: measurement detail')

    fig.suptitle('Task 2: True trajectory vs. noisy radar measurements', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved {save_path}")
    plt.close()


if __name__ == "__main__":
    rng = np.random.default_rng(cfg.BASE_SEED)
    x_true, y_true, _, _ = generate_truth()
    x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
        generate_noisy_measurements(x_true, y_true, rng)

    plot_measurements(x_true, y_true, x_meas, y_meas,
                       save_path="outputs/02_noisy_measurements.png")

    range_err = r_meas - r_true
    az_err = theta_meas - theta_true
    print(f"Sample range-error std dev:   {range_err.std():.2f} m  (target {cfg.SIGMA_R})")
    print(f"Sample azimuth-error std dev: {np.rad2deg(az_err.std()):.4f} deg "
          f"(target {cfg.SIGMA_THETA_DEG})")
