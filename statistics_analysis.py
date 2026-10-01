"""
statistics_analysis.py  --  Task 6 (single-run part): mean, variance,
covariance, and confidence intervals of the measurement noise.

The Monte Carlo part of Task 6 (statistics ACROSS many independent runs)
lives in monte_carlo_evaluation.py -- this module characterizes the
noise WITHIN one run, which you need first to sanity-check the model
before scaling up to many runs.
"""

import numpy as np
from scipy import stats
import config as cfg


def analyze_noise(r_meas, theta_meas, r_true, theta_true):
    """Returns a dict of noise statistics for the range and azimuth
    measurement errors of a single run."""
    range_err = r_meas - r_true
    az_err = theta_meas - theta_true  # radians

    n = len(range_err)

    def ci_95(sample):
        mean = sample.mean()
        sem = stats.sem(sample)  # standard error of the mean
        ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
        return mean, ci

    range_mean, range_ci = ci_95(range_err)
    az_mean, az_ci = ci_95(az_err)

    cov_matrix = np.cov(range_err, az_err)  # 2x2: [range, azimuth]

    return {
        "range_mean": range_mean,
        "range_var": range_err.var(ddof=1),
        "range_std": range_err.std(ddof=1),
        "range_95ci": range_ci,
        "azimuth_mean_deg": np.rad2deg(az_mean),
        "azimuth_var_deg2": np.rad2deg(az_err).var(ddof=1),
        "azimuth_std_deg": np.rad2deg(az_err).std(ddof=1),
        "azimuth_95ci_deg": tuple(np.rad2deg(az_ci)),
        "range_azimuth_covariance": cov_matrix[0, 1],
        "cov_matrix": cov_matrix,
    }


def print_report(stats_dict):
    print("=== Measurement Noise Statistics (single run, N={} scans) ===".format(cfg.N))
    print(f"Range error:   mean={stats_dict['range_mean']:.2f} m, "
          f"var={stats_dict['range_var']:.1f} m^2, "
          f"std={stats_dict['range_std']:.2f} m "
          f"(target std={cfg.SIGMA_R} m)")
    print(f"               95% CI on mean: "
          f"[{stats_dict['range_95ci'][0]:.2f}, {stats_dict['range_95ci'][1]:.2f}] m")
    print(f"Azimuth error: mean={stats_dict['azimuth_mean_deg']:.4f} deg, "
          f"var={stats_dict['azimuth_var_deg2']:.5f} deg^2, "
          f"std={stats_dict['azimuth_std_deg']:.4f} deg "
          f"(target std={cfg.SIGMA_THETA_DEG} deg)")
    print(f"               95% CI on mean: "
          f"[{stats_dict['azimuth_95ci_deg'][0]:.4f}, "
          f"{stats_dict['azimuth_95ci_deg'][1]:.4f}] deg")
    print(f"Range-Azimuth covariance: {stats_dict['range_azimuth_covariance']:.6f} "
          f"(expected ~0, since noise sources are generated independently)")


if __name__ == "__main__":
    from truth_model import generate_truth
    from measurement_model import generate_noisy_measurements

    rng = np.random.default_rng(cfg.BASE_SEED)
    x_true, y_true, _, _ = generate_truth()
    _, _, r_meas, theta_meas, r_true, theta_true = \
        generate_noisy_measurements(x_true, y_true, rng)

    results = analyze_noise(r_meas, theta_meas, r_true, theta_true)
    print_report(results)
