"""
monte_carlo_evaluation.py  --  Task 6 (Monte Carlo statistics) + a
robust, averaged version of the Task 8 RMSE comparison.

A single 100-scan run is one noisy realization of a random process --
its sample statistics won't exactly match the true parameters you set
(sigma_r=50, sigma_theta=0.5deg, PD=0.92, etc). Running MC_RUNS
independent trials and averaging shows the estimator statistics
converging to the true values, and gives a much more trustworthy
picture of how well the Kalman filter performs on average (not just
in one lucky/unlucky run).
"""

import numpy as np
import matplotlib.pyplot as plt
import config as cfg

from run_single_scenario import run_scenario, compute_rmse


def run_monte_carlo(n_runs=cfg.MC_RUNS):
    all_meas_err = np.zeros((n_runs, cfg.N))
    all_kf_err = np.zeros((n_runs, cfg.N))
    rmse_meas_list = np.zeros(n_runs)
    rmse_kf_list = np.zeros(n_runs)

    range_err_all = []
    az_err_all = []

    for run in range(n_runs):
        # distinct seed per run so trials are independent, but the whole
        # sweep is reproducible run-to-run
        seed = cfg.BASE_SEED + run
        result = run_scenario(seed=seed)
        meas_err, kf_err, rmse_meas, rmse_kf = compute_rmse(result)

        all_meas_err[run] = meas_err
        all_kf_err[run] = kf_err
        rmse_meas_list[run] = rmse_meas
        rmse_kf_list[run] = rmse_kf

        range_err_all.append(result["r_meas"] - result["r_true"])
        az_err_all.append(result["theta_meas"] - result["theta_true"])

    range_err_all = np.concatenate(range_err_all)
    az_err_all = np.concatenate(az_err_all)

    return {
        "all_meas_err": all_meas_err,
        "all_kf_err": all_kf_err,
        "rmse_meas_list": rmse_meas_list,
        "rmse_kf_list": rmse_kf_list,
        "range_err_all": range_err_all,
        "az_err_all": az_err_all,
    }


def print_mc_summary(mc):
    print(f"=== Monte Carlo Summary ({cfg.MC_RUNS} runs) ===")
    print(f"Range-noise std dev over all runs:   {mc['range_err_all'].std():.2f} m "
          f"(target {cfg.SIGMA_R} m)")
    print(f"Azimuth-noise std dev over all runs: "
          f"{np.rad2deg(mc['az_err_all'].std()):.4f} deg (target {cfg.SIGMA_THETA_DEG} deg)")
    print()
    print(f"Mean RMSE (measurements vs truth): "
          f"{mc['rmse_meas_list'].mean():.2f} m  (std across runs: {mc['rmse_meas_list'].std():.2f})")
    print(f"Mean RMSE (Kalman estimate vs truth): "
          f"{mc['rmse_kf_list'].mean():.2f} m  (std across runs: {mc['rmse_kf_list'].std():.2f})")
    improvement = (1 - mc['rmse_kf_list'].mean() / mc['rmse_meas_list'].mean()) * 100
    print(f"Average improvement from filtering: {improvement:.1f}%")


def plot_mc_results(mc, save_path=None):
    k = np.arange(cfg.N)
    mean_meas_err = mc["all_meas_err"].mean(axis=0)
    mean_kf_err = mc["all_kf_err"].mean(axis=0)
    std_kf_err = mc["all_kf_err"].std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(k, mean_meas_err, 'r-', alpha=0.7,
            label=f'Mean measurement error ({mc["rmse_meas_list"].mean():.1f} m RMSE)')
    ax.plot(k, mean_kf_err, 'b-', linewidth=1.8,
            label=f'Mean KF error ({mc["rmse_kf_list"].mean():.1f} m RMSE)')
    ax.fill_between(k, mean_kf_err - std_kf_err, mean_kf_err + std_kf_err,
                     color='b', alpha=0.15, label='KF error ±1 std across runs')
    ax.set_xlabel('Scan number'); ax.set_ylabel('Position error (m)')
    ax.set_title(f'Task 6/8: Averaged error over {cfg.MC_RUNS} Monte Carlo runs')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(mc["rmse_meas_list"], bins=25, alpha=0.6, color='r', label='Measurement RMSE')
    ax.hist(mc["rmse_kf_list"], bins=25, alpha=0.6, color='b', label='KF estimate RMSE')
    ax.set_xlabel('Per-run RMSE (m)'); ax.set_ylabel('Count')
    ax.set_title('Distribution of per-run RMSE across Monte Carlo trials')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved {save_path}")
    plt.close()


if __name__ == "__main__":
    mc = run_monte_carlo()
    print_mc_summary(mc)
    plot_mc_results(mc, save_path="outputs/05_monte_carlo_results.png")
