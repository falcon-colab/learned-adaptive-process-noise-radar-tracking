"""
main.py  --  Runs the full EE-891 project pipeline end-to-end, stage by
stage, saving every figure to outputs/ and printing a full text summary
(useful to paste straight into your technical report's Results section).

For LIVE DEMONSTRATION, you can also run each stage module individually
to show incremental build-up:
    python truth_model.py
    python measurement_model.py
    python detection_model.py
    python statistics_analysis.py
    python run_single_scenario.py
    python monte_carlo_evaluation.py
"""

import os
import numpy as np

import config as cfg
from truth_model import generate_truth, plot_truth
from measurement_model import generate_noisy_measurements, plot_measurements
from detection_model import apply_missed_detections, generate_false_alarms, plot_detections
from statistics_analysis import analyze_noise, print_report
from run_single_scenario import run_scenario, compute_rmse, plot_full_result
from monte_carlo_evaluation import run_monte_carlo, print_mc_summary, plot_mc_results


def main():
    os.makedirs("outputs", exist_ok=True)
    print("#" * 70)
    print("EE-891 Stochastic Systems -- Radar + Kalman Filter Project")
    print("#" * 70)

    # ---------------- Task 1 ----------------
    print("\n--- Task 1: True trajectory ---")
    x_true, y_true, vx_true, vy_true = generate_truth()
    plot_truth(x_true, y_true, save_path="outputs/01_truth_trajectory.png")

    # ---------------- Task 2 ----------------
    print("\n--- Task 2: Noisy measurements ---")
    rng = np.random.default_rng(cfg.BASE_SEED)
    x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
        generate_noisy_measurements(x_true, y_true, rng)
    plot_measurements(x_true, y_true, x_meas, y_meas,
                       save_path="outputs/02_noisy_measurements.png")

    # ---------------- Tasks 3 & 4 ----------------
    print("\n--- Tasks 3 & 4: False alarms + missed detections ---")
    x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)
    false_alarms = generate_false_alarms(x_true, y_true, rng)
    plot_detections(x_true, y_true, x_target_meas, y_target_meas, false_alarms,
                     save_path="outputs/03_detections_and_clutter.png")
    print(f"Detected {detected.sum()} / {cfg.N} scans "
          f"({detected.sum()/cfg.N*100:.1f}% -- target {cfg.PD*100:.0f}%)")
    print(f"False alarms: {len(false_alarms)} "
          f"(mean {len(false_alarms)/cfg.N:.2f}/scan -- target {cfg.LAMBDA_FA})")

    # ---------------- Task 6 (single-run noise stats) ----------------
    print("\n--- Task 6: Single-run noise statistics ---")
    stats_dict = analyze_noise(r_meas, theta_meas, r_true, theta_true)
    print_report(stats_dict)

    # ---------------- Tasks 5, 7, 8 (single demo run) ----------------
    print("\n--- Tasks 5/7/8: Single-run Kalman filter + RMSE ---")
    result = run_scenario()
    meas_err, kf_err, rmse_meas, rmse_kf = compute_rmse(result)
    print(f"RMSE (raw measurements vs truth): {rmse_meas:.2f} m")
    print(f"RMSE (Kalman estimate vs truth):  {rmse_kf:.2f} m")
    print(f"Improvement: {(1 - rmse_kf/rmse_meas)*100:.1f}%")
    plot_full_result(result, meas_err, kf_err, rmse_meas, rmse_kf,
                      save_path="outputs/04_single_run_full_pipeline.png")

    # ---------------- Task 6 (Monte Carlo) + robust Task 8 ----------------
    print(f"\n--- Task 6/8: Monte Carlo evaluation ({cfg.MC_RUNS} runs) ---")
    mc = run_monte_carlo()
    print_mc_summary(mc)
    plot_mc_results(mc, save_path="outputs/05_monte_carlo_results.png")

    print("\nAll figures saved to outputs/. Pipeline complete.")


if __name__ == "__main__":
    main()
