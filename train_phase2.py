"""
train_phase2.py  --  RESEARCH EXTENSION, Phase 2: train the adaptive-Q
filter and produce the first classical-vs-adaptive comparison.

Loss: mean squared position error against ground truth, summed over
the trajectory (after a short burn-in to avoid penalizing the initial
transient every filter has right after initialization -- classical
included). This is the simplest standard choice; a full negative-log-
likelihood loss (which also uses P, not just the point estimate) is a
natural upgrade to try later if time permits, and is worth mentioning
as such in the paper.
"""

import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

import config as cfg
from dataset import build_dataset, build_trajectory
from adaptive_kf import AdaptiveKF, run_adaptive_filter_numpy
from maneuver_truth_model import generate_maneuvering_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections
from kalman_filter import run_filter_with_covariance
from metrics import compute_nees, nees_consistency_bounds

BURN_IN = 5


def train(n_train=400, n_epochs=15, batch_size=16, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    print(f"Building {n_train} training trajectories...")
    train_data = build_dataset(n_train, base_seed=10_000)

    model = AdaptiveKF()
    opt = optim.Adam(model.parameters(), lr=lr)

    n_batches = int(np.ceil(n_train / batch_size))
    loss_history = []

    t0 = time.time()
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            opt.zero_grad()
            batch_loss = 0.0
            for i in idx:
                traj = train_data[i]
                x_hist, _, _ = model(traj)
                x_true = torch.tensor(traj["x_true"][:, :2], dtype=torch.float32)
                sq_err = ((x_hist[BURN_IN:, :2] - x_true[BURN_IN:]) ** 2).sum(dim=1)
                batch_loss = batch_loss + sq_err.mean()
            batch_loss = batch_loss / len(idx)
            batch_loss.backward()
            opt.step()
            epoch_loss += batch_loss.item() * len(idx)
        epoch_loss /= n_train
        loss_history.append(epoch_loss)
        rmse_equiv = np.sqrt(epoch_loss)
        print(f"epoch {epoch+1:2d}/{n_epochs}  mean-loss={epoch_loss:10.1f}  "
              f"(~{rmse_equiv:.1f} m RMSE on train)  [{time.time()-t0:.0f}s elapsed]")

    return model, loss_history


def evaluate_on_fixed_scenarios(model, n_runs=100):
    """Runs BOTH classical and adaptive filters on the same held-out
    fixed scenarios used in the Phase 1 checkpoint, for direct
    comparison."""
    results = {}
    for scenario in cfg.MANEUVER_SCENARIOS:
        classical_err = np.zeros((n_runs, cfg.N))
        adaptive_err = np.zeros((n_runs, cfg.N))
        classical_nees = np.zeros((n_runs, cfg.N))
        adaptive_nees = np.zeros((n_runs, cfg.N))
        maneuvering = None

        for run in range(n_runs):
            seed = cfg.BASE_SEED + run  # SAME eval seeds as Phase 1 checkpoint
            rng = np.random.default_rng(seed)
            x_true, y_true, vx_true, vy_true, man = generate_maneuvering_truth(scenario)
            x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
                generate_noisy_measurements(x_true, y_true, rng)
            x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)

            x_true_full = np.stack([x_true, y_true, vx_true, vy_true], axis=1)

            # classical
            x_est_c, P_hist_c = run_filter_with_covariance(
                x_target_meas, y_target_meas, detected, r_meas, theta_meas,
                measurement_covariance
            )
            classical_err[run] = np.sqrt((x_est_c[:, 0] - x_true)**2 + (x_est_c[:, 1] - y_true)**2)
            classical_nees[run] = compute_nees(x_true_full, x_est_c, P_hist_c, state_dims=[0, 1])

            # adaptive -- build the same-shaped traj dict the model expects
            R_seq = np.zeros((cfg.N, 2, 2))
            for k in range(cfg.N):
                if detected[k]:
                    R_seq[k] = measurement_covariance(r_meas[k], theta_meas[k])
            z_seq = np.stack([
                np.where(detected, x_target_meas, 0.0),
                np.where(detected, y_target_meas, 0.0),
            ], axis=1)
            traj = {"x_true": x_true_full, "z": z_seq, "R": R_seq, "detected": detected}
            x_est_a, q_hist, P_hist_a = run_adaptive_filter_numpy(model, traj)
            adaptive_err[run] = np.sqrt((x_est_a[:, 0] - x_true)**2 + (x_est_a[:, 1] - y_true)**2)
            adaptive_nees[run] = compute_nees(x_true_full, x_est_a, P_hist_a, state_dims=[0, 1])

            maneuvering = man

        results[scenario] = {
            "classical_err": classical_err,
            "adaptive_err": adaptive_err,
            "classical_nees": classical_nees,
            "adaptive_nees": adaptive_nees,
            "maneuvering": maneuvering,
        }
    return results


def print_comparison(results, n_runs):
    print(f"\n=== Phase 2: classical vs. adaptive-Q, {n_runs} MC runs/scenario ===")
    lower, upper = nees_consistency_bounds(n_runs, n_state=2)
    print(f"95% NEES consistency envelope (2 DOF): [{lower:.2f}, {upper:.2f}]")
    print(f"{'Scenario':<12} {'Classical RMSE':>15} {'Adaptive RMSE':>15} {'Improvement':>12} "
          f"{'Classical NEES':>15} {'Adaptive NEES':>15}")
    for scenario, r in results.items():
        rmse_c = np.sqrt(np.mean(r["classical_err"]**2))
        rmse_a = np.sqrt(np.mean(r["adaptive_err"]**2))
        improvement = (1 - rmse_a / rmse_c) * 100
        nees_c = r["classical_nees"].mean()
        nees_a = r["adaptive_nees"].mean()
        print(f"{scenario:<12} {rmse_c:>14.2f}m {rmse_a:>14.2f}m {improvement:>11.1f}% "
              f"{nees_c:>15.2f} {nees_a:>15.2f}")


def plot_comparison(results, n_runs, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    k = np.arange(cfg.N)
    for ax, (scenario, r) in zip(axes, results.items()):
        mean_c = r["classical_err"].mean(axis=0)
        mean_a = r["adaptive_err"].mean(axis=0)
        ax.plot(k, mean_c, 'r-', label='Classical (fixed Q)', linewidth=1.8)
        ax.plot(k, mean_a, 'b-', label='Adaptive-Q (GRU)', linewidth=1.8)
        man = r["maneuvering"]
        if man.any():
            ax.fill_between(k, 0, max(mean_c.max(), mean_a.max()) * 1.1,
                             where=man, color='gray', alpha=0.15)
        ax.set_title(scenario)
        ax.set_xlabel("Scan number")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean position error (m)")
    fig.suptitle(f"Phase 2: classical vs. learned adaptive-Q filter ({n_runs} MC runs/scenario)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved {save_path}")
    plt.close()


if __name__ == "__main__":
    # NOTE: this direct-execution path is the ORIGINAL, pre-standardized-
    # protocol training entry point, kept for historical/comparison purposes
    # only (see Table VI's protocol-robustness comparison in the paper).
    # It is NOT part of the automated reproduce_all.py pipeline and does NOT
    # produce the paper's headline checkpoint -- that's train_standardized.py,
    # invoked directly by reproduce_all.py. Running this file directly will
    # train a fresh ad-hoc-protocol model and save it as adaptive_kf_weights.pt,
    # useful only if you want to independently reproduce Table VI's comparison.
    model, loss_hist = train(n_train=400, n_epochs=15, batch_size=16)
    torch.save(model.state_dict(), "outputs/adaptive_kf_weights.pt")

    N_EVAL_RUNS = 100
    results = evaluate_on_fixed_scenarios(model, n_runs=N_EVAL_RUNS)
    print_comparison(results, N_EVAL_RUNS)
    plot_comparison(results, N_EVAL_RUNS, save_path="outputs/09_phase2_classical_vs_adaptive.png")
