"""
dataset.py  --  RESEARCH EXTENSION, Phase 2: training data generation.

Every trajectory here is produced by the SAME simulator used for the
original EE-891 deliverable and the Phase 1 checkpoint --
measurement_model.py and detection_model.py are reused completely
unmodified. The only new piece is the randomized maneuver profile
(maneuver_truth_model.generate_random_maneuvering_truth), which is
deliberately never one of the fixed cfg.MANEUVER_SCENARIOS presets used
for evaluation -- a genuine train/test split at the scenario level.
"""

import numpy as np
import config as cfg

from maneuver_truth_model import generate_random_maneuvering_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections


def build_trajectory(seed):
    """One training example: everything the differentiable filter needs,
    as plain numpy arrays."""
    rng = np.random.default_rng(seed)
    x_true, y_true, vx_true, vy_true, maneuvering = generate_random_maneuvering_truth(rng)

    x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
        generate_noisy_measurements(x_true, y_true, rng)
    x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)

    # Precompute R_k for every scan (2x2), zero-filled where not detected
    R_seq = np.zeros((cfg.N, 2, 2))
    for k in range(cfg.N):
        if detected[k]:
            R_seq[k] = measurement_covariance(r_meas[k], theta_meas[k])

    z_seq = np.stack([
        np.where(detected, x_target_meas, 0.0),
        np.where(detected, y_target_meas, 0.0),
    ], axis=1)  # (N, 2), zero where missed -- network also sees `detected`

    x_true_full = np.stack([x_true, y_true, vx_true, vy_true], axis=1)

    return {
        "x_true": x_true_full,       # (N, 4)
        "z": z_seq,                  # (N, 2)
        "R": R_seq,                  # (N, 2, 2)
        "detected": detected,        # (N,) bool
        "maneuvering": maneuvering,  # (N,) bool, for diagnostics only
    }


def build_dataset(n_trajectories, base_seed=10_000):
    """Builds n_trajectories independent training examples. Uses a seed
    range disjoint from cfg.BASE_SEED-derived eval seeds (which start at
    cfg.BASE_SEED=42 and go up to ~42+300) to avoid any accidental
    overlap between training and evaluation randomness."""
    return [build_trajectory(base_seed + i) for i in range(n_trajectories)]


if __name__ == "__main__":
    data = build_dataset(5)
    for i, d in enumerate(data):
        print(f"traj {i}: {d['detected'].sum()}/{cfg.N} detected, "
              f"{d['maneuvering'].sum()} maneuvering scans")
