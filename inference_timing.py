"""
Wall-clock inference time comparison: adaptive filter (single GRU-based
filter) vs. IMM (two parallel classical filters + explicit mixing step),
addressing the paper's qualitative claim about relative complexity with
an actual measurement.

Times a single full 100-scan trajectory pass per filter (matching this
project's cfg.N=100), averaged over many repetitions for a stable
estimate, run on CPU (matching the paper's own claim that this problem
does not benefit from GPU given the small, sequential per-timestep
recursion).
"""
import time
import numpy as np
import torch

import config as cfg
from maneuver_truth_model import generate_maneuvering_truth
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections
from kalman_filter import run_filter_with_covariance
from imm_filter import run_imm
from adaptive_kf import AdaptiveKF, run_adaptive_filter_numpy

torch.set_num_threads(1)  # fair single-thread comparison, no BLAS parallelism advantage

# Build one fixed, representative trajectory (mild_turn) to time repeatedly
rng = np.random.default_rng(cfg.BASE_SEED)
x_true, y_true, vx_true, vy_true, man = generate_maneuvering_truth("mild_turn")
x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = generate_noisy_measurements(x_true, y_true, rng)
x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)

model = AdaptiveKF(hidden_size=16)
model.load_state_dict(torch.load("outputs/adaptive_kf_standardized_1400.pt"))
model.eval()

R_seq = np.zeros((cfg.N, 2, 2))
for k in range(cfg.N):
    if detected[k]:
        R_seq[k] = measurement_covariance(r_meas[k], theta_meas[k])
z_seq = np.stack([np.where(detected, x_target_meas, 0.0),
                   np.where(detected, y_target_meas, 0.0)], axis=1)
x_true_full = np.stack([x_true, y_true, vx_true, vy_true], axis=1)
traj = {"x_true": x_true_full, "z": z_seq, "R": R_seq, "detected": detected}

N_REPS = 200
N_WARMUP = 20

def time_it(fn, n_reps, n_warmup):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000  # ms

# Classical (naive) filter timing
def run_classical():
    run_filter_with_covariance(x_target_meas, y_target_meas, detected, r_meas, theta_meas, measurement_covariance)

# IMM timing
def run_imm_once():
    run_imm(x_target_meas, y_target_meas, detected, r_meas, theta_meas, measurement_covariance, q_high=500)

# Adaptive filter timing
def run_adaptive_once():
    with torch.no_grad():
        run_adaptive_filter_numpy(model, traj)

t_classical = time_it(run_classical, N_REPS, N_WARMUP)
t_imm = time_it(run_imm_once, N_REPS, N_WARMUP)
t_adaptive = time_it(run_adaptive_once, N_REPS, N_WARMUP)

print(f"Classical (naive):  {t_classical.mean():.3f} +/- {t_classical.std():.3f} ms/trajectory (N={N_REPS})")
print(f"IMM:                {t_imm.mean():.3f} +/- {t_imm.std():.3f} ms/trajectory (N={N_REPS})")
print(f"Adaptive (GRU):     {t_adaptive.mean():.3f} +/- {t_adaptive.std():.3f} ms/trajectory (N={N_REPS})")
print()
print(f"IMM / Classical ratio: {t_imm.mean()/t_classical.mean():.2f}x")
print(f"Adaptive / Classical ratio: {t_adaptive.mean()/t_classical.mean():.2f}x")
print(f"Adaptive / IMM ratio: {t_adaptive.mean()/t_imm.mean():.2f}x")

import pickle
with open("outputs/inference_timing_results.pkl", "wb") as f:
    pickle.dump({"classical_ms": t_classical, "imm_ms": t_imm, "adaptive_ms": t_adaptive}, f)
