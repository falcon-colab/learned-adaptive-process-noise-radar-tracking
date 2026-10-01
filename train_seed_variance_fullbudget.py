"""
train_seed_variance_fullbudget.py  --  Full-budget (1400-trajectory,
25-epoch) training-seed robustness study, closing the gap the paper's
Limitations previously disclosed: the existing seed study
(train_seed_variance.py) only used a reduced 400-trajectory/12-epoch
budget, explicitly not directly comparable to the headline protocol.

model_seed=0 at this exact budget IS the headline checkpoint
(outputs/adaptive_kf_standardized_1400.pt) -- verified against
reproduce_all.py's actual training call, which uses model_seed=0 (the
train_standardized default) and no explicit scheduler_patience (so
also the default, 10). This script trains model_seed=1 and 2 at the
IDENTICAL protocol (n_train=1400, n_val=200, n_epochs=25,
scheduler_patience=10) and reuses the existing headline checkpoint for
seed 0, rather than wastefully retraining it.

Usage:
    python train_seed_variance_fullbudget.py
"""
import pickle

import numpy as np
import torch

from train_standardized import train_standardized
from adaptive_kf import AdaptiveKF
from train_phase2 import evaluate_on_fixed_scenarios

HEADLINE_CKPT = "outputs/adaptive_kf_standardized_1400.pt"
N_EVAL_RUNS = 300


def evaluate_checkpoint(ckpt_path, label):
    model = AdaptiveKF(hidden_size=16)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    results = evaluate_on_fixed_scenarios(model, n_runs=N_EVAL_RUNS)
    print(f"\n--- {label} ({ckpt_path}) ---")
    for sc in ["cv_only", "mild_turn", "sharp_turn"]:
        rmse = np.sqrt(np.mean(results[sc]["adaptive_err"] ** 2))
        nees = results[sc]["adaptive_nees"].mean()
        print(f"  {sc:<12} RMSE={rmse:7.2f}m  NEES={nees:.3f}")
    return results


def main():
    all_results = {}

    # Seed 0 = the existing headline checkpoint, reused, not retrained.
    all_results[0] = evaluate_checkpoint(HEADLINE_CKPT, "model_seed=0 (headline, reused)")

    # Seeds 1, 2: full-budget training runs, identical protocol.
    for seed in (1, 2):
        print(f"\n{'='*70}\nTraining model_seed={seed} at FULL budget (1400 traj, 25 epochs)\n{'='*70}")
        ckpt = f"outputs/seedvar_fullbudget_seed{seed}.pt"
        model, history, _ = train_standardized(
            n_train=1400, n_val=200, n_epochs=25, hidden_size=16, loss="mse",
            checkpoint=ckpt, log=f"outputs/seedvar_fullbudget_seed{seed}_log.jsonl",
            model_seed=seed,  # scheduler_patience defaults to 10, matching the headline
        )
        all_results[seed] = evaluate_checkpoint(ckpt, f"model_seed={seed} (new, full budget)")

    # Summary: RMSE/NEES mean +/- std and CV across the 3 full-budget seeds
    print(f"\n{'='*70}\nFull-budget (1400 traj/25 epoch) seed variance summary, N=3\n{'='*70}")
    summary = {}
    for sc in ["cv_only", "mild_turn", "sharp_turn"]:
        rmses = np.array([np.sqrt(np.mean(all_results[s][sc]["adaptive_err"] ** 2)) for s in (0, 1, 2)])
        neess = np.array([all_results[s][sc]["adaptive_nees"].mean() for s in (0, 1, 2)])
        rmse_mean, rmse_std = rmses.mean(), rmses.std(ddof=1)
        nees_mean, nees_std = neess.mean(), neess.std(ddof=1)
        print(f"{sc:<12} RMSE: {rmse_mean:.1f} +/- {rmse_std:.2f} m (CV={rmse_std/rmse_mean*100:.2f}%)   "
              f"NEES: {nees_mean:.3f} +/- {nees_std:.3f} (CV={nees_std/nees_mean*100:.2f}%)")
        summary[sc] = {"rmse_mean": rmse_mean, "rmse_std": rmse_std, "rmse_cv": rmse_std/rmse_mean*100,
                        "nees_mean": nees_mean, "nees_std": nees_std, "nees_cv": nees_std/nees_mean*100,
                        "rmses": rmses, "neess": neess}

    with open("outputs/seed_variance_fullbudget_results.pkl", "wb") as f:
        pickle.dump(summary, f)
    print("\nSaved outputs/seed_variance_fullbudget_results.pkl")


if __name__ == "__main__":
    main()
