"""
train_seed_variance.py  --  RESEARCH EXTENSION: characterizes whether
the headline result depends on which random weight initialization was
used, by training 3 independent replicates (model_seed 0, 1, 2) that
differ ONLY in weight init and minibatch shuffling order, everything
else (training data, validation data, architecture, optimizer,
scheduler) held identical.

Uses a smaller, disclosed 400-trajectory/12-epoch budget rather than
the full 1400-trajectory/25-epoch headline budget, since repeating the
full protocol three times is not tractable in reasonable time.
Absolute numbers from this script are therefore NOT directly
comparable to the paper's Table I; only the relative spread (std,
coefficient of variation) across seeds at this fixed, matched budget
is the quantity of interest.

Usage:
    python train_seed_variance.py            # trains all 3 seeds + evaluates
    python train_seed_variance.py --seed 1   # trains/evaluates one seed only
"""

import argparse
import pickle

import numpy as np
import torch

from train_standardized import train_standardized
from adaptive_kf import AdaptiveKF
from train_phase2 import evaluate_on_fixed_scenarios


def train_and_evaluate_seed(model_seed, n_train=400, n_epochs=12, n_eval_runs=300, sfx=""):
    print(f"\n{'='*70}\nTraining model_seed={model_seed}\n{'='*70}")
    model, history, _ = train_standardized(
        n_train=n_train, n_val=200, n_epochs=n_epochs, hidden_size=16, loss="mse",
        checkpoint=f"outputs/seedvar_seed{model_seed}{sfx}.pt",
        log=f"outputs/seedvar_seed{model_seed}{sfx}_log.jsonl",
        scheduler_patience=3, model_seed=model_seed,
    )
    print(f"Evaluating model_seed={model_seed} on fixed scenarios ({n_eval_runs} runs)...")
    results = evaluate_on_fixed_scenarios(model, n_runs=n_eval_runs)
    for sc in ["cv_only", "mild_turn", "sharp_turn"]:
        rmse = np.sqrt(np.mean(results[sc]["adaptive_err"] ** 2))
        nees = results[sc]["adaptive_nees"].mean()
        print(f"  {sc:<12} RMSE={rmse:7.2f}m  NEES={nees:.3f}")
    return results


def summarize(all_results):
    print(f"\n{'='*70}\nTraining-seed variance summary "
          f"({len(all_results)} seeds, 300 MC runs each)\n{'='*70}")
    for sc in ["cv_only", "mild_turn", "sharp_turn"]:
        rmses = [np.sqrt(np.mean(r[sc]["adaptive_err"] ** 2)) for r in all_results.values()]
        neess = [r[sc]["adaptive_nees"].mean() for r in all_results.values()]
        rmse_mean, rmse_std = np.mean(rmses), np.std(rmses, ddof=1)
        nees_mean, nees_std = np.mean(neess), np.std(neess, ddof=1)
        print(f"{sc:<12} RMSE: {rmse_mean:.2f} +/- {rmse_std:.2f} m "
              f"(CV={rmse_std/rmse_mean*100:.2f}%)   "
              f"NEES: {nees_mean:.3f} +/- {nees_std:.3f} "
              f"(CV={nees_std/nees_mean*100:.2f}%)")


def main(seeds=(0, 1, 2), n_train=400, n_epochs=12, n_eval_runs=300,
         results_path="outputs/seed_variance_results.pkl", sfx=""):
    all_results = {}
    for seed in seeds:
        all_results[seed] = train_and_evaluate_seed(
            seed, n_train=n_train, n_epochs=n_epochs, n_eval_runs=n_eval_runs, sfx=sfx)
    with open(results_path, "wb") as f:
        pickle.dump(all_results, f)
    if len(all_results) > 1:
        summarize(all_results)
    print(f"\nSaved {results_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None,
                    help="Train/evaluate a single seed only, instead of all 3.")
    args = p.parse_args()
    main(seeds=(args.seed,) if args.seed is not None else (0, 1, 2))
