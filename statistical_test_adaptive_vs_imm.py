"""
statistical_test_adaptive_vs_imm.py  --  Formal paired statistical test
for the paper's CENTRAL comparison (Table 1: adaptive filter vs. IMM),
replacing the earlier informal "matches or slightly exceeds" framing
with the same rigor already applied elsewhere in this project to the
MSE-vs-NLL comparison (see statistical_test_mse_vs_nll.py).

Both the adaptive filter (from evaluate_on_fixed_scenarios, via
train_phase2.py) and IMM (from outputs/imm_results_300.pkl) were
evaluated on the IDENTICAL 300 Monte Carlo noise realizations per
scenario (same cfg.BASE_SEED + run seed formula throughout this
project), so a PAIRED test is the correct and more statistically
powerful choice over an unpaired one.

Reports both statistical significance (paired t-test, Wilcoxon
signed-rank) and practical effect size (Cohen's d for paired
differences), for the same reason as the MSE-vs-NLL test: with n=300
pairs, even small, practically unimportant differences can reach
formal significance, and the two questions (statistically detectable
vs. practically meaningful) are not the same one.

Usage:
    python statistical_test_adaptive_vs_imm.py
"""
import pickle

import numpy as np
import torch
from scipy import stats

import config as cfg
from adaptive_kf import AdaptiveKF
from train_phase2 import evaluate_on_fixed_scenarios


def main(checkpoint="outputs/adaptive_kf_standardized_1400.pt", n_runs=300):
    model = AdaptiveKF(hidden_size=16)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    r_adaptive = evaluate_on_fixed_scenarios(model, n_runs=n_runs)
    with open("outputs/imm_results_300.pkl", "rb") as f:
        r_imm = pickle.load(f)

    print("=== Paired statistical test: Adaptive vs. IMM (Table 1's central comparison) ===\n")
    results = {}
    for sc in ["cv_only", "mild_turn", "sharp_turn"]:
        rmse_a = np.sqrt(np.mean(r_adaptive[sc]["adaptive_err"] ** 2, axis=1))
        rmse_imm = np.sqrt(np.mean(r_imm[sc]["errs"] ** 2, axis=1))
        diff = rmse_a - rmse_imm  # positive = IMM better (lower RMSE)

        t_stat, t_p = stats.ttest_rel(rmse_a, rmse_imm)
        w_stat, w_p = stats.wilcoxon(rmse_a, rmse_imm)

        mean_diff = diff.mean()
        sem_diff = stats.sem(diff)
        ci = stats.t.interval(0.95, df=len(diff) - 1, loc=mean_diff, scale=sem_diff)
        cohens_d = mean_diff / diff.std(ddof=1)
        pct_of_baseline = abs(mean_diff) / rmse_imm.mean() * 100

        nees_a = r_adaptive[sc]["adaptive_nees"].mean(axis=1)
        nees_imm = r_imm[sc]["nees_all"].mean(axis=1)
        t_stat_n, t_p_n = stats.ttest_rel(nees_a, nees_imm)

        print(f"--- {sc} ---")
        print(f"RMSE: Adaptive mean={rmse_a.mean():.2f}m, IMM mean={rmse_imm.mean():.2f}m, "
              f"diff(Adaptive-IMM)={mean_diff:.2f}m ({pct_of_baseline:.2f}% of IMM's RMSE)")
        print(f"  Paired t-test: t={t_stat:.3f}, p={t_p:.4f}   Wilcoxon: p={w_p:.4f}")
        print(f"  95% CI on mean difference: [{ci[0]:.2f}, {ci[1]:.2f}] m")
        print(f"  Cohen's d (paired): {cohens_d:.3f}")
        print(f"NEES: Adaptive mean={nees_a.mean():.3f}, IMM mean={nees_imm.mean():.3f}   "
              f"paired t-test p={t_p_n:.4f}\n")

        results[sc] = {
            "rmse_adaptive": rmse_a, "rmse_imm": rmse_imm, "t_p": t_p, "w_p": w_p,
            "ci": ci, "mean_diff": mean_diff, "cohens_d": cohens_d,
            "pct_of_baseline": pct_of_baseline,
            "nees_adaptive": nees_a, "nees_imm": nees_imm, "t_p_nees": t_p_n,
        }

    with open("outputs/statistical_test_adaptive_vs_imm.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved outputs/statistical_test_adaptive_vs_imm.pkl")


if __name__ == "__main__":
    main()
