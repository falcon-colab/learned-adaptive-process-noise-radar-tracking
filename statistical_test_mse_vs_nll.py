"""
statistical_test_mse_vs_nll.py  --  RESEARCH EXTENSION: formal paired
statistical test comparing the MSE- and NLL-trained adaptive filters,
replacing the earlier informal "these numbers look close" comparison.

Both models were evaluated on the IDENTICAL 300 Monte Carlo noise
realizations per scenario (same seed formula in
evaluate_on_fixed_scenarios), so a PAIRED test is the correct and more
statistically powerful choice over an unpaired one -- it removes
between-run noise-realization variance from the comparison.

Reports both statistical significance (paired t-test, Wilcoxon
signed-rank) and practical effect size (Cohen's d for paired
differences), since with n=300 pairs even small, practically
unimportant differences reach formal significance -- the two are not
the same question and both are reported here rather than conflated.

Requires outputs/phase2_results_300.pkl (MSE) and
outputs/phase2_nll_results_300.pkl (NLL) to already exist.

Usage:
    python statistical_test_mse_vs_nll.py
"""

import pickle

import numpy as np
from scipy import stats


def main():
    with open("outputs/phase2_results_300.pkl", "rb") as f:
        mse = pickle.load(f)
    with open("outputs/phase2_nll_results_300.pkl", "rb") as f:
        nll = pickle.load(f)

    print("=== Paired statistical test: MSE vs NLL loss (same 300 noise realizations per scenario) ===\n")
    results = {}
    for sc in ["cv_only", "mild_turn", "sharp_turn"]:
        rmse_mse = np.sqrt(np.mean(mse[sc]["adaptive_err"] ** 2, axis=1))
        rmse_nll = np.sqrt(np.mean(nll[sc]["adaptive_err"] ** 2, axis=1))
        diff = rmse_mse - rmse_nll

        t_stat, t_p = stats.ttest_rel(rmse_mse, rmse_nll)
        w_stat, w_p = stats.wilcoxon(rmse_mse, rmse_nll)

        mean_diff = diff.mean()
        sem_diff = stats.sem(diff)
        ci = stats.t.interval(0.95, df=len(diff) - 1, loc=mean_diff, scale=sem_diff)
        cohens_d = mean_diff / diff.std(ddof=1)
        pct_of_baseline = abs(mean_diff) / rmse_mse.mean() * 100

        nees_mse = mse[sc]["adaptive_nees"].mean(axis=1)
        nees_nll = nll[sc]["adaptive_nees"].mean(axis=1)
        t_stat_n, t_p_n = stats.ttest_rel(nees_mse, nees_nll)

        print(f"--- {sc} ---")
        print(f"RMSE: MSE mean={rmse_mse.mean():.2f}m, NLL mean={rmse_nll.mean():.2f}m, "
              f"diff(MSE-NLL)={mean_diff:.2f}m ({pct_of_baseline:.2f}% of baseline)")
        print(f"  Paired t-test: t={t_stat:.3f}, p={t_p:.4f}   Wilcoxon: p={w_p:.4f}")
        print(f"  95% CI on mean difference: [{ci[0]:.2f}, {ci[1]:.2f}] m")
        print(f"  Cohen's d (paired): {cohens_d:.3f}")
        print(f"NEES: MSE mean={nees_mse.mean():.3f}, NLL mean={nees_nll.mean():.3f}   "
              f"paired t-test p={t_p_n:.4f}\n")

        results[sc] = {
            "rmse_mse": rmse_mse, "rmse_nll": rmse_nll, "t_p": t_p, "w_p": w_p,
            "ci": ci, "mean_diff": mean_diff, "cohens_d": cohens_d,
            "nees_mse": nees_mse, "nees_nll": nees_nll, "t_p_nees": t_p_n,
        }

    with open("outputs/mse_vs_nll_statistical_test.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved outputs/mse_vs_nll_statistical_test.pkl")


if __name__ == "__main__":
    main()
