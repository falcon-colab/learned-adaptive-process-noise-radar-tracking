# EE-891 Stochastic Systems — Radar + Kalman Filter Project

Simulation of a 2-D radar measurement system with stochastic
uncertainties, and a Kalman filter that tracks the target through that
noise. Built as modular stages so each part of the project brief maps
to one runnable file — useful for both understanding and live demo.

## License

This project's own code is released under the **MIT License** (see
`LICENSE` in this directory) — permissive, allows reuse, modification,
and redistribution with attribution, no obligation to open-source
derivative work.

**Third-party dependencies** (from `requirements.txt`), and their
license types, verified directly against each project's own
documentation rather than assumed:

| Dependency | License | Compatible with this project's MIT license? |
|---|---|---|
| NumPy | BSD | Yes — both permissive, no conflict |
| SciPy | BSD-new (3-clause) | Yes |
| Matplotlib | Matplotlib License (its own BSD/PSF-compatible license) | Yes |
| PyTorch | BSD-3-Clause | Yes |

All four are permissive licenses with no copyleft obligation — using
them here doesn't impose any additional licensing requirement on this
project's own code, and none of their license texts need to be
bundled with this repository (that requirement applies to
redistributing their source, not to depending on them as installed
packages). If you `pip install` these via `requirements.txt`, you are
already complying with their license terms as their own maintainers
intend.

**No third-party code was copied into this repository.** Every `.py`
file here was written from scratch for this project, implementing
standard, textbook-level formulas (Kalman filter recursion, IMM,
coordinated-turn kinematics, Jacobian-linearized measurement
conversion) that are themselves cited in the accompanying paper where
a specific published source is being followed, not reproduced from
any external codebase.

## Project structure

| File | Project task(s) | What it does |
|---|---|---|
| `config.py` | — | All parameters in one place (noise levels, PD, false-alarm rate, KF tuning, MC run count) |
| `truth_model.py` | Task 1 | Generates the true constant-velocity trajectory |
| `measurement_model.py` | Task 2 | Adds Gaussian noise in polar (range/azimuth) domain, converts to Cartesian; also provides the Jacobian-linearized measurement covariance used by the KF |
| `detection_model.py` | Tasks 3, 4 | Bernoulli missed detections + Poisson false alarms |
| `kalman_filter.py` | Task 7 | The `ConstantVelocityKF` class implementing predict/update |
| `statistics_analysis.py` | Task 6 (single-run) | Mean, variance, covariance, 95% confidence intervals of noise |
| `run_single_scenario.py` | Tasks 5, 7, 8 | Wires everything together for one full run; the best script for a live demo |
| `monte_carlo_evaluation.py` | Task 6 (MC), Task 8 | Repeats the full pipeline over many independent trials, aggregates statistics and RMSE |
| `main.py` | All | Runs every stage in order, saves all figures to `outputs/`, prints a full summary |
| `q_sweep.py` | Task 7 (tuning) | Optional add-on: sweeps the process-noise scale q and plots how filter RMSE responds, to justify the chosen q value |

## How to run

Full pipeline (what you'd run to regenerate everything for the report):
```
python main.py
```

Stage-by-stage, for demonstration (show the project literally building up
piece by piece):
```
python truth_model.py             # Task 1: just the true trajectory
python measurement_model.py       # Task 2: + noisy measurements
python detection_model.py         # Tasks 3/4: + false alarms & misses
python statistics_analysis.py     # Task 6: noise statistics
python run_single_scenario.py     # Tasks 5/7/8: full single-run pipeline + KF
python monte_carlo_evaluation.py  # Task 6/8: Monte Carlo statistics
python q_sweep.py                 # Optional: process-noise (q) tuning sensitivity
```

Each script saves its own figure(s) into `outputs/` and prints relevant
numbers to the console — good for narrating live or screenshotting
straight into the report/slides.

## Key modeling choices worth mentioning in your report

- **Noise is added in polar coordinates** (range, azimuth), not directly
  in x/y, because that's physically how a radar measures. This is then
  converted to Cartesian for the filter.
- **The Kalman filter's R matrix is NOT constant** — it's recomputed
  each scan via Jacobian linearization of the polar-to-Cartesian
  transform, since fixed angular noise produces range-dependent
  Cartesian position noise (a target further from the radar has more
  Cartesian position uncertainty for the same angular error).
- **Missed detections** cause the filter to skip its update step and
  coast purely on the motion-model prediction for that scan.
- **False alarms** are simulated and visualized (Task 3) but not fed
  into the single-target filter's update step — associating clutter
  with a target requires a data-association/gating step, which is
  outside the scope of a "basic" Kalman filter as specified.
- **Monte Carlo (300 runs)** shows the noise model's sample statistics
  converge to the specified parameters, and gives an averaged,
  trustworthy RMSE comparison rather than relying on one lucky/unlucky
  100-scan run.
- **q-sweep (`q_sweep.py`)** shows RMSE stays flat and near-minimal for
  q roughly ≤ 1, then rises steeply for large q. This makes sense: the
  simulated target is *exactly* constant-velocity with no real process
  noise, so the true optimal q is near zero -- Q only helps the filter
  track real-world deviations from the motion model, which don't occur
  here. The chosen default q=0.5 sits at the edge of the flat,
  near-optimal region -- a safe choice that still leaves headroom to
  track mild real-world maneuvering.
- **Initial target range ≈ 65 km** (not a few km) -- realistic
  first-detection range for a medium-range air-surveillance radar. This
  also means azimuth-angle noise (0.5°) translates into a much larger
  Cartesian position error at long range than it would close-in, since
  cross-range error ≈ range × angular error in radians.

## Parameters (from `config.py`)

- Scan interval T = 1 s, N = 100 scans
- True velocity: vx = 150 m/s, vy = 70 m/s
- Measurement noise: σ_range = 50 m, σ_azimuth = 0.5°
- Detection probability PD = 0.92
- False alarm rate: 1.5 per scan (Poisson)
- Process noise scale q = 0.5 (white-noise-acceleration model)

---

## Part 2: Research extension — learned adaptive-Q Kalman filter

Everything below extends Part 1 additively; none of the files above
were modified in a way that changes their original behavior. See the
paper draft for full methodology and results.

### Additional modules

| File | What it does |
|---|---|
| `maneuver_truth_model.py` | Exact discrete coordinated-turn trajectory generator; fixed evaluation scenarios (`cv_only`/`mild_turn`/`sharp_turn`) plus a randomized-profile generator used only for training data, so training and evaluation scenarios are strictly disjoint |
| `metrics.py` | NEES (filter consistency) computation, chi-squared consistency envelope, track-loss rate |
| `dataset.py` | Builds batches of training trajectories from the simulator (reuses `measurement_model.py`/`detection_model.py` unchanged) |
| `adaptive_kf.py` | `AdaptiveKF`: the GRU-based adaptive-Q filter, differentiable end-to-end through the KF recursion (PyTorch) |
| `continue_training.py` | Resumable training with a held-out validation set; supports both MSE and NLL loss (`--loss mse\|nll`), any hidden size (`--hidden_size`), and importable as `run_training_round(...)` for scripted use |
| `train_phase2.py` | Evaluates classical vs. adaptive filters on the fixed scenarios (`evaluate_on_fixed_scenarios`) |
| `phase1_checkpoint.py` | Classical-filter-only run on the maneuvering scenarios — the motivating "confidently wrong" result |
| `severity_sweep.py` | Detection-probability generalization sweep + clutter-rate sanity check, using the already-trained headline model. Now evaluates all three filters (naive, adaptive, IMM) at each swept condition -- IMM was added after the paper's initial draft explicitly flagged its absence from this sweep as an open question / limitation |
| `statistical_test_adaptive_vs_imm.py` | Formal paired statistical test (paired t-test, Wilcoxon, 95% CI, Cohen's d) for the paper's central Table 1 comparison (adaptive filter vs. IMM), matching the same rigor already applied to the MSE-vs-NLL comparison. Added in response to external review feedback that the earlier "matches or slightly exceeds" framing needed formal statistical backing -- the result showed IMM's advantage is genuinely large on cv_only and meaningfully sized on sharp_turn, small only on mild_turn |
| `track_loss_evaluation.py` | Computes track-loss rate (fraction of runs where position error ever exceeds 1000 m) for naive/IMM/adaptive across the three headline scenarios, reusing the existing `track_loss_rate` function from `metrics.py`. Added because an earlier paper draft claimed this metric was reported but no table actually contained it -- a real gap caught by external review and fixed by building the evaluation properly rather than removing the claim |
| `inference_timing.py` | Wall-clock inference-time comparison (adaptive filter vs. IMM vs. classical), single CPU thread, 200 repetitions of a full 100-scan trajectory. Added in response to a reviewer request to quantify the paper's qualitative complexity claim -- the honest result is that the adaptive filter is *slower* than IMM (12.1-12.9 ms vs. 10.4-10.5 ms per trajectory), not faster, despite involving one filter rather than two; reported plainly in the paper rather than omitted |
| `reproduce_all.py` | **Runs the entire pipeline, in order, from scratch — every stage, not just the original 6. Covers Stages 0-6 above plus IMM (with its systematic q_high selection), the naive-high-Q baseline, the held-out S-turn evaluation, the training-seed variance study, and the formal MSE-vs-NLL statistical test. If a number is in the paper, this script produces it.** See below. |

### Rigor and robustness modules (added after peer feedback)

| File | What it does |
|---|---|
| `train_standardized.py` | Standardized training protocol: PyTorch's own Adam default learning rate (not hand-picked), `ReduceLROnPlateau` scheduling, best-checkpoint-by-validation selection. Replaces an earlier manually-tuned procedure. Accepts `model_seed` (controls weight init and shuffle order, for seed-variance studies) and `normalize_innovation` (see below). This is the protocol Table I's headline numbers come from |
| `run_standardized_chunk.py` | Thin CLI wrapper around `train_standardized.py` that persists full resumable state (model/optimizer/scheduler) to disk between calls, so training can be split across several shorter runs. Only needed because of this project's own development sandbox time limits — **in Colab, prefer calling `train_standardized()` directly in one uninterrupted run** (see `reproduce_all.py` for the pattern), or use this script's `--epochs_this_call` flag if you'd rather checkpoint periodically anyway |
| `imm_filter.py` | Standard two-model IMM (Bar-Shalom et al.), used as the classical maneuvering-target-tracking baseline. Both models share the naive filter's F/H, differing only in process noise |
| `select_q_high.py` | Systematic grid search (10 candidates) selecting IMM's high-Q mode value on the maneuvering scenarios, not hand-picked. Logs the full candidate sweep to `outputs/q_high_selection.json` |
| `evaluate_naive_high_q.py` | The cheapest possible classical baseline: a single filter with process noise permanently fixed at IMM's high-Q value, no switching, no learning. Evaluated across all four scenarios including `s_turn` |
| `statistical_test_mse_vs_nll.py` | Formal paired statistical test (paired t-test, Wilcoxon, 95% CI, Cohen's d) comparing MSE- and NLL-trained models on the identical 300 noise realizations, replacing an earlier informal comparison |
| `train_seed_variance.py` | Trains 3 independent replicates (`model_seed` 0/1/2) differing only in weight initialization, at a smaller disclosed budget, to characterize whether the headline result depends on a fortunate initialization |
| `evaluate_s_turn.py` | Evaluates classical/IMM/adaptive on a held-out S-turn maneuver (two opposite-direction turns in sequence), structurally different from anything used in training or IMM tuning |
| `external_dataset_opensky.py` | Pulls a real aircraft's historical trajectory from OpenSky's REST API (standard account, OAuth2, no Trino needed) and runs it through this project's own noise/detection simulator. **Requires internet access this project's own development sandbox does not have** — built and pipeline-tested end to end with mock API responses, but never run against real data. This is the one piece of the reproducibility story that needs you, in Colab, not this sandbox. See its docstring for exact usage |

**On `normalize_innovation`** (`adaptive_kf.py`, `train_standardized.py`): an opt-in flag, default `False`, so every checkpoint and script above is completely unaffected unless you pass it explicitly. When `True`, the GRU's input is the innovation whitened elementwise by the innovation covariance's own diagonal, rather than the raw innovation in meters — a scale-invariant feature that empirically improved both maneuvering scenarios at matched budget (see the paper's Results, Section VI). Reported as a single-run, budget-matched finding, not its own multi-seed study; if you want to confirm it more rigorously, `train_seed_variance.py`'s pattern (3 seeds, disclosed budget) is the template to reuse with `normalize_innovation=True`.

### Reproducing every result from scratch

```
pip install -r requirements.txt
python reproduce_all.py            # full reproduction, ~2.5-3.5 hours on CPU
python reproduce_all.py --quick    # ~5-8 minute smoke test (verifies the
                                    # pipeline runs end to end; does NOT
                                    # reproduce paper-accurate numbers)
```

**On the completeness of this script:** an earlier version of `reproduce_all.py` only
covered the original 6 stages and did not include IMM, the naive-high-Q baseline, the
S-turn evaluation, the seed-variance study, or the MSE-vs-NLL statistical test -- meaning
running it end to end would NOT have reproduced several results actually reported in the
paper. This was caught during an explicit re-validation pass and fixed by extending the
script to cover everything; the fix also caught two related bugs along the way, both now
fixed: `severity_sweep.py` was hardcoded to load a checkpoint filename
(`adaptive_kf_weights.pt`) that a fresh run of this script never actually produces (only
the standardized-protocol name, `adaptive_kf_standardized_1400.pt`, is created), which
would have crashed on a genuinely fresh clone; and `train_seed_variance.py` and
`evaluate_naive_high_q.py` didn't apply the `--quick` mode's output-suffix convention,
meaning a quick smoke-test run could have silently overwritten real, paper-reported
checkpoints and results -- the same class of bug documented below, now fixed the same
way. All three fixes were verified by actually running `reproduce_all.py --quick` end to
end (not just re-reading the code) and confirming every real checkpoint and result file
was byte-identical afterward.

No GPU is required or particularly helpful here — the network is a single
16-unit GRU cell, and the differentiable filter recursion is inherently
sequential per timestep, so a GPU does not meaningfully parallelize it.
A standard Google Colab CPU runtime is sufficient; there is no need to
change the runtime type.

This script trains, in order: the headline adaptive-Q model
(hidden_size=16, MSE loss, 1400 trajectories across 7 rounds), the
CV-only-trained ablation, the network-capacity ablation
(hidden_size ∈ {4, 8, 32}; 16 is the headline model, reused), the
NLL-loss ablation (matched to the identical 1400-trajectory budget and
order as the headline model), and finally the detection-probability
and clutter-rate severity sweep. Model checkpoints (`.pt`), result
tables (`.pkl`), training logs (`.jsonl`), and every figure land in
`outputs/`.

**On randomness and exact reproducibility.** All evaluation scenarios
use seeds derived from `config.BASE_SEED = 42`. All training data uses
seeds derived from `TRAIN_SEED_BASE = 100_000` (offset per round) in
`continue_training.py`, and the validation set is fixed at
`VAL_SEED_BASE = 999_000`. Running `reproduce_all.py` without
modification will regenerate the exact same trajectories in the exact
same order used for the paper. PyTorch's own initialization and
CPU floating-point operations are not seeded/pinned bit-for-bit in this
codebase, so exact numerical reproduction to the last decimal is not
guaranteed across machines/PyTorch versions, though results should be
extremely close given the large Monte Carlo sample sizes used
throughout (results are averaged over 150-300 independent trials per
condition specifically to be robust to this kind of small numerical
variation).

### What this extension has NOT been tested on (updated)

Two of the three "another dataset" gaps below have now been closed;
one remains and requires you to run it yourself.

**Closed — structurally novel maneuver, within our own simulator.**
`evaluate_s_turn.py` tests the trained filters against an S-turn
(`config.MANEUVER_SCENARIOS["s_turn"]`): two coordinated turns in
opposite directions in sequence, never seen by the adaptive filter
during training or by IMM during its high-Q selection. Result (150 MC
runs): classical 1025.5m RMSE / NEES 213.4 (never recovers within the
trajectory); IMM 331.0m RMSE / NEES 2.15 (close to ideal); adaptive
386.9m RMSE / NEES 3.25 (outside the 95% consistency envelope). **IMM
generalizes to this novel pattern better than the adaptive filter
does** — reported in the paper's Section VI-G honestly, as evidence
against one of the paper's own working hypotheses, not for it.

**Still open — a genuinely external, real-world dataset.**
`external_dataset_opensky.py` is built, its evaluation pipeline is
tested end to end (using mock API responses matching OpenSky's exact
documented schema), but it has never been run against real data,
because this project's sandbox has no network access to
opensky-network.org. It fetches a real aircraft's historical
trajectory via OpenSky's REST API, projects it into this project's
local Cartesian coordinate convention, and runs it through our own
unmodified noise/detection simulator. To run it:

```
export OPENSKY_CLIENT_ID=your_client_id
export OPENSKY_CLIENT_SECRET=your_client_secret
python external_dataset_opensky.py --icao24 4b1805 --time 1717243200
```

**The `--time` value above is an EXAMPLE ONLY and will be stale by the
time you read this** -- OpenSky's `/tracks` endpoint only covers the
last 30 days, so a hardcoded example date here inevitably expires.
Compute a real, current one:
```
python -c "import datetime; print(int((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)).timestamp()))"
```
then find a real flight active around that date/time and use its
ICAO24 with this timestamp.

**If you get an error, run `debug_opensky_raw.py` first**, before
`external_dataset_opensky.py`. It does only the raw API call (no
parsing, no evaluation pipeline) and prints everything: a sanity check
on your timestamp (catches the two most common mistakes -- passing
milliseconds instead of seconds, and passing a timestamp from today
or the last ~24h, which OpenSky may not have finished processing yet
in its nightly batch), the full token-exchange response, and the full
raw `/tracks` response body. This isolates whether a failure is your
credentials, your timestamp, or something else, before any of this
project's own code touches the data:
```
python debug_opensky_raw.py --icao24 4b1805 --time <your_fresh_timestamp>
```

**Easiest path in Colab: `External_Dataset_Validation.ipynb`.** A
self-contained notebook bundling this script plus every project module
it depends on (config, measurement/detection models, both classical
filters, metrics) inline as cells -- open it directly in Colab, no
need to upload the rest of this repo. You'll still need to upload one
file (your trained checkpoint, e.g. `outputs/adaptive_kf_standardized_1400.pt`)
via the notebook's upload cell, and enter your OpenSky credentials when
prompted. Every bundled cell was verified to execute end-to-end against
a mocked OpenSky response matching their documented schema before being
included here, exercising the exact same code path as running the
script above directly.

Uses a standard OpenSky account (4000 credits/day, create an API
client at https://opensky-network.org/my-opensky/account) via plain
REST + OAuth2 -- no Trino/research-access approval needed. Historical
tracks go back up to 30 days at roughly 30 credits per query. Pick a
flight with a clear turn near an airport (departure/arrival turns are
common) via https://opensky-network.org/network/explorer or
flightradar24.com, get its ICAO24 hex address, and use any Unix
timestamp within that flight. This is the one piece of the
reproducibility story that genuinely requires you, not this sandbox,
to run.

Beyond these two, everything else in this project, training and
evaluation alike, still uses trajectories generated by this project's
own simulator; no real radar recordings have been used anywhere.

## Item 7, resolved: full-budget multi-seed replication -- DONE

Was previously an open gap (training-seed robustness only established at a reduced 400-
trajectory/12-epoch budget, not the full 1400-trajectory/25-epoch headline protocol). Closed
by running `train_seed_variance_fullbudget.py` locally (see `ITEM7_LOCAL_RUN_INSTRUCTIONS.md`
for how this was run) and returning `outputs/seed_variance_fullbudget_results.pkl`.

Verified before use: seed 0's re-evaluated numbers matched the already-published headline
values in Table 1 to 5 decimal places on all three scenarios, confirming the run correctly
reused the headline checkpoint rather than accidentally retraining it, and that the whole
setup is internally consistent with everything else already in the paper.

Result: at the full budget, seed-to-seed variance is *tighter* than the earlier reduced-budget
proxy check -- RMSE CV under 0.6% (vs. up to 1.23% at reduced budget), NEES CV under 1.4% --
consistent with more training data and epochs producing a more stable, less seed-dependent
model. This is now the paper's primary seed-robustness evidence (Results, Section 6), and the
reduced-budget study is kept only as a secondary historical comparison point. The paper's
Limitations and Conclusion sections have been updated accordingly; this is no longer listed
as outstanding future work.

`track_loss_evaluation.py` also now supports threshold sensitivity analysis (500/1000/2000 m
sweep), used to justify the 1000 m default rather than leave it looking arbitrary -- see the
inline sweep code added for this in the same script.

---

## Note on this repository vs. the local development package

A handful of large, superseded intermediate result files were left out of
this repository to keep it a reasonable size for upload (they are
historical/secondary, not required to reproduce any number printed in the
paper): `outputs/seed_variance_results.pkl` (the reduced-budget seed-variance
proxy study -- superseded as primary evidence by
`seed_variance_fullbudget_results.pkl`, which *is* included), an earlier
150-run and an earlier non-1400-trajectory version of the phase-2 results
file, and the single-run `normalize_innovation` ablation's results/checkpoint
(reported in the paper as a single-run finding, not a multi-seed study).
Every file that any paper table, figure, or number in Section 6/7 depends on
is included and was independently re-verified against the paper before this
repository was created.

---

## Camera-ready addition: Sage–Husa online adaptive-Q baseline

Added in response to peer review of the ICCI 2026 submission, which asked for a
classical *adaptive* process-noise baseline (not just the fixed-Q and discrete-mode-switching
ones already in the paper). `sage_husa_kf.py` implements the classical Sage–Husa recursive
process-noise estimator (Sage & Husa, 1969), which re-estimates Q scan-by-scan from the
innovation sequence using a forgetting factor `b` -- no training, no discrete mode bank.

`evaluate_sage_husa.py` selects `b` by the same grid-search methodology already used for
IMM's `q_high` in `select_q_high.py` (minimizing combined mild_turn/sharp_turn RMSE over 40
trials; selected `b = 0.5`), then evaluates the selected filter over the same 300-run,
fixed-scenario protocol used for every other baseline in the paper, using identical seeds.
Results: `outputs/sage_husa_results_300.pkl`, `outputs/sage_husa_b_selection.json`. Reported
in the paper's Results section alongside the naive-high-Q baseline.
