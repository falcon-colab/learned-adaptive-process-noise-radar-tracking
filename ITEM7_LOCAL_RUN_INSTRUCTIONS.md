# Running the full-budget seed-variance study locally (Item 7)

## Why this needs to run locally

The paper's Limitations section currently discloses a real gap: training-seed
robustness (does the headline result depend on which random initialization
was used?) is only established at a *reduced* budget (400 trajectories, 12
epochs), not the full 1400-trajectory/25-epoch budget the headline result
itself uses. Closing this properly means training 2 more models at the full
budget and comparing all 3 (including the existing headline checkpoint as
seed 0).

This was attempted in the development sandbox first. Real, measured finding:
each epoch takes **~185-215 seconds** on that machine, and the sandbox's
per-command time limit only reliably allows *one epoch per call* — a
2-epoch chunk already exceeds it. Finishing both seeds that way would mean
roughly 49 separate back-and-forth exchanges just for training, which isn't
a good use of time when a normal machine with no per-command time limit can
just run this straight through in one sitting.

## What to run

The script needs no chunking on a normal machine — it calls the training
function directly and runs to completion in one process.

```bash
cd radar_kf_project
pip install -r requirements.txt          # if not already installed
python train_seed_variance_fullbudget.py
```

**No GPU needed** — the model is a single 16-unit GRU cell; the sequential,
per-timestep nature of the differentiable filter recursion means a GPU
would not meaningfully speed this up (same reasoning already documented for
the rest of this project). Standard CPU training, same as everything else
here.

## What it does

1. Re-evaluates the *existing* headline checkpoint
   (`outputs/adaptive_kf_standardized_1400.pt`) as seed 0 — confirmed via
   the sandbox run that this checkpoint already **is** `model_seed=0` at
   this exact budget, so it is reused, not wastefully retrained.
2. Trains 2 new models (`model_seed=1`, `model_seed=2`) at the identical
   protocol: 1400 trajectories, 25 epochs, hidden size 16, MSE loss,
   `scheduler_patience=10` (the actual default the headline run used,
   confirmed by reading `reproduce_all.py`'s own training call — this
   matters, since the earlier reduced-budget seed study used a different,
   scaled-down patience of 3, which would not be a fair comparison here).
3. Evaluates all 3 on the same 300-run protocol as Table 1.
4. Prints and saves a summary: RMSE and NEES mean ± std and coefficient of
   variation across the 3 full-budget seeds, per scenario.

## Expected runtime

Calibrated from the real, measured per-epoch cost above:

- **~83 minutes per seed** (25 epochs × ~200s), **~2.8 hours total** for
  both new seeds, on hardware comparable to the sandbox's.
- Could be faster or slower depending on your CPU — the estimate is a
  reasonable ballpark, not a guarantee.
- The script has no built-in checkpointing between seeds, but each seed's
  model checkpoint is saved to disk as soon as that seed's training
  finishes (`outputs/seedvar_fullbudget_seed1.pt`,
  `outputs/seedvar_fullbudget_seed2.pt`), so if you need to stop after
  seed 1 finishes and resume seed 2 separately later, that's possible —
  just ask and this can be split into two independent runs instead of one
  script covering both seeds back to back.

## What to send back

Once it finishes, everything needed is in one file:

```
outputs/seed_variance_fullbudget_results.pkl
```

Send that file back (or just paste the console output — the script prints
the same summary it saves), and the paper's Limitations section, the
seed-variance discussion, and the Conclusion's "future work" line can all
be updated with the real full-budget numbers instead of the current
disclosed gap.
