"""
train_standardized.py  --  Standardized training protocol, replacing
the manual, undocumented LR step-down and per-round data chunking used
during initial development.

Every hyperparameter choice below is either a PyTorch library default
(the most defensible, least arbitrary choice available) or a standard,
named, citable practice. None were tuned to hit a target result.

- Optimizer: Adam with PyTorch's own default learning rate (1e-3).
  Nothing about this problem motivated a different starting point, so
  the library default is used rather than a hand-picked value.
- Scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, using
  PyTorch's own default factor (0.1) and patience (10), monitoring
  held-out validation loss. This replaces the earlier practice of
  manually lowering the learning rate between rounds based on visually
  inspecting the loss curve, which is not reproducible by a reader
  running this script and was never a documented, citable rule.
- Model selection: the checkpoint with the LOWEST held-out validation
  loss over training is kept (standard early-stopping-adjacent
  practice), not simply whichever epoch training happened to end on.
- Data: ONE FIXED training set of n_train trajectories, generated
  once, reused with reshuffling every epoch -- standard practice.
  This replaces the earlier "fresh trajectories every round" pattern,
  which was an artifact of chunking training into many short
  sandbox-time-limited calls during development, not a methodological
  choice, and was never presented as one; see reproduce_all.py.
- Training runs for a fixed, generous number of epochs (no early
  stopping on top of the LR schedule) since compute cost is not a
  constraint for this project; the scheduler and best-checkpoint
  selection together handle convergence and overfitting.
"""

import time
import numpy as np
import torch
import torch.optim as optim

from adaptive_kf import AdaptiveKF
from dataset import build_dataset

BURN_IN = 5
TRAIN_SEED_BASE = 200_000   # fresh, disjoint range from all earlier ad-hoc rounds
VAL_SEED_BASE = 999_000     # unchanged: disjoint from training and from eval seeds


def compute_loss_mse(model, traj):
    x_hist, _, _ = model(traj)
    x_true = torch.tensor(traj["x_true"][:, :2], dtype=torch.float32)
    sq_err = ((x_hist[BURN_IN:, :2] - x_true[BURN_IN:]) ** 2).sum(dim=1)
    return sq_err.mean()


def compute_loss_nll(model, traj):
    x_hist, _, P_hist = model(traj)
    x_true = torch.tensor(traj["x_true"], dtype=torch.float32)
    losses = []
    eye4 = torch.eye(4)
    for t in range(BURN_IN, x_hist.shape[0]):
        e = (x_true[t] - x_hist[t]).unsqueeze(1)
        P = P_hist[t]
        P = 0.5 * (P + P.T) + 1e-6 * eye4
        solved = torch.linalg.solve(P, e)
        quad = (e.transpose(0, 1) @ solved).squeeze()
        _, logdet = torch.linalg.slogdet(P)
        losses.append(quad + logdet)
    return torch.stack(losses).mean()


def evaluate_loss(model, data, loss_fn):
    model.eval()
    with torch.no_grad():
        losses = [loss_fn(model, traj).item() for traj in data]
    model.train()
    return float(np.mean(losses))


def train_standardized(n_train=1400, n_val=200, n_epochs=100, batch_size=16,
                        hidden_size=16, loss="mse", checkpoint=None, log=None,
                        train_seed_offset=0, verbose=True,
                        scheduler_patience=10, resume_state=None, model_seed=0,
                        normalize_innovation=False):
    """Standardized, reproducible training run. Returns (best_model,
    history, final_state) where final_state can be passed back in as
    resume_state to continue training across multiple calls (needed
    only because of this sandbox's per-command time limit -- see
    train_standardized_resumable.py for the driver that does this).

    scheduler_patience defaults to PyTorch's own ReduceLROnPlateau
    default (10). When run with a reduced n_epochs budget, patience
    should be scaled down proportionally (documented at the call site,
    not silently changed here) so the scheduler can still meaningfully
    trigger within the available epochs.

    model_seed controls BOTH the model weight initialization (via
    torch.manual_seed) AND the per-epoch shuffling order (via a
    seeded local numpy Generator, not numpy's global RNG state, which
    was previously unseeded and therefore not actually reproducible
    despite appearances). Varying model_seed across otherwise-identical
    runs is how training-seed variance is characterized in
    train_seed_variance.py; the default of 0 matches every checkpoint
    trained earlier in this project.
    """
    loss_fn = compute_loss_mse if loss == "mse" else compute_loss_nll

    if verbose:
        print(f"Building fixed training set: {n_train} trajectories "
              f"(seed base {TRAIN_SEED_BASE + train_seed_offset})")
    train_data = build_dataset(n_train, base_seed=TRAIN_SEED_BASE + train_seed_offset)
    val_data = build_dataset(n_val, base_seed=VAL_SEED_BASE)

    # Seed BEFORE constructing the model, since AdaptiveKF's weight
    # initialization draws from torch's global RNG at construction time.
    # Setting the seed after construction (an earlier bug in this file)
    # silently fails to control initialization at all.
    if resume_state is not None:
        torch.manual_seed(resume_state["torch_seed"])
    else:
        torch.manual_seed(model_seed)

    model = AdaptiveKF(hidden_size=hidden_size, normalize_innovation=normalize_innovation)
    opt = optim.Adam(model.parameters())  # PyTorch default lr=1e-3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=scheduler_patience)
    shuffle_rng = np.random.default_rng(model_seed)

    best_val = float("inf")
    best_state = None
    history = []
    start_epoch = 0

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        opt.load_state_dict(resume_state["opt_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        best_val = resume_state["best_val"]
        best_state = resume_state["best_state"]
        history = resume_state["history"]
        start_epoch = resume_state["epoch"]
        shuffle_rng = np.random.default_rng(resume_state.get("shuffle_seed", model_seed))
        # advance the shuffle stream past epochs already consumed
        for _ in range(start_epoch):
            shuffle_rng.permutation(n_train)
        if verbose:
            print(f"Resumed from epoch {start_epoch}, best_val so far={best_val:.2f}")

    n_batches = int(np.ceil(n_train / batch_size))
    t0 = time.time()
    end_epoch = start_epoch + n_epochs  # n_epochs = epochs to run THIS call

    for epoch in range(start_epoch, end_epoch):
        perm = shuffle_rng.permutation(n_train)
        train_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            opt.zero_grad()
            batch_loss = sum(loss_fn(model, train_data[i]) for i in idx) / len(idx)
            batch_loss.backward()
            opt.step()
            train_loss += batch_loss.item() * len(idx)
        train_loss /= n_train

        val_loss = evaluate_loss(model, val_data, loss_fn)
        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]["lr"]

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                         "val_loss": val_loss, "lr": current_lr})
        if verbose:
            print(f"  epoch {epoch+1:3d}  train={train_loss:9.2f}  "
                  f"val={val_loss:9.2f}  lr={current_lr:.2e}  [{time.time()-t0:.0f}s]")

    best_model = AdaptiveKF(hidden_size=hidden_size, normalize_innovation=normalize_innovation)
    best_model.load_state_dict(best_state)

    if checkpoint:
        torch.save(best_model.state_dict(), checkpoint)
        if verbose:
            print(f"Best checkpoint so far (val={best_val:.2f}) saved to {checkpoint}")
    if log:
        import json, os
        os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
        with open(log, "w") as f:
            for h in history:
                f.write(json.dumps(h) + "\n")

    final_state = {
        "torch_seed": model_seed, "model_state": model.state_dict(), "opt_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict(), "best_val": best_val,
        "best_state": best_state, "history": history, "epoch": end_epoch,
        "shuffle_seed": model_seed,
    }
    return best_model, history, final_state
