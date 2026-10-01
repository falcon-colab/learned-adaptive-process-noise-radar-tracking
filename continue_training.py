"""
continue_training.py  --  RESEARCH EXTENSION, Phase 2 convergence.

Chunked, resumable training with a proper held-out VALIDATION set
(seeds 999_000+, never used for training or for the fixed-scenario
evaluation) so we can tell genuine convergence apart from just
"train loss went down" -- the two can diverge if the model starts
overfitting to the specific random maneuver profiles it's seen.

Run repeatedly (each call trains a few more epochs on a fresh slice of
training data and re-measures validation loss); progress is logged to
a JSON-lines file so the trend is visible across calls without needing
one single very long-running command.

Usage:
    python continue_training.py --seed_offset 0   --n_train 200 --n_epochs 8
    python continue_training.py --seed_offset 200 --n_train 200 --n_epochs 8
    ...
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim

from adaptive_kf import AdaptiveKF
from dataset import build_dataset

BURN_IN = 5
TRAIN_SEED_BASE = 100_000     # fresh range, distinct from earlier ad-hoc rounds
VAL_SEED_BASE = 999_000       # fixed, disjoint from training AND from
                               # the fixed-scenario eval seeds (cfg.BASE_SEED+run)
VAL_SIZE = 60


def compute_loss_mse(model, traj):
    x_hist, _, _ = model(traj)
    x_true = torch.tensor(traj["x_true"][:, :2], dtype=torch.float32)
    sq_err = ((x_hist[BURN_IN:, :2] - x_true[BURN_IN:]) ** 2).sum(dim=1)
    return sq_err.mean()


def compute_loss_nll(model, traj):
    """Gaussian negative-log-likelihood loss over the FULL 4-D state,
    following Mortada et al. (Recursive KalmanNet, EUSIPCO 2025):
        L_t = e_t^T P_t^-1 e_t + log det P_t
    Unlike RKN, we don't learn P from scratch -- P still comes from the
    classical KF's own exact covariance recursion, just with a learned
    Q-scale. This tests whether an NLL objective further improves
    consistency/accuracy over plain MSE for OUR simpler, structurally
    different setup, not whether NLL works in general (already shown
    by RKN in a different problem).
    """
    x_hist, _, P_hist = model(traj)
    x_true = torch.tensor(traj["x_true"], dtype=torch.float32)  # full 4-D
    losses = []
    eye4 = torch.eye(4)
    for t in range(BURN_IN, x_hist.shape[0]):
        e = (x_true[t] - x_hist[t]).unsqueeze(1)          # (4,1)
        P = P_hist[t]
        P = 0.5 * (P + P.T) + 1e-6 * eye4                  # symmetrize + jitter
        solved = torch.linalg.solve(P, e)                  # P^-1 e
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_offset", type=int, required=True)
    p.add_argument("--n_train", type=int, default=200)
    p.add_argument("--n_epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--hidden_size", type=int, default=16)
    p.add_argument("--checkpoint", type=str, default="outputs/adaptive_kf_weights.pt")
    p.add_argument("--log", type=str, default="outputs/training_log.jsonl")
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "nll"])
    args = p.parse_args()
    run_training_round(args.seed_offset, args.n_train, args.n_epochs, args.batch_size,
                        args.lr, args.hidden_size, args.checkpoint, args.log, args.loss)


def run_training_round(seed_offset, n_train=200, n_epochs=8, batch_size=16, lr=6e-4,
                        hidden_size=16, checkpoint="outputs/adaptive_kf_weights.pt",
                        log="outputs/training_log.jsonl", loss="mse"):
    """Importable version of main() -- runs one training round (build data,
    train n_epochs, checkpoint, log) and returns the trained model. This is
    what reproduce_all.py calls in a loop; the CLI in main() is a thin
    wrapper around this for single-round, ad-hoc invocation."""
    loss_fn = compute_loss_mse if loss == "mse" else compute_loss_nll

    model = AdaptiveKF(hidden_size=hidden_size)
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
        print(f"Loaded checkpoint {checkpoint}")
    else:
        print("No checkpoint found -- starting fresh")

    opt = optim.Adam(model.parameters(), lr=lr)

    print(f"Building {n_train} training trajectories "
          f"(seed base {TRAIN_SEED_BASE + seed_offset}), loss={loss}...")
    train_data = build_dataset(n_train, base_seed=TRAIN_SEED_BASE + seed_offset)
    val_data = build_dataset(VAL_SIZE, base_seed=VAL_SEED_BASE)

    n_batches = int(np.ceil(n_train / batch_size))
    t0 = time.time()
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            opt.zero_grad()
            batch_loss = sum(loss_fn(model, train_data[i]) for i in idx) / len(idx)
            batch_loss.backward()
            opt.step()
            epoch_loss += batch_loss.item() * len(idx)
        epoch_loss /= n_train
        print(f"  epoch {epoch+1}/{n_epochs}  train-loss={epoch_loss:.2f}  "
              f"[{time.time()-t0:.0f}s]")

    val_loss = evaluate_loss(model, val_data, loss_fn)
    if loss == "mse":
        print(f"Validation loss: {val_loss:.1f}  (~{val_loss**0.5:.1f} m RMSE-equivalent)")
    else:
        print(f"Validation NLL: {val_loss:.2f}")

    torch.save(model.state_dict(), checkpoint)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps({
            "seed_offset": seed_offset, "n_train": n_train,
            "n_epochs": n_epochs, "hidden_size": hidden_size,
            "loss_type": loss, "final_train_loss": epoch_loss, "val_loss": val_loss,
        }) + "\n")
    print(f"Checkpoint saved to {checkpoint}, log appended to {log}")
    return model


if __name__ == "__main__":
    main()
