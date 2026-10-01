"""
run_standardized_chunk.py  --  Calls train_standardized for a few more
epochs, persisting full resumable state (model, optimizer, scheduler,
best checkpoint, history) to disk between calls. This exists ONLY
because of this sandbox's per-command time limit; the training
protocol itself (train_standardized.py) is a single, standard,
continuous training run and should be invoked directly and
uninterrupted in reproduce_all.py / on Colab.
"""
import argparse
import os
import pickle
import torch

from train_standardized import train_standardized

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--epochs_this_call", type=int, default=4)
    p.add_argument("--scheduler_patience", type=int, default=5)
    p.add_argument("--hidden_size", type=int, default=16)
    p.add_argument("--loss", type=str, default="mse")
    p.add_argument("--checkpoint", type=str, default="outputs/adaptive_kf_standardized.pt")
    p.add_argument("--log", type=str, default="outputs/standardized_training_log.jsonl")
    p.add_argument("--state_file", type=str, default="outputs/standardized_resume_state.pkl")
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--normalize_innovation", action="store_true")
    args = p.parse_args()

    resume_state = None
    if os.path.exists(args.state_file):
        with open(args.state_file, "rb") as f:
            resume_state = pickle.load(f)

    model, history, final_state = train_standardized(
        n_train=args.n_train, n_val=args.n_val, n_epochs=args.epochs_this_call,
        hidden_size=args.hidden_size, loss=args.loss,
        checkpoint=args.checkpoint, log=args.log,
        scheduler_patience=args.scheduler_patience, resume_state=resume_state,
        model_seed=args.model_seed, normalize_innovation=args.normalize_innovation,
    )

    with open(args.state_file, "wb") as f:
        pickle.dump(final_state, f)
    print(f"State saved to {args.state_file} (total epochs so far: {final_state['epoch']})")
