"""
maneuver_truth_model.py  --  RESEARCH EXTENSION (not part of the
original EE-891 deliverable).

Generates a maneuvering true trajectory by concatenating constant-
velocity (CV) segments with exact discrete coordinated-turn (CT)
segments. This deliberately mismatches the classical filter's motion
model (which always assumes CV, per kalman_filter.py) -- that mismatch
is the whole point: it's what an adaptive/learned process-noise
correction has something real to fix.

Coordinated-turn (CT) model, exact discretization for a KNOWN,
piecewise-constant turn rate omega (rad/s) over interval T:

  x_k  = x_{k-1} + (sin(wT)/w) vx_{k-1} - ((1-cos(wT))/w) vy_{k-1}
  y_k  = y_{k-1} + ((1-cos(wT))/w) vx_{k-1} + (sin(wT)/w) vy_{k-1}
  vx_k = cos(wT) vx_{k-1} - sin(wT) vy_{k-1}
  vy_k = sin(wT) vx_{k-1} + cos(wT) vy_{k-1}

This preserves speed exactly (a coordinated turn only changes heading),
and reduces to the standard CV model as w -> 0. Because it's linear in
state for a KNOWN w, the true trajectory generator itself stays simple
and exact -- no EKF/UKF needed anywhere, including in the filter, which
is one of this project's deliberate design choices (see paper's
Section IV rationale: a lightweight correction within a plain linear
KF, not a switch to UKF/EKF machinery).
"""

import numpy as np
import matplotlib.pyplot as plt
import config as cfg


def _ct_step(x, y, vx, vy, omega, T):
    """One exact CT-model step. omega in rad/s. omega=0 reduces to CV."""
    if abs(omega) < 1e-9:
        return x + vx * T, y + vy * T, vx, vy
    s, c = np.sin(omega * T), np.cos(omega * T)
    x_new = x + (s / omega) * vx - ((1 - c) / omega) * vy
    y_new = y + ((1 - c) / omega) * vx + (s / omega) * vy
    vx_new = c * vx - s * vy
    vy_new = s * vx + c * vy
    return x_new, y_new, vx_new, vy_new


def _generate_from_segments(segments, x0, y0, vx0, vy0, T, n_total):
    """Shared stepper: applies a list of ('cv', n) / ('turn', n, rate_deg_s)
    segments in order. Used by both the fixed named scenarios (eval-only)
    and the randomized training-scenario generator below."""
    x_true = np.zeros(n_total)
    y_true = np.zeros(n_total)
    vx_true = np.zeros(n_total)
    vy_true = np.zeros(n_total)
    maneuvering = np.zeros(n_total, dtype=bool)

    x, y, vx, vy = x0, y0, vx0, vy0
    k = 0
    for seg in segments:
        kind = seg[0]
        n_scans = seg[1]
        omega = np.deg2rad(seg[2]) if kind == "turn" else 0.0
        for _ in range(n_scans):
            x_true[k], y_true[k], vx_true[k], vy_true[k] = x, y, vx, vy
            maneuvering[k] = (kind == "turn")
            x, y, vx, vy = _ct_step(x, y, vx, vy, omega, T)
            k += 1
    return x_true, y_true, vx_true, vy_true, maneuvering


def random_maneuver_profile(rng, n_total=None, p_cv_only=0.2):
    """Draws a RANDOMIZED segment list for TRAINING data generation --
    deliberately never one of the fixed cfg.MANEUVER_SCENARIOS presets,
    so the network's evaluation scenarios (cv_only/mild_turn/sharp_turn)
    are genuinely held out and never seen during training.

    With probability p_cv_only, returns a pure CV profile (no turn) --
    important so the network also learns "when nothing's happening,
    don't inflate Q", not just "always expect a turn".
    """
    n_total = cfg.N if n_total is None else n_total

    if rng.random() < p_cv_only:
        return [("cv", n_total)]

    turn_start = int(rng.integers(10, n_total - 40))
    turn_duration = int(rng.integers(8, 26))
    turn_duration = min(turn_duration, n_total - turn_start - 10)
    turn_rate = float(rng.uniform(1.5, 10.0)) * (1 if rng.random() < 0.5 else -1)
    tail = n_total - turn_start - turn_duration

    return [
        ("cv", turn_start),
        ("turn", turn_duration, turn_rate),
        ("cv", tail),
    ]


def generate_maneuvering_truth(scenario_name, x0=None, y0=None,
                                vx0=None, vy0=None, T=None):
    """Builds a full N-scan true trajectory from a named scenario in
    config.MANEUVER_SCENARIOS. Returns x_true, y_true, vx_true, vy_true
    arrays of length N, plus a boolean `maneuvering` mask (True during
    turn segments) useful for segment-conditioned analysis/plots.
    """
    x0 = cfg.X0 if x0 is None else x0
    y0 = cfg.Y0 if y0 is None else y0
    vx0 = cfg.VX_TRUE if vx0 is None else vx0
    vy0 = cfg.VY_TRUE if vy0 is None else vy0
    T = cfg.T if T is None else T

    segments = cfg.MANEUVER_SCENARIOS[scenario_name]
    total_scans = sum(seg[1] for seg in segments)
    assert total_scans == cfg.N, (
        f"Scenario '{scenario_name}' segments sum to {total_scans} scans, "
        f"expected N={cfg.N}. Fix config.MANEUVER_SCENARIOS."
    )
    return _generate_from_segments(segments, x0, y0, vx0, vy0, T, cfg.N)


def generate_random_maneuvering_truth(rng, x0=None, y0=None, vx0=None, vy0=None, T=None):
    """Training-only counterpart to generate_maneuvering_truth: draws a
    fresh random profile each call instead of using a fixed preset."""
    x0 = cfg.X0 if x0 is None else x0
    y0 = cfg.Y0 if y0 is None else y0
    vx0 = cfg.VX_TRUE if vx0 is None else vx0
    vy0 = cfg.VY_TRUE if vy0 is None else vy0
    T = cfg.T if T is None else T

    segments = random_maneuver_profile(rng, n_total=cfg.N)
    return _generate_from_segments(segments, x0, y0, vx0, vy0, T, cfg.N)


def plot_scenario_comparison(save_path=None):
    """Overlay all configured scenarios for a quick visual sanity check
    that turn segments look physically reasonable (constant speed,
    smooth heading change)."""
    plt.figure(figsize=(8, 7))
    colors = {"cv_only": "g", "mild_turn": "b", "sharp_turn": "r"}
    for name in cfg.MANEUVER_SCENARIOS:
        x, y, vx, vy, man = generate_maneuvering_truth(name)
        plt.plot(x, y, color=colors.get(name, "k"), label=name, linewidth=2)
        plt.scatter(x[man], y[man], color=colors.get(name, "k"), s=10, alpha=0.5)
    plt.scatter([0], [0], marker="^", c="k", s=100, label="Radar")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title("Maneuvering-scenario truth trajectories\n(dots mark turn segments)")
    plt.legend(); plt.axis("equal"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved {save_path}")
    plt.close()


if __name__ == "__main__":
    for name in cfg.MANEUVER_SCENARIOS:
        x, y, vx, vy, man = generate_maneuvering_truth(name)
        speed = np.hypot(vx, vy)
        print(f"{name}: {man.sum()} maneuvering scans, "
              f"speed range [{speed.min():.2f}, {speed.max():.2f}] m/s "
              f"(should be ~constant -- CT preserves speed)")
    plot_scenario_comparison(save_path="outputs/07_maneuver_scenarios.png")
