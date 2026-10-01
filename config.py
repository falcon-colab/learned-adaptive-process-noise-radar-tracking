"""
config.py
---------
Single source of truth for all simulation parameters, taken directly
from the EE-891 project specification. Every other module imports from
here so there is exactly one place to change a parameter (and one place
to point to in your report when justifying a value).
"""

import numpy as np

# ---------------- Scenario timing ----------------
T = 1.0                 # scan interval (s)
N = 100                  # number of scans

# ---------------- True target motion (constant velocity) ----------------
VX_TRUE = 150.0          # m/s
VY_TRUE = 70.0           # m/s
X0, Y0 = 60000.0, 25000.0  # initial position (m) -- radar assumed at origin.
                            # ~65 km initial range: realistic first-detection
                            # range for a medium-range air-surveillance radar
                            # (real radars rarely pick up targets at a few km).

# ---------------- Measurement noise (polar domain) ----------------
SIGMA_R = 50.0                        # range noise std dev (m)
SIGMA_THETA_DEG = 0.5                 # azimuth noise std dev (deg)
SIGMA_THETA = np.deg2rad(SIGMA_THETA_DEG)

# ---------------- Detection model ----------------
PD = 0.92                # probability of detection (Bernoulli)
LAMBDA_FA = 1.5           # mean false alarms per scan (Poisson rate)
FA_BOX_MARGIN = 3000.0   # meters beyond trajectory bounding box, for false alarm placement

# ---------------- Kalman filter tuning ----------------
Q_SCALE = 0.5             # process noise intensity (white-noise-acceleration model)
P0_POS_STD = 500.0        # initial position uncertainty (m) std dev
P0_VEL_STD = 200.0        # initial velocity uncertainty (m/s) std dev

# ---------------- Monte Carlo ----------------
MC_RUNS = 300             # number of independent Monte Carlo trials

# ---------------- Reproducibility ----------------
BASE_SEED = 42

# =====================================================================
# RESEARCH EXTENSION (ICCI paper): maneuvering-scenario parameters
# Added on top of the original EE-891 deliverable without altering any
# of the constants above, so the original coursework results remain
# exactly reproducible. See maneuver_truth_model.py.
# =====================================================================

# Each scenario is a list of segments applied in order; segments must
# sum to N scans. A 'cv' segment holds current heading/speed constant.
# A 'turn' segment applies a coordinated turn at the given rate
# (deg/s, positive = counter-clockwise) for its duration.
MANEUVER_SCENARIOS = {
    "cv_only": [
        ("cv", 100),
    ],
    "mild_turn": [
        ("cv", 30),
        ("turn", 20, 3.0),   # ~60 deg turn over 20 s
        ("cv", 50),
    ],
    "sharp_turn": [
        ("cv", 25),
        ("turn", 15, 9.0),   # ~135 deg turn over 15 s
        ("cv", 60),
    ],
    "s_turn": [
        ("cv", 20),
        ("turn", 15, 6.0),    # ~90 deg turn right over 15 s
        ("cv", 10),
        ("turn", 15, -6.0),   # ~90 deg turn left over 15 s, opposite direction
        ("cv", 40),
    ],
}

