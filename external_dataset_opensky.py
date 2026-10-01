"""
external_dataset_opensky.py  --  RESEARCH EXTENSION: tests the trained
filters against REAL flight trajectories, not just our own simulator's
truth generator.

IMPORTANT: this script requires internet access to reach the OpenSky
Network REST API (opensky-network.org). It will NOT run inside the
sandboxed environment this project was otherwise developed in. Run
this in Google Colab or on a machine with normal internet access.

Uses OpenSky's REST API directly (plain `requests`, OAuth2 client-
credentials flow), NOT the `traffic`/`pyopensky` libraries' Trino
path -- a standard registered OpenSky account (4000 credits/day,
created at https://opensky-network.org/my-opensky/account) is
sufficient. No Trino/research-access approval is needed. Historical
tracks are available for up to 30 days in the past via the /tracks
endpoint, at a cost of ~30 credits per 1-2 day query (out of your
4000/day budget) -- see the OpenSky REST API docs for the exact
credit table: https://openskynetwork.github.io/opensky-api/rest.html

What this does:
  1. Exchanges your client_id/client_secret for a Bearer token, then
     pulls a real aircraft's historical trajectory via GET /tracks
     (sparse waypoints: added on >2.5 deg track changes, >100m
     altitude changes, on-ground transitions, or at least every 15
     min in cruise -- NOT a uniform 1 Hz series).
  2. Converts lat/lon to a local flat-Earth Cartesian (x, y) frame
     centered on the radar's assumed position, matching this
     project's coordinate convention.
  3. Linearly interpolates the sparse waypoints onto a uniform T=1s
     grid of config.N=100 scans, matching every other scenario in
     this project.
  4. Treats this REAL trajectory as ground truth ("x_true", "y_true")
     and runs it through our OWN, unmodified measurement/detection
     simulator (measurement_model.py, detection_model.py) -- i.e. we
     supply the real motion, our simulator supplies the sensor noise.
  5. Evaluates classical, IMM, and the adaptive filter against it,
     exactly as in evaluate_s_turn.py.

This tests whether the network's learned correction (and IMM's fixed
q_high) generalize to a REAL aircraft's turn dynamics, not just to
variations we ourselves constructed synthetically. This is a
meaningfully stronger generalization claim than anything in the paper
as submitted, and has NOT yet been run.

Picking a good source flight:
  Use https://opensky-network.org/network/explorer or
  flightradar24.com to find a flight with a clear turn near an
  airport (departure/arrival turns are common), get its ICAO24 hex
  address, and note a Unix timestamp during that flight (any time
  within it works -- /tracks returns the whole track for that flight).

Usage (in Colab):
    export OPENSKY_CLIENT_ID=your_client_id
    export OPENSKY_CLIENT_SECRET=your_client_secret
    python external_dataset_opensky.py --icao24 4b1805 --time 1717243200

    !! The --time value above is an EXAMPLE ONLY and will be stale by the
    time you read this (OpenSky's /tracks only covers the last 30 days).
    Always compute a fresh one for a real, recent flight, e.g.:
        python -c "import datetime; print(int((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=3)).timestamp()))"
    then use that number with a real ICAO24 you've found near that date.
"""

import argparse
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

import config as cfg
from adaptive_kf import AdaptiveKF, run_adaptive_filter_numpy
from measurement_model import generate_noisy_measurements, measurement_covariance
from detection_model import apply_missed_detections
from kalman_filter import run_filter_with_covariance
from imm_filter import run_imm
from metrics import compute_nees


def latlon_to_local_xy(lat, lon, lat0, lon0):
    """Simple equirectangular flat-Earth projection, adequate for the
    short ranges (tens of km) this project's radar model assumes.
    Returns (x, y) in meters, x = east, y = north, origin at (lat0, lon0).
    """
    R_EARTH = 6371000.0
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    lat0_rad, lon0_rad = np.radians(lat0), np.radians(lon0)
    x = R_EARTH * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = R_EARTH * (lat_rad - lat0_rad)
    return x, y


import os
import time as time_module

import requests

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"


class TokenManager:
    """Handles OAuth2 client-credentials auth for a standard OpenSky
    account. Reads OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET from the
    environment (set these before running -- never hardcode credentials
    in this file). Tokens expire after 30 minutes and are refreshed
    automatically."""

    def __init__(self):
        self.client_id = os.environ.get("OPENSKY_CLIENT_ID")
        self.client_secret = os.environ.get("OPENSKY_CLIENT_SECRET")
        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "Set OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET environment "
                "variables first (from your OpenSky account's API Client page)."
            )
        self.token = None
        self.expires_at = 0

    def headers(self):
        if self.token is None or time_module.time() > self.expires_at - 30:
            self._refresh()
        return {"Authorization": f"Bearer {self.token}"}

    def _refresh(self):
        r = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        self.expires_at = time_module.time() + data.get("expires_in", 1800)


def fetch_opensky_trajectory(icao24, unix_time, tokens=None):
    """Fetches a real historical trajectory via GET /tracks and returns
    (x_true, y_true) of length config.N, in this project's coordinate
    convention (radar at origin, target at realistic range).

    icao24: lowercase hex transponder address, e.g. "4b1805"
    unix_time: any Unix timestamp (seconds) that falls within the
        flight you want -- /tracks returns the whole track containing
        that moment. Must be within the last 30 days per OpenSky's
        /tracks limitation.
    tokens: a TokenManager instance (created automatically if omitted).

    Requires internet access -- will raise ConnectionError in this
    project's own sandbox.
    """
    if tokens is None:
        tokens = TokenManager()

    resp = requests.get(
        "https://opensky-network.org/api/tracks",
        params={"icao24": icao24, "time": unix_time},
        headers=tokens.headers(),
    )
    resp.raise_for_status()
    track = resp.json()

    path = track["path"]  # list of [time, lat, lon, baro_altitude, true_track, on_ground]
    if len(path) < 2:
        raise ValueError(
            f"Track for icao24={icao24} at time={unix_time} has too few waypoints "
            f"({len(path)}) to interpolate. Try a different timestamp or aircraft."
        )

    t = np.array([p[0] for p in path], dtype=float)
    lat = np.array([p[1] for p in path], dtype=float)
    lon = np.array([p[2] for p in path], dtype=float)

    # Drop any null lat/lon waypoints before projecting/interpolating.
    valid = ~(np.isnan(lat) | np.isnan(lon))
    t, lat, lon = t[valid], lat[valid], lon[valid]
    order = np.argsort(t)
    t, lat, lon = t[order], lat[order], lon[order]

    lat0, lon0 = lat[0], lon[0]
    x, y = latlon_to_local_xy(lat, lon, lat0, lon0)

    # Radar is assumed at the origin in this project's convention;
    # shift the trajectory to sit at a realistic range, matching X0/Y0.
    x = x - x.min() + cfg.X0
    y = y - y.min() + cfg.Y0

    # /tracks waypoints are SPARSE and irregularly spaced (see module
    # docstring) -- interpolate onto a uniform T=1s grid of cfg.N scans,
    # matching every other scenario in this project. Uses the first
    # config.N seconds of the track; pass a different unix_time if you
    # want a different segment of a longer flight.
    t_rel = t - t[0]
    t_grid = np.linspace(0, (cfg.N - 1) * cfg.T, cfg.N)
    if t_rel[-1] < t_grid[-1]:
        raise ValueError(
            f"Track only spans {t_rel[-1]:.0f}s, need at least "
            f"{(cfg.N - 1) * cfg.T:.0f}s for {cfg.N} scans at T={cfg.T}s. "
            f"Pick a longer flight or reduce cfg.N for this run."
        )
    x_resampled = np.interp(t_grid, t_rel, x)
    y_resampled = np.interp(t_grid, t_rel, y)
    return x_resampled, y_resampled


def evaluate_on_real_trajectory(x_true, y_true, model, q_high=500, n_runs=150):
    """Same evaluation as evaluate_s_turn.py, but x_true/y_true come
    from a real flight instead of our synthetic truth generator. Noise,
    missed detections, and clutter are still drawn from OUR simulator,
    with a fresh random seed per Monte Carlo run -- only the underlying
    motion is real."""
    errs_c, errs_imm, errs_a = [], [], []
    nees_c, nees_imm, nees_a = [], [], []

    for run in range(n_runs):
        rng = np.random.default_rng(cfg.BASE_SEED + run)
        x_meas, y_meas, r_meas, theta_meas, r_true, theta_true = \
            generate_noisy_measurements(x_true, y_true, rng)
        x_target_meas, y_target_meas, detected = apply_missed_detections(x_meas, y_meas, rng)

        # No velocity ground truth from position-only ADS-B resampling;
        # approximate via finite differences for NEES's velocity terms.
        vx_true = np.gradient(x_true, cfg.T)
        vy_true = np.gradient(y_true, cfg.T)
        x_true_full = np.stack([x_true, y_true, vx_true, vy_true], axis=1)

        x_est_c, P_hist_c = run_filter_with_covariance(
            x_target_meas, y_target_meas, detected, r_meas, theta_meas, measurement_covariance)
        errs_c.append(np.sqrt((x_est_c[:, 0] - x_true) ** 2 + (x_est_c[:, 1] - y_true) ** 2))
        nees_c.append(compute_nees(x_true_full, x_est_c, P_hist_c, state_dims=[0, 1]))

        x_est_imm, P_hist_imm, _ = run_imm(
            x_target_meas, y_target_meas, detected, r_meas, theta_meas,
            measurement_covariance, q_high=q_high)
        errs_imm.append(np.sqrt((x_est_imm[:, 0] - x_true) ** 2 + (x_est_imm[:, 1] - y_true) ** 2))
        nees_imm.append(compute_nees(x_true_full, x_est_imm, P_hist_imm, state_dims=[0, 1]))

        R_seq = np.zeros((cfg.N, 2, 2))
        for k in range(cfg.N):
            if detected[k]:
                R_seq[k] = measurement_covariance(r_meas[k], theta_meas[k])
        z_seq = np.stack([np.where(detected, x_target_meas, 0.0),
                           np.where(detected, y_target_meas, 0.0)], axis=1)
        traj = {"x_true": x_true_full, "z": z_seq, "R": R_seq, "detected": detected}
        x_est_a, _, P_hist_a = run_adaptive_filter_numpy(model, traj)
        errs_a.append(np.sqrt((x_est_a[:, 0] - x_true) ** 2 + (x_est_a[:, 1] - y_true) ** 2))
        nees_a.append(compute_nees(x_true_full, x_est_a, P_hist_a, state_dims=[0, 1]))

    return (np.array(errs_c), np.array(errs_imm), np.array(errs_a),
            np.array(nees_c), np.array(nees_imm), np.array(nees_a))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--icao24", type=str, required=True,
                    help="ICAO24 hex address of the aircraft, e.g. '4b1805'")
    p.add_argument("--time", type=int, required=True,
                    help="Any Unix timestamp (seconds) within the flight you want; "
                         "/tracks returns the whole track containing that moment.")
    p.add_argument("--checkpoint", type=str, default="outputs/adaptive_kf_standardized_1400.pt")
    p.add_argument("--n_runs", type=int, default=150)
    args = p.parse_args()

    print(f"Fetching real ADS-B track for icao24={args.icao24} covering time={args.time}...")
    x_true, y_true = fetch_opensky_trajectory(args.icao24, args.time)

    model = AdaptiveKF(hidden_size=16)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    print(f"Evaluating classical/IMM/adaptive on this real trajectory ({args.n_runs} MC noise runs)...")
    errs_c, errs_imm, errs_a, nees_c, nees_imm, nees_a = evaluate_on_real_trajectory(
        x_true, y_true, model, n_runs=args.n_runs)

    print(f"Classical: RMSE={np.sqrt(np.mean(errs_c**2)):.1f}m  NEES={nees_c.mean():.2f}")
    print(f"IMM:       RMSE={np.sqrt(np.mean(errs_imm**2)):.1f}m  NEES={nees_imm.mean():.2f}")
    print(f"Adaptive:  RMSE={np.sqrt(np.mean(errs_a**2)):.1f}m  NEES={nees_a.mean():.2f}")

    with open("outputs/real_trajectory_results.pkl", "wb") as f:
        pickle.dump({
            "icao24": args.icao24, "x_true": x_true, "y_true": y_true,
            "classical_err": errs_c, "imm_err": errs_imm, "adaptive_err": errs_a,
            "classical_nees": nees_c, "imm_nees": nees_imm, "adaptive_nees": nees_a,
        }, f)
    print("Saved outputs/real_trajectory_results.pkl")


if __name__ == "__main__":
    main()
