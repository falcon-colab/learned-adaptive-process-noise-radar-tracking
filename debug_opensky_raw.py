"""
debug_opensky_raw.py -- MINIMAL diagnostic tool, no dependency on the
rest of this project. Does ONLY the raw OpenSky request and prints
everything about the response, so we can see exactly what's failing
before it reaches any of our parsing/interpolation code.

Run this FIRST when troubleshooting, before external_dataset_opensky.py.

Usage:
    export OPENSKY_CLIENT_ID=your_client_id
    export OPENSKY_CLIENT_SECRET=your_client_secret
    python debug_opensky_raw.py --icao24 4b1805 --time 1717243200

    !! The --time value above is an EXAMPLE ONLY and will be stale by the
    time you read this. This script's own sanity_check_timestamp() will
    tell you if your value is too old/too new/malformed -- trust that
    over any hardcoded example in this docstring.
"""
import argparse
import datetime
import json
import os
import sys
import time as time_module

import requests

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
TRACKS_URL = "https://opensky-network.org/api/tracks"


def sanity_check_timestamp(unix_time):
    print(f"\n--- Timestamp sanity check ---")
    print(f"Raw value passed in: {unix_time}")
    print(f"Digit count: {len(str(int(unix_time)))}")
    if len(str(int(unix_time))) > 10:
        print("!! WARNING: This looks like MILLISECONDS, not seconds.")
        print(f"   OpenSky (like Unix time generally) expects SECONDS since epoch.")
        print(f"   If this came from JavaScript's Date.now() or a web tool that")
        print(f"   reports milliseconds, divide by 1000: {int(unix_time) // 1000}")
    try:
        dt = datetime.datetime.fromtimestamp(unix_time, tz=datetime.timezone.utc)
        print(f"Interpreted as: {dt} UTC")
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        age_days = (now - dt).total_seconds() / 86400
        print(f"Age: {age_days:.2f} days ago")
        if age_days < 1:
            print("!! WARNING: This is less than 1 day old. OpenSky processes")
            print("   historical flight data in a NIGHTLY BATCH for the")
            print("   previous completed UTC day -- today's flights, including")
            print("   ones that already landed a few hours ago, may not be")
            print("   in /tracks yet. Try a timestamp from 2+ days ago.")
        if age_days > 30:
            print("!! WARNING: This is more than 30 days old. OpenSky's /tracks")
            print("   endpoint only covers the last 30 days.")
    except (ValueError, OverflowError, OSError) as e:
        print(f"!! This value does not parse as a valid Unix timestamp at all: {e}")
        print("   This alone would explain 'error reading the time'.")
    print("--- end sanity check ---\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--icao24", type=str, required=True)
    p.add_argument("--time", type=float, required=True,
                    help="Unix timestamp in SECONDS (not milliseconds)")
    args = p.parse_args()

    sanity_check_timestamp(args.time)

    client_id = os.environ.get("OPENSKY_CLIENT_ID")
    client_secret = os.environ.get("OPENSKY_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("ERROR: OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET not set.")
        sys.exit(1)

    print("--- Requesting OAuth2 token ---")
    r = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials",
              "client_id": client_id, "client_secret": client_secret},
    )
    print(f"Token endpoint status: {r.status_code}")
    if r.status_code != 200:
        print(f"Token request FAILED. Full response body:\n{r.text}")
        print("\nIf this fails, the issue is your credentials, not timestamps --")
        print("double check client_id/client_secret on your OpenSky account page.")
        sys.exit(1)
    token = r.json()["access_token"]
    print("Token obtained OK.\n")

    print(f"--- Requesting /tracks for icao24={args.icao24}, time={int(args.time)} ---")
    resp = requests.get(
        TRACKS_URL,
        params={"icao24": args.icao24, "time": int(args.time)},
        headers={"Authorization": f"Bearer {token}"},
    )
    print(f"Tracks endpoint status: {resp.status_code}")
    print(f"Response headers: {dict(resp.headers)}")
    print(f"\nFull raw response body:")
    print(resp.text[:3000])  # cap in case it's huge

    if resp.status_code == 200:
        try:
            data = resp.json()
            print(f"\nParsed OK as JSON. Top-level keys: {list(data.keys())}")
            if "path" in data:
                print(f"Number of waypoints in path: {len(data['path'])}")
                if len(data["path"]) > 0:
                    print(f"First waypoint: {data['path'][0]}")
                    print(f"Last waypoint:  {data['path'][-1]}")
        except json.JSONDecodeError as e:
            print(f"\n!! Response was status 200 but is NOT valid JSON: {e}")
    else:
        print(f"\n!! Non-200 status. This is likely an auth, rate-limit, or")
        print(f"   invalid-parameter issue rather than a code bug -- the response")
        print(f"   body above should say which.")


if __name__ == "__main__":
    main()
