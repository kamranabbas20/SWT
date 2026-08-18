#!/usr/bin/env python3
"""Fetch sample data, and explain clearly when the network will not allow it.

Most open seismic datasets live on hosts that a restricted egress proxy blocks
(see ``data/README.md``).  This script therefore does two things: it downloads
what it can, and for what it cannot it prints the manual instructions instead of
a stack trace.  A blocked download is a normal outcome here, not an error.

    python data/fetch.py           # fetch what is reachable
    python data/fetch.py --check   # report reachability, download nothing
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")

#: Small files served from hosts that are typically reachable.  These are
#: format smoke tests -- real headers, real quirks -- not tie-quality data.
REACHABLE = [
    {
        "name": "f3.sgy",
        "url": "https://raw.githubusercontent.com/equinor/segyio/master/test-data/f3.sgy",
        "what": "F3 (Netherlands) subset, 414 traces, 4 ms - SEG-Y reader smoke test",
    },
    {
        "name": "P-129.LAS",
        "url": "https://raw.githubusercontent.com/agilescientific/welly/main/tests/assets/P-129_out.LAS",
        "what": "Real LAS with a full curve suite - LAS reader and unit-detection test",
    },
]

#: Datasets worth having that need a manual download.
MANUAL = [
    {
        "name": "F3 Demo (full)",
        "where": "https://terranubis.com/datainfo/F3-Demo-2020",
        "what": "Full F3 survey with wells, horizons and a complete SEG-Y volume.",
        "how": "Download the OpendTect project or the SEG-Y volume, put the .sgy "
               "and .las files in data/f3/.",
    },
    {
        "name": "Volve",
        "where": "https://data.equinor.com/",
        "what": "Equinor's fully open field dataset: seismic, well logs, checkshots.",
        "how": "Register, download the seismic and the well logs for 15/9-F-11 or "
               "similar, put them in data/volve/.",
    },
    {
        "name": "Poseidon",
        "where": "https://wiki.seg.org/wiki/Open_data",
        "what": "Australian 3D survey with wells; good for deviated-well testing.",
        "how": "Follow the SEG open-data links, put files in data/poseidon/.",
    },
]

TIMEOUT = 60


def reachable(url: str) -> tuple[bool, str]:
    """Probe a URL without downloading its body."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        # A HEAD may be refused where a GET succeeds; treat 4xx as reachable.
        return exc.code < 500, f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError) as exc:
        return False, str(getattr(exc, "reason", exc))


def download(entry: dict) -> bool:
    """Fetch one file, skipping it if it is already present."""
    os.makedirs(SAMPLES, exist_ok=True)
    target = os.path.join(SAMPLES, entry["name"])

    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"  [have] {entry['name']} ({os.path.getsize(target):,} bytes)")
        return True

    print(f"  [get ] {entry['name']} ... ", end="", flush=True)
    try:
        with urllib.request.urlopen(entry["url"], timeout=TIMEOUT) as response:
            payload = response.read()
    except (urllib.error.URLError, OSError) as exc:
        print(f"blocked ({getattr(exc, 'reason', exc)})")
        return False

    with open(target, "wb") as handle:
        handle.write(payload)
    print(f"{len(payload):,} bytes")
    return True


def print_manual_instructions() -> None:
    print("\nDatasets that need a manual download:\n")
    for entry in MANUAL:
        print(f"  {entry['name']}")
        print(f"    {entry['what']}")
        print(f"    from: {entry['where']}")
        print(f"    then: {entry['how']}\n")
    print(
        "None of these is required to develop or test SWT. The synthetic forward\n"
        "model in swt.forward is the primary validation target precisely because it\n"
        "has a known answer, which no real dataset does."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report reachability without downloading"
    )
    args = parser.parse_args()

    if args.check:
        print("Reachability:\n")
        for entry in REACHABLE + [{"name": e["name"], "url": e["where"]} for e in MANUAL]:
            ok, detail = reachable(entry["url"])
            print(f"  {'OK     ' if ok else 'BLOCKED'}  {entry['name']:<20} {detail}")
        return 0

    print("Fetching sample data:\n")
    results = [download(entry) for entry in REACHABLE]
    print_manual_instructions()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
