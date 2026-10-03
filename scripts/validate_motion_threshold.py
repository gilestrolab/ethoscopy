#!/usr/bin/env python3
"""
Score ground-truth recordings to validate the movement threshold.

Record a night with dead or anaesthetised flies (every 10-s window is truly
immobile) under each illumination, ideally with live flies in the same run. This
script loads the recordings and reports, per fly and summarised per group:

- how often an immobile fly is scored as moving (false-positive rate) with the
  fixed threshold and with velocity_threshold="auto";
- the resulting sleep fraction, which for an immobile fly should be ~100%;
- how much of the time the tracker lost the fly (untracked fraction), which the
  default scoring silently counts as sleep.

The metadata CSV is a normal ethoscopy metadata file with two extra columns:
one marking the ground truth (default ``truth``: ``immobile`` for dead or
anaesthetised flies, anything else for live ones) and one naming the condition
being compared (default ``condition``, e.g. old_IR / new_IR).

Example:
    python scripts/validate_motion_threshold.py \\
        --metadata ground_truth.csv --local-dir /mnt/ethoscope_data/results \\
        --reference-hour 9 --out validation.csv
"""

import argparse
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import ethoscopy as etho

METHODS = {
    "fixed": {},
    "auto": {"velocity_threshold": "auto"},
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-hour",
        type=float,
        default=None,
        help="Hour (UTC) of lights on; default estimates it from the snapshots.",
    )
    parser.add_argument("--truth-column", default="truth")
    parser.add_argument("--immobile-label", default="immobile")
    parser.add_argument("--group-by", default="condition")
    parser.add_argument("--day-length", type=int, default=24)
    parser.add_argument("--lights-off", type=int, default=12)
    parser.add_argument(
        "--skip-hours",
        type=float,
        default=1.0,
        help="Hours to drop at the start (handling, settling).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Per-fly CSV.")
    return parser.parse_args()


def reference_hour(meta: pd.DataFrame, given: float | None) -> float:
    """
    Return the lights-on hour, measuring it from snapshots when not given.

    Args:
        meta (pd.DataFrame): Linked metadata.
        given (float | None): Hour supplied on the command line.

    Returns:
        float: Lights-on hour in UTC.
    """
    if given is not None:
        return given
    cycle = etho.estimate_light_cycle(meta, progress=False)
    hour = float(np.nanmedian(cycle["reference_hour"]))
    print(f"Estimated lights-on hour from snapshots: {hour:.2f} UTC")
    return hour


def score_fly(
    raw: pd.DataFrame, detector, skip_s: float, day_length: int, lights_off: int
) -> list[dict]:
    """
    Score one fly with every method and summarise per light phase.

    Args:
        raw (pd.DataFrame): Raw tracking data of one fly, without 'id'.
        detector: Motion detector with the light cycle bound.
        skip_s (float): Seconds to drop from the start.
        day_length (int): Day length in hours.
        lights_off (int): Hour of lights off.

    Returns:
        list[dict]: One record per method and phase.
    """
    records = []
    for method, kwargs in METHODS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scored = etho.sleep_annotation(
                raw, motion_detector_function=detector, **kwargs
            )
        if scored is None:
            continue
        scored = scored[scored["t"] >= scored["t"].min() + skip_s]
        phase = etho.motion_calibration.light_phase(
            scored["t"].to_numpy(dtype=float), day_length, lights_off
        )
        for name in ("light", "dark"):
            part = scored[phase == name]
            if part.empty:
                continue
            tracked = ~part["is_interpolated"]
            records.append(
                {
                    "method": method,
                    "phase": name,
                    "moving_rate": part.loc[tracked, "moving"].mean(),
                    "sleep": part["asleep"].mean(),
                    "untracked_fraction": 1 - tracked.mean(),
                    "threshold": (
                        part["velocity_threshold"].median()
                        if "velocity_threshold" in part
                        else 1.0
                    ),
                }
            )
    return records


def main() -> None:
    """Load the recordings, score every fly and print the summary."""
    args = parse_args()
    meta = etho.link_meta_index(args.metadata, args.local_dir)
    for column in (args.truth_column, args.group_by):
        if column not in meta.columns:
            raise KeyError(f"metadata has no '{column}' column")

    raw = etho.load_ethoscope(
        meta, reference_hour=reference_hour(meta, args.reference_hour), progress=True
    )
    detector = partial(
        etho.max_velocity_detector,
        day_length=args.day_length,
        lights_off=args.lights_off,
    )
    info = meta.set_index("id")[[args.truth_column, args.group_by]]

    rows = []
    for fly, group in raw.groupby("id"):
        for record in score_fly(
            group.drop(columns="id").reset_index(drop=True),
            detector,
            args.skip_hours * 3600,
            args.day_length,
            args.lights_off,
        ):
            rows.append({"id": fly, **info.loc[fly].to_dict(), **record})
    per_fly = pd.DataFrame(rows)
    per_fly["immobile"] = per_fly[args.truth_column] == args.immobile_label

    summary = (
        per_fly.groupby(["immobile", args.group_by, "phase", "method"])[
            ["moving_rate", "sleep", "untracked_fraction", "threshold"]
        ]
        .median()
        .round(3)
    )
    pd.set_option("display.width", 200)
    print("\nMedian per group (immobile=True: every window is truly immobile)")
    print(summary.to_string())
    immobile = per_fly[per_fly["immobile"]]
    if not immobile.empty:
        print(
            "\nFor immobile flies, moving_rate is the false-positive rate and sleep "
            "should be ~1. A high untracked_fraction means sleep there is inferred, "
            "not observed."
        )
    if args.out is not None:
        per_fly.to_csv(args.out, index=False)
        print(f"\nPer-fly results written to {args.out}")


if __name__ == "__main__":
    main()
