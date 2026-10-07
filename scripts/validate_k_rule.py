"""
Check sleep_annotation(rule="k") against the reference implementation of the
k-rule, and export the parity fixtures used by the tests.

The reference is sleep_rule.py (with its helper module bona_fide_sleep.py) from
the sleep-scoring analysis of the lab archive. Pass the directory holding both,
unmodified; nothing here edits them.

    # Whole-database parity: the per-ROI sleep fraction (asleep bins over bins
    # with frames) must equal the reference's rule_sustained_k3 and _k2 exactly.
    python scripts/validate_k_rule.py parity --reference DIR DB [DB ...]

    # Export fixture segments: raw frames and the reference's per-bin results.
    python scripts/validate_k_rule.py export --reference DIR \\
        --segment LABEL DB ROI START_S DURATION_S [--segment ...] \\
        --out tests/data [--r-out ../rethomics/sleepr/tests/testthat]

ethoscopy reads the databases through load_ethoscope with reference_hour=None,
so its 10-s bins start at the recording start, as the reference's do, and scores
them with untracked="break", the reference's treatment of bins without frames.
The exported fixtures also carry ethoscopy's own untracked="immobile" results,
for the R port to match.
"""

import argparse
import shutil
import sqlite3
import sys
import urllib.parse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ethoscopy.analyse import sleep_annotation
from ethoscopy.load import load_ethoscope
from ethoscopy.sleep_rules import k_rule_bins

K_VALUES = (3, 2)


def import_reference(directory):
    """
    Import the unmodified reference module.

    Args:
        directory (str): Folder holding sleep_rule.py and bona_fide_sleep.py.

    Returns:
        module: The sleep_rule module.
    """
    sys.path.insert(0, str(Path(directory).resolve()))
    import sleep_rule

    return sleep_rule


def ethoscopy_fractions(db, rois):
    """
    Per-ROI k-rule sleep fraction from ethoscopy, as the reference defines it.

    Args:
        db (str): Database path.
        rois (list): ROI numbers to score.

    Returns:
        pd.DataFrame: One row per ROI with 'roi' and 'k3', 'k2' (None when
            sleep_annotation declines the ROI).
    """
    meta = pd.DataFrame(
        {
            "path": db,
            "machine_name": Path(db).stem,
            "region_id": rois,
            "id": [f"roi_{r}" for r in rois],
        }
    )
    data = load_ethoscope(meta, reference_hour=None, progress=False, verbose=False)
    rows = []
    for roi in rois:
        fly = data[data.id == f"roi_{roi}"].drop(columns="id")
        row = {"roi": roi}
        for k in K_VALUES:
            scored = sleep_annotation(fly, rule="k", k=k, untracked="break")
            row[f"k{k}"] = (
                None
                if scored is None
                else scored.asleep.sum() / (~scored.is_interpolated).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def parity(reference, dbs):
    """
    Compare ethoscopy with the reference on every ROI of every database.

    Args:
        reference (module): The sleep_rule module.
        dbs (list): Database paths.

    Returns:
        bool: True when every ROI agrees exactly for every k.
    """
    all_equal = True
    for db in dbs:
        ref = reference.analyse_db(db)
        mine = ethoscopy_fractions(db, ref.roi.tolist())
        both = ref[["roi", "rule_sustained_k3", "rule_sustained_k2"]].merge(
            mine, on="roi"
        )
        line = [Path(db).name, f"{len(both)} ROIs"]
        for k in K_VALUES:
            a, b = both[f"rule_sustained_k{k}"], both[f"k{k}"].astype(float)
            equal = int((a == b).sum())
            all_equal &= equal == len(both)
            line.append(
                f"k{k}: {equal}/{len(both)} equal, max |diff| {np.nanmax(np.abs(a - b)):.3g}"
            )
        print(" | ".join(line))
        bad = both[
            (both.rule_sustained_k3 != both.k3.astype(float))
            | (both.rule_sustained_k2 != both.k2.astype(float))
        ]
        if len(bad):
            print(bad.to_string(index=False))
    return all_equal


def read_segment(db, roi, start_s, duration_s):
    """
    Raw rows of one ROI in a time window, inferred rows included.

    Args:
        db (str): Database path.
        roi (int): ROI number.
        start_s (float): Window start, seconds since the recording start.
        duration_s (float): Window length in seconds.

    Returns:
        pd.DataFrame: Columns t (ms), x, y, xy_dist_log10x1000, is_inferred.
    """
    con = sqlite3.connect(
        f"file:{urllib.parse.quote(db)}?mode=ro&immutable=1", uri=True
    )
    try:
        rows = pd.read_sql(
            f"SELECT t, x, y, xy_dist_log10x1000, CAST(is_inferred AS INTEGER) AS is_inferred "
            f"FROM ROI_{int(roi)} WHERE t >= ? AND t < ?",
            con,
            params=(int(start_s * 1000), int((start_s + duration_s) * 1000)),
        )
    finally:
        con.close()
    return rows


def reference_bins(reference, rows):
    """
    The reference's per-bin result for one segment, captured without editing it.

    analyse_fly() calls sleep_fraction() once per scoring, in a fixed order; the
    wrapper records each call's candidate mask, and the fractions it implies are
    checked against those analyse_fly() returns.

    Args:
        reference (module): The sleep_rule module.
        rows (pd.DataFrame): Raw rows from read_segment().

    Returns:
        pd.DataFrame: One row per bin: t (s), has_data, walking, asleep_k3, asleep_k2.
    """
    calls = []
    original = reference.sleep_fraction

    def recording(candidate, has):
        calls.append((np.asarray(candidate & has), np.asarray(has)))
        return original(candidate, has)

    names = ["native", "walk_only"]
    names += [f"rule_k{k}_s{s}" for k, s in reference.RULES]
    names += [f"rule_{c}_k{k}" for c, k in reference.CLASS_RULES]
    observed = rows[rows.is_inferred == 0]
    reference.sleep_fraction = recording
    try:
        result = reference.analyse_fly(
            observed.t.to_numpy(np.int64),
            observed.x.to_numpy(float),
            observed.y.to_numpy(float),
            observed.xy_dist_log10x1000.to_numpy(float),
        )
    finally:
        reference.sleep_fraction = original
    assert len(calls) == len(names), "analyse_fly's call order changed"
    asleep = {}
    for name, (candidate, has) in zip(names, calls):
        mask = np.zeros(len(candidate), dtype=bool)
        starts, ends = reference._runs(candidate)
        for s, e in zip(starts, ends):
            if e - s >= reference.MIN_BOUT_BINS:
                mask[s:e] = True
        assert mask.sum() / max(has.sum(), 1) == result[name], name
        asleep[name] = mask
    has = calls[0][1]
    first_bin = observed.t.min() // (reference.BIN_S * 1000)
    return pd.DataFrame(
        {
            "t": (first_bin + np.arange(len(has))) * reference.BIN_S,
            "has_data": has,
            "walking": has & ~calls[names.index("walk_only")][0],
            **{f"asleep_k{k}": asleep[f"rule_sustained_k{k}"] for k in K_VALUES},
        }
    )


def export(reference, segments, out, r_out=None):
    """
    Write the fixture frames and the reference's per-bin results.

    Args:
        reference (module): The sleep_rule module.
        segments (list): (label, db, roi, start_s, duration_s) tuples.
        out (str): Folder for k_rule_frames.csv and k_rule_bins.csv.
        r_out (str, optional): Second folder (the R package's tests) to copy them to.
    """
    frames, bins = [], []
    for label, db, roi, start_s, duration_s in segments:
        rows = read_segment(db, int(roi), float(start_s), float(duration_s))
        frames.append(rows.assign(fly=label))
        expected = reference_bins(reference, rows)
        observed = rows[rows.is_inferred == 0]
        for k in K_VALUES:
            mine = k_rule_bins(
                observed.t.to_numpy() / 1000.0,
                observed.x.to_numpy(float),
                observed.y.to_numpy(float),
                observed.xy_dist_log10x1000.to_numpy(float),
                k=k,
                untracked="immobile",
            )
            assert mine.t.tolist() == expected.t.tolist(), label
            expected[f"asleep_k{k}_immobile"] = mine.asleep.to_numpy()
        bins.append(expected.assign(fly=label))
        print(
            f"{label}: {len(rows)} rows ({int((rows.is_inferred != 0).sum())} inferred)"
        )
    frames = pd.concat(frames)[
        ["fly", "t", "x", "y", "xy_dist_log10x1000", "is_inferred"]
    ]
    columns = ["fly", "t", "has_data", "walking", "asleep_k3", "asleep_k2"]
    columns += [f"asleep_k{k}_immobile" for k in K_VALUES]
    bins = pd.concat(bins)[columns]
    out = Path(out)
    frames.to_csv(out / "k_rule_frames.csv", index=False)
    bins.to_csv(out / "k_rule_bins.csv", index=False)
    if r_out:
        for name in ("k_rule_frames.csv", "k_rule_bins.csv"):
            shutil.copy(out / name, Path(r_out) / name)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("parity")
    p.add_argument("--reference", required=True)
    p.add_argument("dbs", nargs="+")
    e = sub.add_parser("export")
    e.add_argument("--reference", required=True)
    e.add_argument(
        "--segment",
        nargs=5,
        action="append",
        required=True,
        metavar=("LABEL", "DB", "ROI", "START_S", "DURATION_S"),
    )
    e.add_argument("--out", default="tests/data")
    e.add_argument("--r-out")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    reference = import_reference(args.reference)
    if args.command == "parity":
        sys.exit(0 if parity(reference, args.dbs) else 1)
    export(reference, args.segment, args.out, args.r_out)


if __name__ == "__main__":
    main()
