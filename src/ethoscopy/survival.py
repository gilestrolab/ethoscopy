"""
Survival analysis helpers: immobility-based death detection, Kaplan-Meier
estimation and per-subject survival tables.

The functions here are pure numpy/pandas and know nothing about behavpy; the
behavpy methods (``curate_dead_animals``, ``survival_table``, ``km_survival_plot``)
are thin wrappers around them so both code paths share one definition of
"death".
"""

from math import floor
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Two-sided 95 % normal quantile used for the Greenwood confidence band.
Z_95 = 1.959963984540054


def sliding_window_death(
    t: np.ndarray,
    mov: np.ndarray,
    time_window_s: float,
    step: int,
    prop_immobile: float,
    min_coverage: Optional[float] = None,
) -> Optional[float]:
    """
    Return the start of the first window in which the animal looks dead.

    Windows start every ``step`` seconds from ``int(t.min())`` up to (but not
    including) ``int(t.max())`` and cover ``[start, start + time_window_s]``
    with both ends inclusive. The animal is dead from ``start`` if the mean of
    ``mov`` inside the window is ``<= prop_immobile``. This is exactly the rule
    ``curate_dead_animals`` has always applied; windows with no data are
    ignored (their mean is NaN).

    Args:
        t (np.ndarray): Timestamps in seconds (any order).
        mov (np.ndarray): Movement per timestamp (bool or numeric, NaN allowed).
        time_window_s (float): Window length in seconds.
        step (int): Distance between consecutive window starts in seconds.
        prop_immobile (float): Mean movement at or below which the animal is dead.
        min_coverage (Optional[float]): If given (0-1), a window is only judged
            when it holds at least this fraction of the samples expected from
            the animal's own median sampling interval. Guards against a short
            tail of data at a recording stop being read as death. ``None``
            keeps the historical behaviour of judging every window.

    Returns:
        Optional[float]: Window start (seconds) of the first death window, or
        ``None`` if the animal never qualifies.
    """
    t = np.asarray(t, dtype=float)
    mov = np.asarray(mov, dtype=float)
    if len(t) == 0:
        return None

    starts = np.arange(int(t.min()), int(t.max()), step)
    if len(starts) == 0:
        return None

    order = np.argsort(t, kind="stable")
    ts, ms = t[order], mov[order]

    # Reason: prefix sums give every window mean in O(n) instead of one
    # boolean mask per window, which mattered for multi-week recordings.
    valid = ~np.isnan(ms)
    csum = np.concatenate([[0.0], np.cumsum(np.where(valid, ms, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(valid)])
    lo = np.searchsorted(ts, starts, side="left")
    hi = np.searchsorted(ts, starts + time_window_s, side="right")

    counts = ccnt[hi] - ccnt[lo]
    means = np.full(len(starts), np.nan)
    has_data = counts > 0
    means[has_data] = (csum[hi] - csum[lo])[has_data] / counts[has_data]

    if min_coverage is not None:
        interval = median_sampling_interval(ts)
        if interval is not None:
            expected = time_window_s / interval + 1
            means[(hi - lo) < min_coverage * expected] = np.nan

    dead = np.flatnonzero(means <= prop_immobile)
    return float(starts[dead[0]]) if len(dead) else None


def median_sampling_interval(t_sorted: np.ndarray) -> Optional[float]:
    """
    Median gap between consecutive distinct timestamps.

    Args:
        t_sorted (np.ndarray): Timestamps in ascending order.

    Returns:
        Optional[float]: Median positive gap in seconds, ``None`` if fewer than
        two distinct timestamps exist.
    """
    gaps = np.diff(t_sorted)
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if len(gaps) else None


def zero_run_death(
    t: np.ndarray, mov: np.ndarray, zero_run_s: float
) -> Optional[float]:
    """
    Return the start of the first run of zero movement lasting ``zero_run_s``.

    Missing movement values are treated as movement, so a run of zeros is only
    ever broken by real activity or by NaN.

    Args:
        t (np.ndarray): Timestamps in seconds, ascending.
        mov (np.ndarray): Movement per timestamp.
        zero_run_s (float): Minimum run length in seconds.

    Returns:
        Optional[float]: Timestamp (seconds) at which the qualifying run
        begins, or ``None``.
    """
    t = np.asarray(t, dtype=float)
    is_zero = np.nan_to_num(np.asarray(mov, dtype=float), nan=1.0) == 0
    if not is_zero.any():
        return None

    edges = np.diff(np.concatenate([[False], is_zero, [False]]).astype(int))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1)  # exclusive
    durations = t[run_ends - 1] - t[run_starts]
    qualifying = np.flatnonzero(durations >= zero_run_s)
    return float(t[run_starts[qualifying[0]]]) if len(qualifying) else None


def kaplan_meier(
    times: Sequence[float], events: Sequence[int], z: float = Z_95
) -> pd.DataFrame:
    """
    Kaplan-Meier survival estimate with log-transformed Greenwood confidence band.

    Args:
        times (Sequence[float]): Time to death or to censoring for each subject.
        events (Sequence[int]): 1 if the subject died at ``times``, 0 if censored.
        z (float): Normal quantile for the confidence band. Defaults to 95 %.

    Returns:
        pd.DataFrame: One row per distinct observed time, ordered by time, with
        columns ``time``, ``n_at_risk``, ``n_events``, ``n_censored``,
        ``survival``, ``ci_lower``, ``ci_upper``. Rows where only censoring
        happened carry the survival unchanged, so the curve extends to the last
        observation rather than the last death.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    if len(times) == 0:
        return pd.DataFrame(
            columns=[
                "time",
                "n_at_risk",
                "n_events",
                "n_censored",
                "survival",
                "ci_lower",
                "ci_upper",
            ]
        )

    uniq, inverse = np.unique(times, return_inverse=True)
    n_events = np.bincount(inverse, weights=events, minlength=len(uniq))
    n_total = np.bincount(inverse, minlength=len(uniq))
    n_censored = n_total - n_events
    # Subjects at risk just before each time = everyone with time >= t.
    n_at_risk = len(times) - np.concatenate([[0], np.cumsum(n_total)[:-1]])

    survival = np.cumprod(1 - n_events / n_at_risk)

    # Greenwood variance of log S(t); undefined once S hits zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            n_at_risk > n_events,
            n_events / (n_at_risk * (n_at_risk - n_events)),
            np.nan,
        )
        var_log = np.cumsum(np.nan_to_num(terms, nan=0.0))
        se_log = np.sqrt(var_log)
        log_s = np.log(survival)
        ci_lower = np.exp(log_s - z * se_log)
        ci_upper = np.exp(log_s + z * se_log)

    dead_curve = survival <= 0
    ci_lower[dead_curve] = 0.0
    ci_upper[dead_curve] = 0.0

    return pd.DataFrame(
        {
            "time": uniq,
            "n_at_risk": n_at_risk.astype(int),
            "n_events": n_events.astype(int),
            "n_censored": n_censored.astype(int),
            "survival": survival,
            "ci_lower": np.clip(ci_lower, 0, 1),
            "ci_upper": np.clip(ci_upper, 0, 1),
        }
    )


def _segment_bounds(t_sorted: np.ndarray, gap_s: float) -> list[tuple[int, int]]:
    """Split an ascending time array into ``(start, end)`` slices at gaps > ``gap_s``."""
    breaks = np.flatnonzero(np.diff(t_sorted) > gap_s) + 1
    edges = np.concatenate([[0], breaks, [len(t_sorted)]])
    return list(zip(edges[:-1], edges[1:]))


def survival_table(
    data: pd.DataFrame,
    subject: pd.Series,
    t_column: str = "t",
    mov_column: str = "moving",
    time_window: int = 24,
    prop_immobile: float = 0.01,
    resolution: int = 24,
    restart_gap: float = 1.0,
    min_coverage: Optional[float] = None,
    zero_run_hours: Optional[float] = None,
    second_mov_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a per-subject survival table (time and event indicator) for Kaplan-Meier.

    Each subject's recording is split into segments wherever the time axis
    jumps by more than ``restart_gap`` hours (a machine stop or restart). Death
    is searched in every segment with the same sliding-window rule as
    ``curate_dead_animals`` and optionally a zero-movement run rule; the
    earliest detection wins. Time is measured from the subject's first sample.

    Args:
        data (pd.DataFrame): Behavioural data indexed by specimen id.
        subject (pd.Series): Maps every specimen id to a subject key. Ids
            sharing a key are treated as one animal recorded in several
            sessions; their time ranges must not overlap.
        t_column (str): Timestamp column, seconds.
        mov_column (str): Movement column.
        time_window (int): Immobility window in hours.
        prop_immobile (float): Mean movement threshold for death.
        resolution (int): Window starts per window length.
        restart_gap (float): Gap in hours that separates recording segments.
        min_coverage (Optional[float]): Passed to :func:`sliding_window_death`.
        zero_run_hours (Optional[float]): If set, a contiguous zero-movement
            run of at least this many hours also counts as death.
        second_mov_column (Optional[str]): Second movement column; death in
            either column counts.

    Returns:
        pd.DataFrame: Indexed by subject key with columns ``id`` (first
        specimen id of the subject), ``T`` (hours from first sample to death
        or to last sample), ``E`` (1 death, 0 censored), ``start_time``,
        ``end_time`` (seconds) and ``n_segments``.

    Raises:
        ValueError: If ``resolution`` is invalid or merged specimens overlap in time.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if resolution > time_window:
        raise ValueError("resolution cannot be larger than time_window")

    time_window_s = 3600 * time_window
    step = floor(time_window_s / resolution)
    gap_s = 3600 * restart_gap
    zero_run_s = 3600 * zero_run_hours if zero_run_hours else None
    mov_columns = [mov_column] + ([second_mov_column] if second_mov_column else [])

    frame = data[[t_column] + mov_columns].copy()
    frame["_subject"] = subject.reindex(frame.index).values
    frame["_id"] = frame.index
    _check_no_overlap(frame, t_column)

    rows = []
    for key, grp in frame.groupby("_subject", sort=False):
        grp = grp.sort_values(t_column)
        t = grp[t_column].to_numpy(dtype=float)
        movs = [grp[c].to_numpy(dtype=float) for c in mov_columns]
        segments = _segment_bounds(t, gap_s)

        death = None
        for s, e in segments:
            for mov in movs:
                candidates = [
                    sliding_window_death(
                        t[s:e],
                        mov[s:e],
                        time_window_s,
                        step,
                        prop_immobile,
                        min_coverage,
                    )
                ]
                if zero_run_s:
                    candidates.append(zero_run_death(t[s:e], mov[s:e], zero_run_s))
                found = [c for c in candidates if c is not None]
                if found and (death is None or min(found) < death):
                    death = min(found)

        end_time = death if death is not None else t[-1]
        rows.append(
            {
                "subject": key,
                "id": grp["_id"].iloc[0],
                "T": (end_time - t[0]) / 3600.0,
                "E": int(death is not None),
                "start_time": t[0],
                "end_time": end_time,
                "n_segments": len(segments),
            }
        )

    return pd.DataFrame(rows).set_index("subject")


def _check_no_overlap(frame: pd.DataFrame, t_column: str) -> None:
    """Raise if two specimens merged into one subject overlap in time."""
    spans = frame.groupby(["_subject", "_id"])[t_column].agg(["min", "max"])
    for key, sub in spans.groupby(level=0):
        if len(sub) < 2:
            continue
        sub = sub.sort_values("min")
        if (sub["min"].values[1:] <= sub["max"].values[:-1]).any():
            raise ValueError(
                f"Specimens merged into subject {key!r} overlap in time. Merging "
                "recording sessions needs a shared clock: shift each session "
                "with baseline() or load with a common reference before "
                "using subject_cols."
            )
