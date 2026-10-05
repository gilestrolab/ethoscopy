"""
The k-rule: sleep scored from walking and sustained movement events.

The classic rule calls a 10-s bin "moving" when any frame's velocity exceeds a
threshold, so a single noisy frame breaks a sleep bout. The k-rule asks two
questions instead:

- did the animal walk? Its median position moved more than 10 px from the
  previous bin;
- did it move in place? At least ``k`` sustained movement events started in a
  centred 60-s window around the bin.

A movement event is a run of consecutive frames above the classic velocity test.
Events in which the position never leaves the pixel it started from
("subpixel"), and those that jump out and land back within a pixel in one or two
frames ("flicker"), are tracking noise and are ignored; only the remaining
"sustained" events count. Sleep is then 5 minutes or more of tracked bins with
neither walking nor such movement. A bin without frames is never sleep.

This reproduces ``rule_sustained_k3`` of the sleep-scoring analysis of the lab
archive (sleep_rule.py, October 2026), with ``k = 2`` as the alternative. The
rule is tentative and opt-in: ``sleep_annotation(rule="k")``.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ethoscopy.motion_calibration import pixel_size

# The rule is defined on 10-s bins: a 60-s event window and 30-bin bouts.
K_RULE_BIN_SECONDS = 10
# Median-position step between consecutive bins above which a bin is walking.
WALK_SHIFT_PIXELS = 10.0
# A flicker returns this close to where it started.
RETURN_TOLERANCE_PIXELS = 1.0
# Longest run of moving frames that can be a flicker.
MAX_FLICKER_FRAMES = 2
# Events are counted in bins [i - 3, i + 3).
EVENT_WINDOW_BINS = 6
SUBPIXEL, FLICKER, SUSTAINED = 0, 1, 2


def _runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the runs of True in a boolean array.

    Args:
        mask (np.ndarray): Boolean array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Start (inclusive) and end (exclusive)
            index of every run.
    """
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def classify_events(
    x: np.ndarray,
    y: np.ndarray,
    moving: np.ndarray,
    tolerance: float = RETURN_TOLERANCE_PIXELS,
    max_flicker_frames: int = MAX_FLICKER_FRAMES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split runs of moving frames into subpixel, flicker and sustained events.

    An event is a run of consecutive moving frames; the frame just before it is
    its anchor, so a run that starts on the first frame is not an event. In
    order of precedence, an event is:

    - subpixel: every frame of the run is exactly at the anchor's position;
    - flicker: the run has at most ``max_flicker_frames`` frames and the frame
      after it is back within ``tolerance`` of the anchor;
    - sustained: anything else, including a run that ends on the last frame.

    Args:
        x (np.ndarray): x positions, one per frame, in time order.
        y (np.ndarray): y positions.
        moving (np.ndarray): Boolean, True where the frame passes the movement test.
        tolerance (float, optional): How close a flicker must return, in the units
            of x/y. Default is 1 pixel.
        max_flicker_frames (int, optional): Longest run that can be a flicker.
            Default is 2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Index of the first frame of each event and
            its class (SUBPIXEL, FLICKER or SUSTAINED).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    starts, ends = _runs(np.asarray(moving, dtype=bool))
    keep = starts > 0
    starts, ends = starts[keep], ends[keep]
    if len(starts) == 0:
        return starts, np.zeros(0, dtype=np.int8)
    anchor = starts - 1
    lengths = ends - starts
    offsets = np.cumsum(lengths) - lengths
    # Reason: index every frame of every run at once, instead of looping over events.
    run_of_frame = np.repeat(np.arange(len(starts)), lengths)
    frame = np.arange(lengths.sum()) - offsets[run_of_frame] + starts[run_of_frame]
    away = (x[frame] != x[anchor[run_of_frame]]) | (y[frame] != y[anchor[run_of_frame]])
    subpixel = ~np.logical_or.reduceat(away, offsets)
    has_next = ends < len(x)
    after = np.minimum(ends, len(x) - 1)
    back = np.hypot(x[after] - x[anchor], y[after] - y[anchor]) <= tolerance
    flicker = (lengths <= max_flicker_frames) & has_next & back
    classes = np.where(subpixel, SUBPIXEL, np.where(flicker, FLICKER, SUSTAINED))
    return starts, classes.astype(np.int8)


def k_rule_bins(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xy_dist_log10x1000: np.ndarray,
    k: int = 3,
    pixel: float = 1.0,
    velocity_correction_coef: float = 3e-3,
    min_sleep_bins: float = 30,
) -> pd.DataFrame:
    """
    Score one animal's frames with the k-rule, bin by bin.

    Args:
        t (np.ndarray): Frame times in seconds, in recording order.
        x (np.ndarray): x positions.
        y (np.ndarray): y positions.
        xy_dist_log10x1000 (np.ndarray): The tracker's per-frame displacement.
        k (int, optional): Sustained events in the 60-s window that make a bin
            awake. Default is 3.
        pixel (float, optional): One pixel in the units of x/y. Default is 1.0.
        velocity_correction_coef (float, optional): As in max_velocity_detector;
            a frame moves when its velocity exceeds 1. Default is 3e-3.
        min_sleep_bins (float, optional): Shortest sleep bout in bins. Default is 30.

    Returns:
        pd.DataFrame: One row per 10-s bin from the first to the last bin with
            frames: 't' (bin start, s), 'has_data', 'walking', 'sustained' (events
            starting in the bin), 'micro_awake' and 'asleep'.
    """
    columns = ["t", "has_data", "walking", "sustained", "micro_awake", "asleep"]
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        return pd.DataFrame(columns=columns)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    frame_bin = np.floor(t / K_RULE_BIN_SECONDS).astype(np.int64)
    medians = (
        pd.DataFrame({"bin": frame_bin, "x": x, "y": y})
        .groupby("bin")
        .agg(x=("x", "median"), y=("y", "median"), n=("x", "size"))
    )
    first = medians.index.min()
    grid = medians.reindex(np.arange(first, medians.index.max() + 1))
    has_data = grid["n"].notna().to_numpy()
    step = np.hypot(
        np.diff(grid["x"].to_numpy(), prepend=np.nan),
        np.diff(grid["y"].to_numpy(), prepend=np.nan),
    )
    # Reason: an unknown step (first bin, or a neighbouring bin without frames) counts as walking.
    still = has_data & (np.nan_to_num(step, nan=np.inf) <= WALK_SHIFT_PIXELS * pixel)

    velocity = 10 ** (np.asarray(xy_dist_log10x1000, dtype=float) / 1000.0)
    moving = velocity / velocity_correction_coef > 1.0
    starts, classes = classify_events(
        x, y, moving, tolerance=RETURN_TOLERANCE_PIXELS * pixel
    )
    n_bins = len(grid)
    event_bin = frame_bin[starts[classes == SUSTAINED]] - first
    sustained = np.bincount(event_bin, minlength=n_bins)[:n_bins]
    cumulative = np.r_[0, np.cumsum(sustained)]
    half = EVENT_WINDOW_BINS // 2
    lo = np.clip(np.arange(n_bins) - half, 0, n_bins)
    hi = np.clip(np.arange(n_bins) + EVENT_WINDOW_BINS - half, 0, n_bins)
    micro_awake = (cumulative[hi] - cumulative[lo]) >= k

    asleep = np.zeros(n_bins, dtype=bool)
    for start, end in zip(*_runs(still & ~micro_awake)):
        if end - start >= min_sleep_bins:
            asleep[start:end] = True
    return pd.DataFrame(
        {
            "t": grid.index.to_numpy() * K_RULE_BIN_SECONDS,
            "has_data": has_data,
            "walking": has_data & ~still,
            "sustained": sustained,
            "micro_awake": micro_awake,
            "asleep": asleep,
        }
    )


def _tracked_frames(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the frames the tracker inferred rather than observed.

    Row order is kept: events are runs of consecutive rows, in recording order.

    Args:
        data (pd.DataFrame): Raw frames of one animal.

    Returns:
        pd.DataFrame: The observed frames.
    """
    if "is_inferred" in data.columns:
        # Reason: the column can be stored as TEXT '0'/'1'; anything not numerically 0 is dropped.
        observed = pd.to_numeric(data["is_inferred"], errors="coerce") == 0
        data = data[observed.to_numpy()]
    return data


def k_rule_annotation(
    data: pd.DataFrame,
    motion_detector_function,
    k: int = 3,
    pixel: Optional[float] = None,
    min_sleep_duration: int = 300,
    masking_duration: int = 6,
    velocity_correction_coef: float = 3e-3,
) -> Optional[pd.DataFrame]:
    """
    Sleep annotation with the k-rule; the body of sleep_annotation(rule="k").

    The classic detector still runs, on the same frames, so its columns
    ('moving', 'max_velocity', ...) are reported unchanged next to the k-rule's.

    Args:
        data (pd.DataFrame): Raw tracking data from a single animal.
        motion_detector_function (callable): Detector for the classic columns.
        k (int, optional): Sustained events per 60 s that make a bin awake. Default is 3.
        pixel (float, optional): One pixel in the units of x/y. None infers it:
            1 for positions in pixels (load_ethoscope), 1/500 for positions as a
            fraction of the ROI width. Default is None.
        min_sleep_duration (int, optional): Shortest sleep bout in seconds. Default is 300.
        masking_duration (int, optional): Passed to the classic detector. Default is 6.
        velocity_correction_coef (float, optional): As in max_velocity_detector. Default is 3e-3.

    Returns:
        Optional[pd.DataFrame]: The classic columns plus 'walking', 'sustained' and
            'micro_awake', with 'asleep' from the k-rule and 'is_interpolated' True
            for bins without frames; None for fewer than 100 observed frames.
    """
    frames = _tracked_frames(data)
    if len(frames.index) < 100:
        return None
    binned = motion_detector_function(
        frames,
        K_RULE_BIN_SECONDS,
        masking_duration=masking_duration,
        velocity_correction_coef=velocity_correction_coef,
    )
    if binned is None:
        binned = pd.DataFrame({"t": [], "moving": []})
    # Reason: the rule is scored even where the classic detector found too few frames.
    binned = binned.astype({"t": np.int64})
    x = frames["x"].to_numpy(dtype=float)
    rule = k_rule_bins(
        frames["t"].to_numpy(),
        x,
        frames["y"].to_numpy(),
        frames["xy_dist_log10x1000"].to_numpy(),
        k=k,
        pixel=pixel_size(x) if pixel is None else pixel,
        velocity_correction_coef=velocity_correction_coef,
        min_sleep_bins=min_sleep_duration / K_RULE_BIN_SECONDS,
    )
    # Reason: the classic detector drops sparse windows, so its bins are a subset of the rule's.
    out = rule[["t"]].merge(binned, how="left", on="t")
    # Reason: as in the classic path, a bin the detector did not score is not moving.
    out["moving"] = np.where(out["moving"].isna(), False, out["moving"]).astype(bool)
    out["is_interpolated"] = ~rule["has_data"].to_numpy()
    for column in ("walking", "sustained", "micro_awake", "asleep"):
        out[column] = rule[column].to_numpy()
    return out
