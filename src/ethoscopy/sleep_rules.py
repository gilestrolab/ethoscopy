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
"sustained" events count. Sleep is then 5 minutes or more of bins with neither
walking nor such movement.

Bins without frames are handled as in the classic rule. By default
(``untracked="immobile"``) they count as still, and the first bin with frames
after them is walking only if the animal is found more than 10 px from where it
was last seen. Background-subtraction tracking (AdaptiveBGModel) loses still
flies, so this is what keeps their sleep: against pixel-motion truth on two
recordings, night-time error per tube fell from 0.21-0.43 to 0.03-0.05.
``untracked="break"`` never scores such bins as sleep.

In the light the rule scores more sleep than pixel truth, in bins where the fly
does not walk and at most moves a little in place. Air-puff arousal shows these
bins are sleep-like: flies the rule scores asleep, by day too, respond to puffs
like sleeping flies (real minus sham response +3.2 percentage points in bins only
k = 3 calls asleep, against +5.2 awake and +1.2 asleep by every rule).

With ``untracked="break"`` the rule reproduces ``rule_sustained_k3`` of the
sleep-scoring analysis of the lab archive (sleep_rule.py, October 2026), with
``k = 2`` as the alternative. Choose it with ``sleep_annotation(rule="k")`` or
declare it once with ``set_sleep_rule("k")``; sleep_annotation has no default rule.
"""

import os
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

DOCS_URL = "https://github.com/gilestrolab/ethoscopy#choosing-a-sleep-rule"
SLEEP_RULE_ENV = "ETHOSCOPY_SLEEP_RULE"
RULE_REQUIRED_MESSAGE = f"""sleep_annotation now needs a sleep rule: 'classic' or 'k'.

rule='classic' is the 5-minute rule as before: any frame faster than the velocity
threshold counts as movement. On current ethoscope data, tracking noise and brief
twitches break sleep into fragments under it.

rule='k' (k=3 by default) ignores flickers and isolated micro-movements, and counts
movement only when it is sustained or walking. It matches video ground truth at night,
and flies it scores asleep respond to air puffs like sleeping flies.

Use 'classic' to reproduce earlier analyses, and 'k' for new ones. Declare it once:

    etho.set_sleep_rule("classic")     # or "k", "k3", "k2": once, at the top
    # or, without touching the code:  export {SLEEP_RULE_ENV}=classic

or per call, e.g. when loading:

    from functools import partial
    data = etho.load_ethoscope(meta, FUN=partial(etho.sleep_annotation, rule="k"))

See {DOCS_URL}"""

_declared = {"rule": None, "k": None}


def _parse_rule(value: str) -> Tuple[str, Optional[int]]:
    """
    Split a rule name such as "classic", "k" or "k3" into the rule and its k.

    Args:
        value (str): The rule name.

    Returns:
        Tuple[str, Optional[int]]: ("classic" or "k", k or None).

    Raises:
        ValueError: If the name is not "classic", "k" or "k" followed by a positive integer.
    """
    name = str(value).strip().lower()
    if name in ("classic", "k"):
        return name, None
    if name[:1] == "k" and name[1:].isdigit() and int(name[1:]) >= 1:
        return "k", int(name[1:])
    raise ValueError(f'unknown sleep rule {value!r}: use "classic", "k" or e.g. "k3"')


def set_sleep_rule(rule: Optional[str]) -> None:
    """
    Declare the sleep rule once, for every later sleep_annotation call.

    Works like matplotlib's rcParams: a value passed to sleep_annotation still wins.
    Without a declaration, the environment variable ETHOSCOPY_SLEEP_RULE is used,
    so existing notebooks can be re-run unchanged.

    Args:
        rule (str or None): "classic", "k", or "k" with its k (e.g. "k3", "k2").
            None clears the declaration.

    Raises:
        ValueError: If the rule name is not recognised.
    """
    if rule is None:
        _declared.update(rule=None, k=None)
        return
    name, k = _parse_rule(rule)
    _declared.update(rule=name, k=k)


def get_sleep_rule() -> Optional[str]:
    """
    The sleep rule in force without an explicit argument.

    Returns:
        Optional[str]: "classic", "k" or e.g. "k3"; None when nothing is declared.
    """
    if _declared["rule"] is not None:
        return _declared["rule"] + (str(_declared["k"]) if _declared["k"] else "")
    return os.environ.get(SLEEP_RULE_ENV) or None


def resolve_rule(rule: Optional[str], k: Optional[int]) -> Tuple[str, int]:
    """
    Decide the rule and k for one call: argument, then declaration, then environment.

    Args:
        rule (str or None): The rule passed to sleep_annotation ("k3" style allowed).
        k (int or None): The k passed to sleep_annotation.

    Returns:
        Tuple[str, int]: The rule and k to use (k is 3 unless set).

    Raises:
        ValueError: If no rule is given anywhere (with an explanation), or it is unknown.
    """
    source = rule if rule is not None else get_sleep_rule()
    if source is None:
        raise ValueError(RULE_REQUIRED_MESSAGE)
    name, k_from_name = _parse_rule(source)
    return name, k if k is not None else (k_from_name or 3)


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
    untracked: str = "immobile",
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
        untracked (str, optional): "immobile" counts bins without frames as still and
            measures the step after them from the last position seen; "break" never
            scores them as sleep and counts the bin after them as walking (the
            reference rule). Default is "immobile".

    Returns:
        pd.DataFrame: One row per 10-s bin from the first to the last bin with
            frames: 't' (bin start, s), 'has_data', 'walking', 'sustained' (events
            starting in the bin), 'micro_awake' and 'asleep'.

    Raises:
        ValueError: If untracked is not "immobile" or "break".
    """
    if untracked not in ("immobile", "break"):
        raise ValueError('untracked must be "immobile" or "break"')
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
    if untracked == "immobile":
        # Reason: compare with the last position seen, so a gap hides no walking it bridges.
        grid = grid.ffill()
    step = np.hypot(
        np.diff(grid["x"].to_numpy(), prepend=np.nan),
        np.diff(grid["y"].to_numpy(), prepend=np.nan),
    )
    # Reason: an unknown step (the first bin; with "break", a bin next to a gap) is walking.
    still = has_data & (np.nan_to_num(step, nan=np.inf) <= WALK_SHIFT_PIXELS * pixel)
    if untracked == "immobile":
        still |= ~has_data

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
    untracked: str = "immobile",
) -> Optional[pd.DataFrame]:
    """
    Sleep annotation with the k-rule; the body of sleep_annotation(rule="k").

    The classic detector still runs, on every frame as rule="classic" does, so
    its columns ('moving', 'max_velocity', ...) are reported unchanged next to
    the k-rule's; only the k-rule ignores inferred frames.

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
        untracked (str, optional): How bins without frames are scored; see
            k_rule_bins(). Default is "immobile".

    Returns:
        Optional[pd.DataFrame]: The classic columns plus 'walking', 'sustained' and
            'micro_awake', with 'asleep' from the k-rule and 'is_interpolated' True
            for bins without frames; None for fewer than 100 observed frames.
    """
    frames = _tracked_frames(data)
    if len(frames.index) < 100:
        return None
    binned = motion_detector_function(
        data,
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
        untracked=untracked,
    )
    # Reason: the classic detector drops sparse windows, so its bins are a subset of the rule's.
    out = rule[["t"]].merge(binned, how="left", on="t")
    # Reason: as in the classic path, a bin the detector did not score is not moving.
    out["moving"] = np.where(out["moving"].isna(), False, out["moving"]).astype(bool)
    out["is_interpolated"] = ~rule["has_data"].to_numpy()
    for column in ("walking", "sustained", "micro_awake", "asleep"):
        out[column] = rule[column].to_numpy()
    return out
