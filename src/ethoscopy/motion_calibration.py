"""
Tracking-noise diagnostics for movement scoring.

The classic motion detector calls a 10-s bin "moving" when its peak frame-to-frame
velocity exceeds a threshold. When a motionless animal's tracking noise sits near
that threshold, spurious "movements" fragment every immobility bout, and sleep is
unusually sensitive to it: a 5-minute bout needs 30 consecutive immobile bins, so a
per-bin false-positive rate p leaves only (1 - p)^30 of true rests intact.

motion_qc() reports, per animal and light phase, how often a still animal crosses
the fixed threshold. Still bins are found *from position alone* - the position over
the half-minute either side of the bin stays within one pixel - so the check does
not depend on the velocity it checks. Where the fixed threshold fails, score sleep
with sleep_annotation(rule="k"), which ignores tracking noise (ethoscopy.sleep_rules).
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Positions in pixels are integers of order hundreds; positions normalised to the
# ROI width lie in [0, 1]. Anything whose maximum is below this is treated as
# normalised.
_NORMALISED_MAX = 1.5
# One pixel as a fraction of ROI width (0.002 of a ~500 px ROI is 1 px).
_PIXEL_NORMALISED = 2e-3
# Positional tolerance for a still bin, in pixels. Validated against
# tracker-independent pixel motion on two recordings (a 1-h video under the new
# double-band IR and 47 h under a dim IR): at 1 px the calibrated threshold
# matched the one chosen from the pixels (median ratio 1.00-1.02) and still bins
# were scored as moving 0.9-1.4% of the time. At 0.5 px the selection skips still
# flies whose tracked position jitters, and the threshold comes out too low.
STILL_SHIFT_PIXELS = 1.0


def pixel_size(x: np.ndarray) -> float:
    """
    Return one pixel in the units of the positions.

    load_ethoscope returns positions in pixels, while the legacy reader and some
    pickled datasets hold them as a fraction of ROI width, so the unit is
    inferred from the range of x.

    Args:
        x (np.ndarray): x positions of one animal.

    Returns:
        float: One pixel expressed in the units of ``x``.
    """
    finite = x[np.isfinite(x)]
    if len(finite) and np.max(finite) <= _NORMALISED_MAX:
        return _PIXEL_NORMALISED
    return 1.0


def default_still_shift(x: np.ndarray) -> float:
    """
    Pick the positional tolerance for a still bin in the units of the data.

    Args:
        x (np.ndarray): x positions of one animal.

    Returns:
        float: STILL_SHIFT_PIXELS (1 px) expressed in the units of ``x``.
    """
    return STILL_SHIFT_PIXELS * pixel_size(x)


def find_spikes(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    max_frames: int = 2,
    min_jump: Optional[float] = None,
    tolerance: Optional[float] = None,
    max_span: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find tracking spikes: the centroid jumps away and lands back where it was.

    A spike is a run of up to ``max_frames`` frames displaced by more than
    ``min_jump`` from the preceding frame, followed by a frame back within
    ``tolerance`` of it, all within ``max_span`` seconds. This happens when the
    tracker briefly locks onto something other than the fly. A real movement
    does not return to the same pixel, so these frames carry no behaviour, yet
    each one produces a large "velocity" going out and coming back.

    Args:
        t (np.ndarray): Frame times in seconds, sorted.
        x (np.ndarray): x positions.
        y (np.ndarray): y positions.
        max_frames (int, optional): Longest displaced run treated as a spike. Default is 2.
        min_jump (float, optional): Smallest displacement treated as a jump, in
            the units of x/y. None uses 3 pixels.
        tolerance (float, optional): How close the return must be. None uses
            1 pixel.
        max_span (float, optional): Longest time from the last good frame to
            the return, in seconds. Default is 2.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Boolean masks over frames: velocity
            artefacts (the displaced frames and the return frame) and position
            artefacts (the displaced frames only).
    """
    px = pixel_size(np.asarray(x, dtype=float))
    min_jump = 3 * px if min_jump is None else min_jump
    tolerance = px if tolerance is None else tolerance
    t, x, y = (np.asarray(v, dtype=float) for v in (t, x, y))
    n = len(t)
    velocity_mask = np.zeros(n, dtype=bool)
    position_mask = np.zeros(n, dtype=bool)
    for m in range(1, max_frames + 1):
        if n < m + 2:
            break
        # anchor a = k - 1 (last good frame), displaced k .. k+m-1, return k+m
        a = np.arange(0, n - m - 1)
        displaced = np.ones(len(a), dtype=bool)
        for j in range(1, m + 1):
            displaced &= np.hypot(x[a + j] - x[a], y[a + j] - y[a]) > min_jump
        back = np.hypot(x[a + m + 1] - x[a], y[a + m + 1] - y[a]) <= tolerance
        quick = (t[a + m + 1] - t[a]) <= max_span
        hit = a[displaced & back & quick]
        for j in range(1, m + 1):
            velocity_mask[hit + j] = True
            position_mask[hit + j] = True
        velocity_mask[hit + m + 1] = True
    return velocity_mask, position_mask


def _row_quantile(values: np.ndarray, q: float) -> np.ndarray:
    """
    Quantile of each row, ignoring NaN, for short rows.

    Same definition as numpy's default ("linear") and R's type 7, but vectorised:
    np.nanquantile along an axis falls back to a per-row loop, which is ~100x
    slower on the (n, 7) windows used here.

    Args:
        values (np.ndarray): 2-D array; quantiles are taken along axis 1.
        q (float): Quantile in [0, 1].

    Returns:
        np.ndarray: One value per row, NaN for rows without finite values.
    """
    ordered = np.sort(values, axis=1)  # NaN sorts last
    count = np.isfinite(values).sum(axis=1)
    pos = np.maximum(count - 1, 0) * q
    lower = np.floor(pos).astype(int)
    upper = np.ceil(pos).astype(int)
    rows = np.arange(len(values))
    low_val, high_val = ordered[rows, lower], ordered[rows, upper]
    result = low_val + (high_val - low_val) * (pos - lower)
    return np.where(count > 0, result, np.nan)


def find_still_bins(
    binned: pd.DataFrame,
    time_window_length: int = 10,
    context_bins: int = 3,
    max_shift: Optional[float] = None,
) -> np.ndarray:
    """
    Flag bins in which the animal did not change position.

    Two conditions over a window of ``context_bins`` bins either side must hold,
    both within ``max_shift``:

    - the median position before the bin and the median position after it
      agree (the animal did not end up somewhere else), and
    - positions across the whole window agree once the single highest and
      lowest bin are set aside (it did not wander and come back, e.g. pacing in
      the tube).

    A tracking glitch - a centroid that jumps for a frame, or even corrupts one
    bin, and returns - leaves both untouched.

    Args:
        binned (pd.DataFrame): One animal's binned data with columns 't', 'x', 'y'
            (the output of max_velocity_detector).
        time_window_length (int, optional): Bin size in seconds. Default is 10.
        context_bins (int, optional): Bins on each side used for the median
            positions. Default is 3 (30 s).
        max_shift (float, optional): Largest shift still counted as still, in the
            units of x/y. None picks one pixel via default_still_shift().

    Returns:
        np.ndarray: Boolean array aligned with ``binned`` rows. Bins lacking
            enough tracked context on either side are False.
    """
    if max_shift is None:
        max_shift = default_still_shift(binned["x"].to_numpy(dtype=float))

    pos = binned.set_index("t")[["x", "y"]]
    grid = range(int(pos.index.min()), int(pos.index.max()) + 1, time_window_length)
    # Reason: reindex to a regular grid so windows count bins, not rows, across gaps.
    pos = pos.reindex(grid)
    width = 2 * context_bins + 1
    still = np.zeros(len(pos), dtype=bool)
    if len(pos) >= width:
        min_side = max(1, context_bins - 1)
        ok = np.ones(len(pos) - width + 1, dtype=bool)
        shift_sq = np.zeros(len(ok))
        for axis in ("x", "y"):
            win = np.lib.stride_tricks.sliding_window_view(
                pos[axis].to_numpy(dtype=float), width
            )
            side_before = win[:, :context_bins]
            side_after = win[:, context_bins + 1 :]
            ok &= (np.isfinite(side_before).sum(axis=1) >= min_side) & (
                np.isfinite(side_after).sum(axis=1) >= min_side
            )
            shift_sq += (
                _row_quantile(side_after, 0.5) - _row_quantile(side_before, 0.5)
            ) ** 2
            # Reason: trimming one bin at each end forgives a single glitched bin,
            # while back-and-forth movement spreads over several bins and is kept.
            spread = _row_quantile(win, 5 / 6) - _row_quantile(win, 1 / 6)
            ok &= spread <= max_shift
        # NaN comparisons above are False, so windows inside tracking gaps are rejected
        ok &= np.sqrt(shift_sq) <= max_shift
        still[context_bins : context_bins + len(ok)] = ok
    return (
        pd.Series(still, index=pos.index)
        .reindex(binned["t"].to_numpy())
        .fillna(False)
        .to_numpy(dtype=bool)
    )


def light_phase(
    t: np.ndarray, day_length: int = 24, lights_off: int = 12
) -> np.ndarray:
    """
    Label timestamps as 'light' or 'dark', matching behavpy.add_day_phase().

    Args:
        t (np.ndarray): Time in seconds, with 0 at lights on.
        day_length (int, optional): Length of the day in hours. Default is 24.
        lights_off (int, optional): Hour of lights off. Default is 12.

    Returns:
        np.ndarray: Array of 'light' / 'dark' strings.
    """
    in_day = np.mod(t, day_length * 3600)
    return np.where(in_day < lights_off * 3600, "light", "dark")


def motion_qc(
    data: pd.DataFrame,
    time_window_length: int = 10,
    velocity_correction_coef: float = 3e-3,
    velocity_threshold: float = 1.0,
    min_sleep_duration: int = 300,
    day_length: int = 24,
    lights_off: int = 12,
) -> pd.DataFrame:
    """
    Report, per animal and light phase, how well the fixed threshold fits the noise.

    Run this on raw tracking data before scoring sleep. It shows, for a still
    animal, how often the fixed threshold is crossed ('fp_rate_fixed') and what
    fraction of genuine 5-minute rests would survive that ('rest_survival_fixed',
    (1 - fp)^bins). Values of fp_rate_fixed above about 0.01 mean classic sleep
    estimates are unreliable; score sleep with sleep_annotation(rule="k") instead.
    A large 'untracked_fraction' matters under either rule: with the default
    untracked="immobile", bins where the animal was lost count as still.

    Args:
        data (pd.DataFrame): Raw tracking data for one or more animals, with 'id'
            as index or column; t must be 0 at lights on for the phase split.
        time_window_length (int, optional): Bin size in seconds. Default is 10.
        velocity_correction_coef (float, optional): As in max_velocity_detector.
            Default is 3e-3.
        velocity_threshold (float, optional): The fixed threshold being checked.
            Default is 1.0.
        min_sleep_duration (int, optional): Sleep criterion in seconds, used for
            rest_survival_fixed. Default is 300.
        day_length (int, optional): Length of the day in hours. Default is 24.
        lights_off (int, optional): Hour of lights off. Default is 12.

    Returns:
        pd.DataFrame: One row per animal and phase with 'n_bins',
            'untracked_fraction', 'spike_fraction' (frames that are tracking
            spikes, see find_spikes()), 'n_still', 'fp_rate_fixed' and
            'rest_survival_fixed'.
    """
    # Reason: analyse imports this module, so importing it at module level would be circular.
    from ethoscopy.analyse import max_velocity_detector

    frame = data.reset_index() if "id" not in data.columns else data
    bins_per_rest = min_sleep_duration / time_window_length
    rows = []
    for animal, group in frame.groupby("id", sort=True):
        binned = max_velocity_detector(
            group.drop(columns="id"),
            time_window_length=time_window_length,
            velocity_correction_coef=velocity_correction_coef,
            velocity_threshold=velocity_threshold,
            walk_threshold=max(2.5, velocity_threshold * 1.01),
        )
        if binned is None:
            continue
        frames = group.sort_values("t")
        spikes, _ = find_spikes(
            frames["t"].to_numpy(), frames["x"].to_numpy(), frames["y"].to_numpy()
        )
        frame_phase = light_phase(
            frames["t"].to_numpy(dtype=float), day_length, lights_off
        )
        velocity = binned["max_velocity"].to_numpy(dtype=float)
        still = find_still_bins(binned, time_window_length)
        t = binned["t"].to_numpy()
        phase = light_phase(t.astype(float), day_length, lights_off)
        grid = np.arange(t.min(), t.max() + time_window_length, time_window_length)
        grid_phase = light_phase(grid.astype(float), day_length, lights_off)
        for name in ("light", "dark"):
            in_phase = phase == name
            if not in_phase.any():
                continue
            mask = still & in_phase
            fp = (
                float(np.mean(velocity[mask] > velocity_threshold))
                if mask.any()
                else np.nan
            )
            expected = int((grid_phase == name).sum())
            rows.append(
                {
                    "id": animal,
                    "phase": name,
                    "n_bins": int(in_phase.sum()),
                    "untracked_fraction": (
                        1 - in_phase.sum() / expected if expected else np.nan
                    ),
                    "spike_fraction": float(spikes[frame_phase == name].mean()),
                    "n_still": int(mask.sum()),
                    "fp_rate_fixed": fp,
                    "rest_survival_fixed": (1 - fp) ** bins_per_rest,
                }
            )
    return pd.DataFrame(rows)
