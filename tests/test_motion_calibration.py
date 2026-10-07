"""
Tests for tracking-noise diagnostics (ethoscopy.motion_calibration).

Tracks are synthetic with known truth: a fly that rests at a fixed position and
walks during scheduled windows, with per-frame tracking noise of a chosen size.
A short 2-hour "day" (1 h light, 1 h dark) keeps the recordings small while
still exercising the per-phase estimate.
"""

import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest

from ethoscopy.analyse import max_velocity_detector, sleep_annotation
from ethoscopy.behavpy_core import behavpy_core
from ethoscopy.motion_calibration import (
    default_still_shift,
    find_spikes,
    find_still_bins,
    light_phase,
    motion_qc,
)

ROI_WIDTH = 508.0
COEF = 3e-3
DAY = dict(day_length=2, lights_off=1)


def make_track(
    hours=4,
    fps=2,
    noise_light=0.9,
    noise_dark=0.9,
    active_minutes=10,
    always_active=False,
    gap=None,
    spike_rate=0.0,
    seed=0,
):
    """
    Build raw tracking data for one fly.

    Args:
        hours (int): Recording length in hours.
        fps (int): Frames per second.
        noise_light (float): Typical noise velocity of a still fly in the light phase,
            in units of the detector threshold (1.0 = default threshold).
        noise_dark (float): Same for the dark phase.
        active_minutes (int): Minutes of walking at the start of every hour.
        always_active (bool): Walk for the whole recording.
        gap (tuple): (start_s, end_s) with no tracked frames.
        spike_rate (float): Fraction of resting frames where the tracker jumps
            40 px for one frame and lands back on the same pixel.
        seed (int): Random seed.

    Returns:
        tuple: (raw DataFrame, boolean array of active frames).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, hours * 3600, 1 / fps)
    active = always_active | ((t % 3600) < active_minutes * 60)
    x = 200 + np.where(active, 60 * np.sin(2 * np.pi * t / 20), 0.0)
    y = np.full_like(t, 30.0)
    phase = light_phase(t, **DAY)
    noise = np.where(phase == "light", noise_light, noise_dark)
    # Reason: a still fly's device distance is lognormal jitter around the noise level.
    jitter = noise * np.exp(rng.normal(0, 0.08, len(t))) * COEF
    spike = ~active & (rng.random(len(t)) < spike_rate)
    x = x + np.where(spike, 40.0, 0.0)
    step = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])) / ROI_WIDTH
    dist = step + jitter
    raw = pd.DataFrame(
        {
            "t": t,
            "x": np.round(x),
            "y": y,
            "w": 24.0,
            "h": 11.0,
            "phi": 0.0,
            "xy_dist_log10x1000": np.round(1000 * np.log10(dist)),
        }
    )
    if gap is not None:
        keep = (t < gap[0]) | (t >= gap[1])
        raw, active = raw[keep].reset_index(drop=True), active[keep]
    return raw, active


def rest_sleep(result, active_minutes=10, settle_minutes=6):
    """Fraction of rest bins scored asleep, skipping the minutes needed to qualify."""
    in_hour = result["t"] % 3600
    rest = in_hour >= (active_minutes + settle_minutes) * 60
    return result.loc[rest, "asleep"].mean()


class TestUntracked:
    """sleep_annotation(untracked=...) treatment of bins without frames."""

    @pytest.mark.unit
    def test_break_ends_sleep_at_untracked_bins(self):
        gap = (3600 + 30 * 60, 3600 + 40 * 60)
        raw, _ = make_track(noise_light=0.3, noise_dark=0.3, gap=gap)
        immobile = sleep_annotation(raw, rule="classic", untracked="immobile")
        broken = sleep_annotation(raw, rule="classic", untracked="break")
        in_gap = (immobile["t"] >= gap[0]) & (immobile["t"] < gap[1])
        assert immobile.loc[in_gap, "is_interpolated"].all()
        assert immobile.loc[in_gap, "asleep"].all()
        assert not broken.loc[in_gap, "asleep"].any()
        # movement columns are not altered by the option
        pd.testing.assert_series_equal(immobile["moving"], broken["moving"])

    @pytest.mark.unit
    def test_invalid_untracked_raises(self):
        raw, _ = make_track(hours=1)
        with pytest.raises(ValueError, match="untracked"):
            sleep_annotation(raw, rule="classic", untracked="skip")


class TestStillBins:
    """find_still_bins and unit handling."""

    @staticmethod
    def binned(x):
        return pd.DataFrame(
            {"t": np.arange(len(x)) * 10, "x": np.asarray(x, float), "y": 30.0}
        )

    @pytest.mark.unit
    def test_one_bin_glitch_does_not_break_stillness(self):
        x = np.full(40, 200.0)
        x[20] = 203.0
        still = find_still_bins(self.binned(x))
        assert still[20]
        assert still[5:35].all()

    @pytest.mark.unit
    def test_lasting_shift_is_not_still(self):
        x = np.r_[np.full(20, 200.0), np.full(20, 205.0)]
        still = find_still_bins(self.binned(x))
        assert not still[19] and not still[20]
        assert still[5] and still[35]

    @pytest.mark.unit
    def test_sub_pixel_jitter_is_still_at_default_tolerance(self):
        # Reason: tracked positions of a still fly wander by a fraction of a pixel;
        # excluding them (the old 0.5 px) biased the calibrated threshold low.
        jitter = 200.0 + np.tile([0.0, 0.4, 0.8, 0.3, 0.6], 8)
        assert find_still_bins(self.binned(jitter))[5:35].all()
        assert find_still_bins(self.binned(jitter), max_shift=0.5)[5:35].mean() < 0.8

    @pytest.mark.unit
    def test_pacing_back_and_forth_is_not_still(self):
        # Reason: bin means alternate, so before/after medians agree; the spread must catch it.
        still = find_still_bins(self.binned(np.tile([162.0, 238.0], 20)))
        assert not still.any()

    @pytest.mark.unit
    def test_edges_without_context_are_not_still(self):
        still = find_still_bins(self.binned(np.full(20, 200.0)))
        assert not still[0] and not still[-1]

    @pytest.mark.unit
    def test_default_shift_follows_units(self):
        assert default_still_shift(np.array([12.0, 480.0])) == 1.0
        assert default_still_shift(np.array([0.02, 0.95])) == 2e-3


class TestSpikes:
    """find_spikes and spike removal in the detector."""

    @staticmethod
    def frames(x):
        x = np.asarray(x, float)
        return np.arange(len(x)) * 0.25, x, np.full(len(x), 30.0)

    @pytest.mark.unit
    def test_one_and_two_frame_spikes_are_found(self):
        x = np.full(30, 200.0)
        x[10] = 240.0  # one frame out
        x[20:22] = 230.0  # two frames out
        velocity, position = find_spikes(*self.frames(x))
        assert np.flatnonzero(position).tolist() == [10, 20, 21]
        assert np.flatnonzero(velocity).tolist() == [10, 11, 20, 21, 22]

    @pytest.mark.unit
    def test_real_moves_and_small_jitter_are_not_spikes(self):
        step = np.r_[np.full(10, 200.0), np.full(10, 240.0)]  # moves and stays
        jitter = np.tile([200.0, 201.0], 10)  # 1 px out and back: noise, not a spike
        slow_return = np.r_[np.full(10, 200.0), 240.0, 240.0, 240.0, np.full(10, 200.0)]
        for x in (step, jitter, slow_return):
            velocity, _ = find_spikes(*self.frames(x))
            assert not velocity.any()

    @pytest.mark.unit
    def test_return_after_a_long_gap_is_not_a_spike(self):
        t = np.array([0, 0.25, 0.5, 10.0, 10.25])
        x = np.array([200.0, 200.0, 240.0, 200.0, 200.0])
        velocity, _ = find_spikes(t, x, np.full(5, 30.0))
        assert not velocity.any()

    @pytest.mark.unit
    def test_motion_qc_reports_spikes(self):
        raw, _ = make_track(spike_rate=0.004)
        qc = motion_qc(raw.assign(id="fly"), **DAY)
        assert (qc["spike_fraction"] > 0.002).all()


class TestMotionQC:
    """motion_qc report."""

    @pytest.mark.unit
    def test_flags_noisy_recording(self):
        noisy, _ = make_track(noise_light=0.9, noise_dark=0.9)
        clean, _ = make_track(noise_light=0.3, noise_dark=0.3, seed=1)
        data = pd.concat([noisy.assign(id="noisy"), clean.assign(id="clean")])
        qc = motion_qc(data, **DAY).set_index(["id", "phase"])
        assert len(qc) == 4
        assert (qc.loc["noisy", "fp_rate_fixed"] > 0.5).all()
        assert (qc.loc["noisy", "rest_survival_fixed"] < 0.01).all()
        assert (qc.loc["clean", "fp_rate_fixed"] == 0).all()
        assert "auto_threshold" not in qc.columns


class TestBehavpyMethods:
    """behavpy motion_detector / sleep_contiguous pass the new options through."""

    @pytest.mark.unit
    def test_sleep_contiguous_break(self):
        raw, _ = make_track(noise_light=0.3, noise_dark=0.3, gap=(5400, 5700))
        data = raw.assign(id="fly_1").set_index("id")
        meta = pd.DataFrame({"id": ["fly_1"], "group": ["a"]}).set_index("id")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            df = behavpy_core(data, meta, check=True)
            moved = df.motion_detector()
            kept = moved.sleep_contiguous()
            broken = moved.sleep_contiguous(untracked="break")
        assert broken["asleep"].sum() < kept["asleep"].sum()
        with pytest.raises(ValueError, match="untracked"):
            moved.sleep_contiguous(untracked="skip")
