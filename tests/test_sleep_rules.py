"""
Tests for the k-rule (ethoscopy.sleep_rules and sleep_annotation(rule="k")).

The per-bin parity tests use frames and expected values exported from the
reference implementation (sleep_rule.py) by scripts/validate_k_rule.py.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ethoscopy.sleep_rules as etho_rules
from ethoscopy.analyse import sleep_annotation
from ethoscopy.sleep_rules import (
    FLICKER,
    SUBPIXEL,
    SUSTAINED,
    classify_events,
    k_rule_bins,
)

DATA = Path(__file__).parent / "data"
STILL, MOVE = -3000, -2000  # xy_dist_log10x1000 for velocity ~0.33 and ~3.3


def _classes(x, moving, y=None):
    """Classify events on a 1-D track (y constant unless given)."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x) if y is None else np.asarray(y, dtype=float)
    starts, classes = classify_events(x, y, np.asarray(moving, dtype=bool))
    return starts.tolist(), classes.tolist()


class TestClassifyEvents:
    """Subpixel, flicker and sustained events."""

    def test_subpixel_flicker_and_sustained(self):
        x = [5, 5, 5, 9, 5, 5, 8, 9, 5, 5, 7, 8, 9, 5]
        moving = [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
        starts, classes = _classes(x, moving)
        assert starts == [1, 3, 6, 10]
        # 1: never left the pixel; 3: one frame out and back; 6: two frames out and
        # back; 10: three frames out and back is too long for a flicker.
        assert classes == [SUBPIXEL, FLICKER, FLICKER, SUSTAINED]

    def test_return_tolerance_is_inclusive(self):
        assert _classes([0, 4, 1], [0, 1, 0])[1] == [FLICKER]
        assert _classes([0, 4, 1.01], [0, 1, 0])[1] == [SUSTAINED]
        # Diagonal return of exactly one pixel
        x, y = [0, 4, 0.6], [0, 4, 0.8]
        assert _classes(x, [0, 1, 0], y=y)[1] == [FLICKER]

    def test_run_on_first_frame_is_not_an_event(self):
        assert _classes([5, 9, 5, 9, 5], [1, 1, 0, 1, 0]) == ([3], [FLICKER])

    def test_run_on_last_frame_cannot_flicker(self):
        assert _classes([5, 5, 9], [0, 0, 1]) == ([2], [SUSTAINED])
        # ...but can still be subpixel
        assert _classes([5, 5, 5], [0, 0, 1]) == ([2], [SUBPIXEL])

    def test_subpixel_needs_both_coordinates(self):
        assert _classes([5, 5, 5], [0, 1, 0], y=[3, 4, 3])[1] == [FLICKER]

    def test_no_events(self):
        starts, classes = classify_events(
            np.arange(5.0), np.zeros(5), np.zeros(5, dtype=bool)
        )
        assert len(starts) == 0 and len(classes) == 0


def _bins(x, n_per_bin=10, d=None, shift=None, k=3, y=None, t0=0.0, **kwargs):
    """
    k_rule_bins on a track given one x position per 10-s bin.

    Args:
        x (list): Position for each bin (np.nan for a bin without frames).
        n_per_bin (int, optional): Frames per bin (1 fps). Default is 10.
        d (dict, optional): {frame index: xy_dist value} for moving frames.
        shift (dict, optional): {frame index: x} overriding single frames.
        k (int, optional): The rule's k. Default is 3.
        y (list, optional): y position per bin; zeros if None.
        t0 (float, optional): Start time in seconds. Default is 0.

    Returns:
        pd.DataFrame: The per-bin result.
    """
    x = np.repeat(np.asarray(x, dtype=float), n_per_bin)
    y = np.zeros_like(x) if y is None else np.repeat(np.asarray(y, float), n_per_bin)
    t = t0 + np.arange(len(x), dtype=float)
    xy = np.full(len(x), STILL)
    for i, value in (d or {}).items():
        xy[i] = value
    for i, value in (shift or {}).items():
        x[i] = value
    keep = ~np.isnan(x)
    return k_rule_bins(t[keep], x[keep], y[keep], xy[keep], k=k, **kwargs)


def _sustained_at(bins_with_event, n_per_bin=10):
    """
    One sustained event per listed bin: a moving frame that leaves its pixel and
    does not come back on the next frame (the bin median stays put).

    Returns:
        dict: Keyword arguments d and shift for _bins().
    """
    frames = [b * n_per_bin + 3 for b in bins_with_event]
    return {
        "d": {i: MOVE for i in frames},
        "shift": {j: 130.0 for i in frames for j in (i, i + 1)},
    }


class TestKRuleBins:
    """Walking, the event window and sleep bouts."""

    def test_still_track_sleeps_after_first_bin(self):
        out = _bins([100.0] * 40)
        assert out.has_data.all()
        # The first bin has no step and counts as walking.
        assert out.walking.tolist() == [True] + [False] * 39
        assert out.asleep.tolist() == [False] + [True] * 39

    def test_walking_threshold_is_inclusive(self):
        out = _bins([100, 110, 120.5, 120.5])
        assert out.walking.tolist() == [True, False, True, False]

    def test_bouts_of_29_and_30_bins(self):
        assert not _bins([100.0] * 30).asleep.any()  # 29 still bins after the first
        assert _bins([100.0] * 31).asleep.sum() == 30

    def test_bin_without_frames_break(self):
        x = [100.0] * 35 + [np.nan] + [100.0] * 35
        out = _bins(x, untracked="break")
        assert not out.has_data[35] and not out.asleep[35]
        assert out.walking[36]  # no previous median to compare with
        assert out.asleep[1:35].all() and out.asleep[37:].all()
        assert not out.asleep[36]

    def test_bin_without_frames_immobile(self):
        x = [100.0] * 35 + [np.nan] * 5 + [100.0] * 35
        out = _bins(x)  # the default
        # The gap counts as still and the fly is found where it was lost.
        assert not out.has_data[35:40].any() and not out.walking[35:41].any()
        assert out.asleep[1:].all()

    def test_fly_found_elsewhere_after_a_gap_walked(self):
        x = [100.0] * 35 + [np.nan] * 5 + [130.0] * 35
        out = _bins(x)
        # The step after the gap is measured from the last position seen (30 px).
        assert out.walking[40] and not out.walking[35:40].any()
        assert out.asleep[1:40].all() and not out.asleep[40] and out.asleep[41:].all()

    def test_invalid_untracked(self):
        with pytest.raises(ValueError):
            _bins([100.0] * 5, untracked="nope")

    def test_sustained_event_window_and_k(self):
        x = np.full(60, 100.0)
        events = _sustained_at([20, 21, 22])
        out = _bins(x, k=3, **events)
        # Three events in bins 20-22: bins whose window [i-3, i+3) holds all three
        assert out.sustained.sum() == 3
        assert out.micro_awake[out.micro_awake].index.tolist() == [20, 21, 22, 23]
        two = _bins(x, k=2, **events)
        assert two.micro_awake[two.micro_awake].index.tolist() == list(range(19, 25))

    def test_window_is_clipped_at_both_ends(self):
        x = np.full(40, 100.0)
        out = _bins(x, k=1, **_sustained_at([0, 39]))
        awake = out.micro_awake[out.micro_awake].index.tolist()
        # An event in bin e wakes bins e-2 .. e+3, clipped to the grid.
        assert awake == [0, 1, 2, 3, 37, 38, 39]

    def test_flicker_and_subpixel_events_do_not_wake(self):
        x = np.repeat(np.full(40, 100.0), 10)
        t = np.arange(len(x), dtype=float)
        xy = np.full(len(x), STILL)
        # A 3-frame subpixel run, a flicker and a 1-frame subpixel run
        xy[[50, 51, 52, 150, 250]] = MOVE
        x_flick = x.copy()
        x_flick[150] = 140.0  # out and straight back: flicker
        out = k_rule_bins(t, x_flick, np.zeros_like(x), xy, k=1)
        assert out.sustained.sum() == 0 and not out.micro_awake.any()

    def test_pixel_scales_the_limits(self):
        x_px = [100, 109, 125, 125, 125]  # steps of 9 and 16 px
        in_px = _bins(x_px)
        assert in_px.walking.tolist() == [True, False, True, False, False]
        scaled = _bins(np.asarray(x_px) / 551.0, pixel=1 / 551.0)
        pd.testing.assert_series_equal(in_px.walking, scaled.walking)
        assert not _bins(np.asarray(x_px) / 551.0).walking[2]  # pixel left at 1

    def test_no_frames(self):
        out = k_rule_bins(np.array([]), np.array([]), np.array([]), np.array([]))
        assert out.empty


def _raw(n_bins=120, n_per_bin=10, inferred=None):
    """
    Raw frames of a still animal, at 1 fps, in the columns the detector needs.

    sleep_annotation needs at least 100 bins, so the default is 120.
    """
    n = n_bins * n_per_bin
    raw = pd.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "x": np.full(n, 100.0),
            "y": np.full(n, 20.0),
            "w": np.full(n, 25.0),
            "h": np.full(n, 10.0),
            "phi": np.zeros(n),
            "xy_dist_log10x1000": np.full(n, STILL),
            "has_interacted": np.zeros(n, dtype=int),
        }
    )
    if inferred is not None:
        raw["is_inferred"] = inferred
    return raw


class TestSleepAnnotationKRule:
    """sleep_annotation(rule="k")."""

    def test_adds_columns_and_keeps_classic_ones(self):
        raw = _raw(inferred=np.zeros(1200, dtype=int))
        # Inferred frames repeating a movement: classic counts them, the k-rule drops them.
        raw.loc[500:504, ["xy_dist_log10x1000", "is_inferred"]] = [MOVE, 1]
        classic = sleep_annotation(raw.copy(), rule="classic")
        k = sleep_annotation(raw.copy(), rule="k")
        assert {"walking", "sustained", "micro_awake"} <= set(k.columns)
        assert not {"walking", "sustained", "micro_awake"} & set(classic.columns)
        shared = [c for c in classic.columns if c != "asleep"]
        pd.testing.assert_frame_equal(
            k[shared].reset_index(drop=True), classic[shared].reset_index(drop=True)
        )
        assert classic.moving[50] and k.moving[50]
        assert k.asleep.tolist() == [False] + [True] * 119

    def test_inferred_frames_are_dropped(self):
        n = 1600
        inferred = np.where((np.arange(n) >= 800) & (np.arange(n) < 820), "1", "0")
        raw = _raw(n_bins=160, inferred=inferred)
        # Bins 80 and 81 lose all their frames.
        out = sleep_annotation(raw.copy(), rule="k")
        assert out.is_interpolated[80:82].all() and not out.is_interpolated[82:].any()
        assert out.asleep[1:].all()  # untracked="immobile", the default
        out = sleep_annotation(raw.copy(), rule="k", untracked="break")
        # Never sleep, and bin 82 has no step to compare with.
        assert out.walking[82] and not out.asleep[80:83].any()
        assert out.asleep[1:80].all() and out.asleep[83:].all()

    def test_null_inferred_is_dropped(self):
        inferred = pd.Series([0] * 1200, dtype=object)
        inferred[300:320] = None
        out = sleep_annotation(_raw(inferred=inferred), rule="k")
        assert out.is_interpolated[30:32].all()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rule": "nope"},
            {"rule": "k", "velocity_threshold": 1.0},
            {"rule": "k", "velocity_threshold": "auto"},
            {"rule": "k", "time_window_length": 20},
            {"rule": "k", "k": 0},
            {"rule": "k", "k": True},
            {"rule": "k", "k": 2.5},
            {"rule": "k", "pixel": 0},
            {"rule": "k", "untracked": "nope"},
        ],
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            sleep_annotation(_raw(), **kwargs)

    def test_too_few_frames(self):
        assert sleep_annotation(_raw(n_bins=5), rule="k") is None


@pytest.fixture(scope="module")
def parity():
    """Frames and per-bin results exported from the reference sleep_rule.py."""
    frames = pd.read_csv(DATA / "k_rule_frames.csv")
    expected = pd.read_csv(DATA / "k_rule_bins.csv")
    return frames, expected


class TestParity:
    """Bin-by-bin agreement with the reference on real recordings."""

    def test_fixtures_cover_the_cases(self, parity):
        frames, expected = parity
        assert frames.fly.nunique() >= 3
        assert (frames.is_inferred != 0).any()
        assert (~expected.has_data).any()  # a bin without frames
        assert expected.asleep_k3.any() and not expected.asleep_k3.all()
        # Flies lost while asleep, where the two untracked policies differ
        assert (expected.asleep_k3 != expected.asleep_k3_immobile).sum() > 100

    @pytest.mark.parametrize("k", [3, 2])
    @pytest.mark.parametrize("untracked", ["break", "immobile"])
    def test_bins_match_reference(self, parity, k, untracked):
        # The reference is sleep_rule.py for "break", ethoscopy itself for "immobile".
        frames, expected = parity
        column = f"asleep_k{k}" + ("_immobile" if untracked == "immobile" else "")
        for fly, raw in frames.groupby("fly"):
            # Reason: the classic detector needs these columns; they do not enter the k-rule.
            raw = raw.drop(columns="fly").assign(
                t=raw.t / 1000.0, w=25.0, h=10.0, phi=0.0, has_interacted=0
            )
            out = sleep_annotation(raw, rule="k", k=k, untracked=untracked)
            ref = expected[expected.fly == fly].reset_index(drop=True)
            assert out.t.tolist() == ref.t.tolist(), fly
            assert (~out.is_interpolated).tolist() == ref.has_data.tolist(), fly
            if untracked == "break":
                assert out.walking.tolist() == ref.walking.tolist(), fly
            assert out.asleep.tolist() == ref[column].tolist(), fly


class TestRuleRequired:
    """rule has no built-in default since 3.0; it can be declared once."""

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.delenv("ETHOSCOPY_SLEEP_RULE", raising=False)
        etho_rules.set_sleep_rule(None)
        yield
        etho_rules.set_sleep_rule(None)

    def test_missing_rule_raises_and_explains(self):
        with pytest.raises(ValueError) as err:
            sleep_annotation(_raw())
        message = str(err.value)
        assert "rule='classic'" in message and "rule='k'" in message
        assert "set_sleep_rule" in message and "ETHOSCOPY_SLEEP_RULE" in message
        assert "partial(etho.sleep_annotation" in message

    def test_declared_rule_applies(self):
        etho_rules.set_sleep_rule("k")
        assert "walking" in sleep_annotation(_raw()).columns
        etho_rules.set_sleep_rule("classic")
        assert "walking" not in sleep_annotation(_raw()).columns

    def test_argument_beats_declaration(self):
        etho_rules.set_sleep_rule("classic")
        assert "walking" in sleep_annotation(_raw(), rule="k").columns

    def test_environment_variable(self, monkeypatch):
        monkeypatch.setenv("ETHOSCOPY_SLEEP_RULE", "k2")
        assert etho_rules.get_sleep_rule() == "k2"
        assert "walking" in sleep_annotation(_raw()).columns

    def test_rule_names_carry_k(self):
        raw = _raw(n_bins=200)
        # Three sustained events in bins 100-102 wake bins under k=3 and k=2 differently
        for b in (100, 101, 102):
            raw.loc[b * 10 + 3, "xy_dist_log10x1000"] = MOVE
            raw.loc[[b * 10 + 3, b * 10 + 4], "x"] = 130.0
        k2 = sleep_annotation(raw.copy(), rule="k2")
        assert (
            k2.micro_awake.sum()
            == sleep_annotation(raw.copy(), rule="k", k=2).micro_awake.sum()
        )
        assert (
            k2.micro_awake.sum()
            > sleep_annotation(raw.copy(), rule="k3").micro_awake.sum()
        )
        with pytest.raises(ValueError):
            etho_rules.set_sleep_rule("k0")

    def test_load_ethoscope_partial_pattern(self, tmp_path):
        from functools import partial

        from ethoscopy.load import load_ethoscope

        db = tmp_path / "x.db"
        con = sqlite3.connect(str(db))
        con.execute(
            "CREATE TABLE ROI_MAP (roi_idx INT, roi_value INT, x INT, y INT, w INT, h INT)"
        )
        con.execute("INSERT INTO ROI_MAP VALUES (1, 1, 0, 0, 550, 50)")
        con.execute(
            "CREATE TABLE VAR_MAP (var_name TEXT, sql_type TEXT, functional_type TEXT)"
        )
        con.execute("CREATE TABLE METADATA (field TEXT, value TEXT)")
        con.execute("INSERT INTO METADATA VALUES ('date_time', '1790958651')")
        con.execute(
            "CREATE TABLE ROI_1 (id INTEGER PRIMARY KEY, t INTEGER, x SMALLINT, y SMALLINT, "
            "w SMALLINT, h SMALLINT, phi SMALLINT, xy_dist_log10x1000 SMALLINT, "
            "is_inferred BOOLEAN, has_interacted SMALLINT)"
        )
        con.executemany(
            "INSERT INTO ROI_1 (t, x, y, w, h, phi, xy_dist_log10x1000, is_inferred, "
            "has_interacted) VALUES (?, 100, 20, 25, 10, 0, -3000, 0, 0)",
            [(i * 1000,) for i in range(1500)],
        )
        con.commit()
        con.close()
        meta = pd.DataFrame(
            {"path": [str(db)], "machine_name": ["M"], "region_id": [1], "id": ["fly"]}
        )
        data = load_ethoscope(
            meta, FUN=partial(sleep_annotation, rule="k"), progress=False, verbose=False
        )
        assert "micro_awake" in data.columns and data.asleep.any()
