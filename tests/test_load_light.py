"""
Unit tests for the light-schedule readers.

Real SQLite files with real JPEG snapshots are built rather than mocked: what is
being tested is whether a given firmware recorded anything usable, and what can
be measured back out of the images when it did not.
"""

import calendar
import io
import sqlite3

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ethoscopy.load import (
    _circular_mean_hour,
    _hours_from_clock,
    estimate_light_cycle,
    load_ethoscope_light_schedule,
)

# 2026-08-13 06:00:00 UTC - a start time three hours before lights-on, so the
# first snapshot lands in the dark phase and the first transition is a sunrise.
START_EPOCH = calendar.timegm((2026, 8, 13, 6, 0, 0, 0, 0, 0))

LIGHTS_ON_HOUR = 9.0
LIGHTS_OFF_HOUR = 21.0

SNAPSHOT_INTERVAL_S = 300
RECORDING_HOURS = 72

DARK_LEVEL = 30
LIGHT_LEVEL = 200


def _jpeg(level):
    """Encode a uniform grey frame as JPEG bytes."""
    image = Image.fromarray(np.full((48, 64), level, dtype=np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=50)
    return buffer.getvalue()


def _make_db(
    path,
    experimental_info=None,
    with_snapshots=True,
    constant_light=False,
    corrupt_every=0,
):
    """
    Build a minimal ethoscope database.

    Args:
        path: Destination file
        experimental_info (dict, optional): Written verbatim to METADATA; None
            omits the field entirely, as pre-schedule firmwares do
        with_snapshots (bool): Whether to write an IMG_SNAPSHOTS table
        constant_light (bool): Hold the lights on, so there is no cycle to find
        corrupt_every (int): Write a truncated JPEG every nth snapshot
    """
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE METADATA (field TEXT, value TEXT)")
    conn.execute("INSERT INTO METADATA VALUES ('date_time', ?)", (str(START_EPOCH),))
    if experimental_info is not None:
        conn.execute(
            "INSERT INTO METADATA VALUES ('experimental_info', ?)",
            (repr(experimental_info),),
        )

    if with_snapshots:
        conn.execute("CREATE TABLE IMG_SNAPSHOTS (id int, t int, img longblob)")
        rows = []
        n = int(RECORDING_HOURS * 3600 / SNAPSHOT_INTERVAL_S)
        for i in range(n):
            t_s = i * SNAPSHOT_INTERVAL_S
            hour = ((START_EPOCH + t_s) / 3600) % 24
            lit = constant_light or (LIGHTS_ON_HOUR <= hour < LIGHTS_OFF_HOUR)
            blob = _jpeg(LIGHT_LEVEL if lit else DARK_LEVEL)
            if corrupt_every and i % corrupt_every == 0:
                blob = blob[: len(blob) // 3]  # truncated beyond recovery
            rows.append((i, t_s * 1000, blob))
        conn.executemany("INSERT INTO IMG_SNAPSHOTS VALUES (?, ?, ?)", rows)

    conn.commit()
    conn.close()
    return path


def _metadata(paths):
    """Build a per-ROI metadata frame pointing at the given databases."""
    rows = []
    for index, (name, path) in enumerate(paths.items()):
        for region_id in (1, 2):
            rows.append(
                {
                    "id": f"2026-08-13_06-00-00_{index:06d}|{region_id:02d}",
                    "machine_id": f"machine{index}",
                    "machine_name": name,
                    "date": "2026-08-13",
                    "region_id": region_id,
                    "path": str(path),
                }
            )
    return pd.DataFrame(rows)


class TestHoursFromClock:
    """Parsing the recorded HH:MM strings, which are usually unset."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("09:00", 9.0),
            ("21:30", 21.5),
            ("00:00", 0.0),
            ("9:05", 9 + 5 / 60),
            ("23:59", 23 + 59 / 60),
        ],
    )
    def test_valid_times(self, value, expected):
        """Expected use."""
        assert _hours_from_clock(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", ["", "   ", None, np.nan, "nonsense", "25:00", 9])
    def test_unset_or_malformed_returns_nan(self, value):
        """Edge case: the field is empty far more often than it is filled."""
        assert np.isnan(_hours_from_clock(value))


class TestCircularMeanHour:
    """Wall-clock hours have to be averaged as angles."""

    def test_simple_mean(self):
        """Expected use."""
        assert _circular_mean_hour([9.0, 9.2, 8.8]) == pytest.approx(9.0, abs=1e-6)

    def test_wraps_around_midnight(self):
        """The failure a plain mean would produce: 23:50 and 00:10 -> midday."""
        assert _circular_mean_hour([23 + 50 / 60, 10 / 60]) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_empty_is_nan(self):
        """Edge case."""
        assert np.isnan(_circular_mean_hour([]))
        assert np.isnan(_circular_mean_hour([np.nan]))


class TestLoadEthoscopeLightSchedule:
    """Reading the schedule the device recorded, when it recorded one."""

    def test_recorded_schedule_is_reduced_to_usable_terms(self, tmp_path):
        """Expected use: lights_on becomes reference_hour, period becomes day length."""
        db = _make_db(
            tmp_path / "driven.db",
            experimental_info={
                "lights_on": "09:00",
                "lights_off": "21:00",
                "light_period_minutes": 1440,
                "light_cycle_anchor": "",
                "fade_in_seconds": 5,
                "fade_out_seconds": 5,
                "max_light": 80,
                "crepuscular": 1,
            },
            with_snapshots=False,
        )

        result = load_ethoscope_light_schedule(
            _metadata({"ETHOSCOPE_001": db}), progress=False
        )

        assert len(result) == 1
        row = result.iloc[0]
        assert row["source"] == "recorded"
        assert row["reference_hour"] == pytest.approx(9.0)
        assert row["day_length_h"] == pytest.approx(24.0)
        assert row["lights_off_h"] == pytest.approx(12.0)
        assert row["photoperiod_h"] == pytest.approx(12.0)
        assert row["max_light"] == 80
        assert row["crepuscular"] == 1

    def test_anchor_outranks_lights_on(self, tmp_path):
        """light_cycle_anchor is what the daemon actually counted ZT0 from."""
        anchor = calendar.timegm((2026, 8, 13, 7, 30, 0, 0, 0, 0))
        db = _make_db(
            tmp_path / "anchored.db",
            experimental_info={
                "lights_on": "09:00",
                "lights_off": "21:00",
                "light_period_minutes": 1440,
                "light_cycle_anchor": str(anchor),
            },
            with_snapshots=False,
        )

        result = load_ethoscope_light_schedule(
            _metadata({"ETHOSCOPE_001": db}), progress=False
        )

        assert result.iloc[0]["source"] == "anchor"
        assert result.iloc[0]["reference_hour"] == pytest.approx(7.5)

    def test_non_24h_cycle_is_carried_through(self, tmp_path):
        """T-cycles are supported by the firmware and must not be forced to 24 h."""
        db = _make_db(
            tmp_path / "tcycle.db",
            experimental_info={
                "lights_on": "09:00",
                "lights_off": "19:00",
                "light_period_minutes": 1200,  # T = 20 h
                "light_cycle_anchor": "",
            },
            with_snapshots=False,
        )

        result = load_ethoscope_light_schedule(
            _metadata({"ETHOSCOPE_001": db}), progress=False
        )

        assert result.iloc[0]["day_length_h"] == pytest.approx(20.0)
        assert result.iloc[0]["lights_off_h"] == pytest.approx(10.0)

    def test_empty_fields_report_absent_and_warn(self, tmp_path):
        """The common case: an incubator drove the light, so nothing was recorded."""
        db = _make_db(
            tmp_path / "incubator.db",
            experimental_info={
                "lights_on": "",
                "lights_off": "",
                "light_period_minutes": 1440,
                "light_cycle_anchor": "",
            },
            with_snapshots=False,
        )

        with pytest.warns(RuntimeWarning, match="No light schedule recorded"):
            result = load_ethoscope_light_schedule(
                _metadata({"ETHOSCOPE_001": db}), progress=False
            )

        assert result.iloc[0]["source"] == "absent"
        assert np.isnan(result.iloc[0]["reference_hour"])
        # partial information is still worth keeping
        assert result.iloc[0]["day_length_h"] == pytest.approx(24.0)

    def test_missing_field_entirely(self, tmp_path):
        """Edge case: firmware predating the schedule fields."""
        db = _make_db(tmp_path / "old.db", experimental_info=None, with_snapshots=False)

        with pytest.warns(RuntimeWarning):
            result = load_ethoscope_light_schedule(
                _metadata({"ETHOSCOPE_001": db}), progress=False
            )

        assert result.iloc[0]["source"] == "absent"

    def test_malformed_info_does_not_abort_the_read(self, tmp_path):
        """Failure case: one unparseable blob must not cost the other machines."""
        good = _make_db(
            tmp_path / "good.db",
            experimental_info={
                "lights_on": "09:00",
                "lights_off": "21:00",
                "light_period_minutes": 1440,
                "light_cycle_anchor": "",
            },
            with_snapshots=False,
        )
        bad = tmp_path / "bad.db"
        conn = sqlite3.connect(str(bad))
        conn.execute("CREATE TABLE METADATA (field TEXT, value TEXT)")
        conn.execute("INSERT INTO METADATA VALUES ('date_time', ?)", (str(START_EPOCH),))
        conn.execute(
            "INSERT INTO METADATA VALUES ('experimental_info', '{not a dict')"
        )
        conn.commit()
        conn.close()

        with pytest.warns(RuntimeWarning):
            result = load_ethoscope_light_schedule(
                _metadata({"ETHOSCOPE_001": good, "ETHOSCOPE_002": bad}), progress=False
            )

        assert len(result) == 2
        assert set(result["source"]) == {"recorded", "absent"}

    def test_disagreeing_schedules_warn(self, tmp_path):
        """Pooling machines on different regimes is a silent analysis error."""
        first = _make_db(
            tmp_path / "a.db",
            experimental_info={
                "lights_on": "09:00", "lights_off": "21:00",
                "light_period_minutes": 1440, "light_cycle_anchor": "",
            },
            with_snapshots=False,
        )
        second = _make_db(
            tmp_path / "b.db",
            experimental_info={
                "lights_on": "12:00", "lights_off": "00:00",
                "light_period_minutes": 1440, "light_cycle_anchor": "",
            },
            with_snapshots=False,
        )

        with pytest.warns(RuntimeWarning, match="different light schedules"):
            load_ethoscope_light_schedule(
                _metadata({"ETHOSCOPE_001": first, "ETHOSCOPE_002": second}),
                progress=False,
            )

    def test_empty_metadata_returns_empty(self):
        """Edge case."""
        assert load_ethoscope_light_schedule(pd.DataFrame(), progress=False).empty

    def test_one_row_per_recording_not_per_roi(self, tmp_path):
        """Two ROIs of one machine must not duplicate its device-level row."""
        db = _make_db(
            tmp_path / "one.db",
            experimental_info={
                "lights_on": "09:00", "lights_off": "21:00",
                "light_period_minutes": 1440, "light_cycle_anchor": "",
            },
            with_snapshots=False,
        )
        result = load_ethoscope_light_schedule(
            _metadata({"ETHOSCOPE_001": db}), progress=False
        )
        assert len(result) == 1


class TestEstimateLightCycle:
    """Measuring the cycle back out of the stored snapshots."""

    def test_recovers_a_12_12_cycle(self, tmp_path):
        """Expected use, and the case that matters: nothing was recorded."""
        db = _make_db(tmp_path / "cycle.db", experimental_info=None)

        result = estimate_light_cycle(
            _metadata({"ETHOSCOPE_001": db}), progress=False
        )

        row = result.iloc[0]
        # snapshots land every 5 min, so the edge is placed to within that
        assert row["lights_on_utc"] == pytest.approx(LIGHTS_ON_HOUR, abs=0.1)
        assert row["lights_off_utc"] == pytest.approx(LIGHTS_OFF_HOUR, abs=0.1)
        assert row["photoperiod_h"] == pytest.approx(12.0, abs=0.1)
        assert row["reference_hour"] == row["lights_on_utc"]
        assert row["n_transitions"] >= 5
        assert row["contrast"] > 100

    def test_constant_light_reports_no_cycle(self, tmp_path):
        """Edge case: LL. A midpoint threshold on noise would invent a cycle."""
        db = _make_db(
            tmp_path / "ll.db", experimental_info=None, constant_light=True
        )

        result = estimate_light_cycle(
            _metadata({"ETHOSCOPE_001": db}), progress=False
        )

        assert np.isnan(result.iloc[0]["lights_on_utc"])
        assert result.iloc[0]["n_transitions"] == 0
        assert result.iloc[0]["contrast"] < 5.0

    def test_corrupt_snapshots_are_skipped(self, tmp_path):
        """Failure case: truncated JPEGs are common at the tail of a killed run."""
        db = _make_db(tmp_path / "corrupt.db", experimental_info=None, corrupt_every=7)

        result = estimate_light_cycle(
            _metadata({"ETHOSCOPE_001": db}), progress=False
        )

        assert result.iloc[0]["photoperiod_h"] == pytest.approx(12.0, abs=0.2)

    def test_missing_table_warns_and_is_skipped(self, tmp_path):
        """Failure case: mixed firmware, one without snapshots."""
        good = _make_db(tmp_path / "good.db", experimental_info=None)
        bare = _make_db(
            tmp_path / "bare.db", experimental_info=None, with_snapshots=False
        )

        with pytest.warns(RuntimeWarning, match="No usable snapshots"):
            result = estimate_light_cycle(
                _metadata({"ETHOSCOPE_001": good, "ETHOSCOPE_002": bare}),
                progress=False,
            )

        assert set(result["machine_name"]) == {"ETHOSCOPE_001"}

    def test_stride_does_not_move_the_answer(self, tmp_path):
        """Sampling fewer snapshots must cost precision, not correctness."""
        db = _make_db(tmp_path / "stride.db", experimental_info=None)
        meta = _metadata({"ETHOSCOPE_001": db})

        dense = estimate_light_cycle(meta, stride=1, progress=False)
        sparse = estimate_light_cycle(meta, stride=4, progress=False)

        assert sparse.iloc[0]["lights_on_utc"] == pytest.approx(
            dense.iloc[0]["lights_on_utc"], abs=0.4
        )

    def test_empty_metadata_returns_empty(self):
        """Edge case."""
        assert estimate_light_cycle(pd.DataFrame(), progress=False).empty

    def test_unreadable_database_is_skipped(self, tmp_path):
        """Failure case: one broken path must not lose the others."""
        good = _make_db(tmp_path / "good.db", experimental_info=None)
        meta = _metadata(
            {"ETHOSCOPE_001": good, "ETHOSCOPE_002": tmp_path / "missing.db"}
        )

        result = estimate_light_cycle(meta, progress=False)
        assert set(result["machine_name"]) == {"ETHOSCOPE_001"}
