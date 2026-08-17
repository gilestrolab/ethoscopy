"""
Unit tests for load_ethoscope_diagnostics.

Real SQLite files are built rather than mocked: the behaviour under test is
mostly about which tables a given firmware wrote, which only shows up when a
statement actually runs.
"""

import sqlite3

import pandas as pd
import pytest

from ethoscopy.load import _one_row_per_database, load_ethoscope_diagnostics

# 10:23:58 UTC on 2026-08-13, the start of a real recording
START_EPOCH = 1786616638.6997

DIAGNOSTIC_ROWS = [
    # t (ms), fps, image_noise, sharpness, jitter, n_rois_sampled, cpu_temp, frame_noise
    (1377, None, 0.0, 31.26, None, 0, 54.8, 0.644),
    (61556, 4.05, 1.48, 30.74, 0.00225, 16, 60.1, 0.637),
    (3661556, 3.67, 1.37, 31.60, 0.00230, 17, 61.2, 0.640),
]


def _make_db(path, with_diagnostics=True, rows=DIAGNOSTIC_ROWS):
    """Build a minimal ethoscope database, optionally carrying DIAGNOSTICS."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE METADATA (field TEXT, value TEXT)")
    conn.execute(
        "INSERT INTO METADATA VALUES ('date_time', ?)", (str(START_EPOCH),)
    )

    if with_diagnostics:
        conn.execute(
            "CREATE TABLE DIAGNOSTICS (t INTEGER, fps REAL, image_noise REAL, "
            "sharpness REAL, jitter REAL, n_rois_sampled INTEGER, cpu_temp REAL, "
            "frame_noise REAL)"
        )
        conn.executemany(
            "INSERT INTO DIAGNOSTICS VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
        )

    conn.commit()
    conn.close()
    return path


@pytest.fixture
def metadata(tmp_path):
    """Per-ROI metadata for two machines, only the first of which has diagnostics."""
    equipped = _make_db(tmp_path / "equipped.db", with_diagnostics=True)
    bare = _make_db(tmp_path / "bare.db", with_diagnostics=False)

    rows = []
    for region_id in (1, 2):
        rows.append(
            {
                "id": f"2026-08-13_10-23-58_aaaaaa|{region_id:02d}",
                "machine_id": "aaaaaa",
                "machine_name": "ETHOSCOPE_017",
                "date": "2026-08-13",
                "region_id": region_id,
                "path": str(equipped),
            }
        )
        rows.append(
            {
                "id": f"2026-08-13_10-23-28_bbbbbb|{region_id:02d}",
                "machine_id": "bbbbbb",
                "machine_name": "ETHOSCOPE_001",
                "date": "2026-08-13",
                "region_id": region_id,
                "path": str(bare),
            }
        )

    return pd.DataFrame(rows)


class TestLoadEthoscopeDiagnostics:
    """Expected use, edge cases and failure modes."""

    def test_reads_diagnostics_and_labels_machine(self, metadata):
        """Expected use: every sample is returned, tagged with its machine."""
        with pytest.warns(RuntimeWarning, match="ETHOSCOPE_001"):
            result = load_ethoscope_diagnostics(metadata, progress=False)

        assert len(result) == len(DIAGNOSTIC_ROWS)
        assert set(result["machine_name"]) == {"ETHOSCOPE_017"}
        assert result["machine_id"].iloc[0] == "aaaaaa"
        assert result["date"].iloc[0] == "2026-08-13"
        for column in ("fps", "image_noise", "sharpness", "jitter", "cpu_temp"):
            assert column in result.columns

    def test_database_read_once_per_recording(self, metadata):
        """Two ROIs of one machine must not duplicate its device-level rows."""
        collapsed = _one_row_per_database(metadata)
        assert len(collapsed) == 2  # one row per machine, not per ROI

    def test_time_converted_to_seconds(self, metadata):
        """t arrives in milliseconds and must be handed back in seconds."""
        with pytest.warns(RuntimeWarning):
            result = load_ethoscope_diagnostics(metadata, progress=False)

        assert result["t"].iloc[0] == pytest.approx(1377 / 1000)
        assert result["t"].iloc[-1] == pytest.approx(3661556 / 1000)

    def test_reference_hour_offsets_time(self, metadata):
        """A reference hour re-zeroes t to that wall-clock hour."""
        with pytest.warns(RuntimeWarning):
            result = load_ethoscope_diagnostics(
                metadata, reference_hour=9, progress=False
            )

        # recording starts 10:23:58 UTC, so 1:23:58 = 5038 s after ZT0
        offset = (10 + 23 / 60 + 58 / 3600 - 9) * 3600
        assert result["t"].iloc[0] == pytest.approx(1377 / 1000 + offset)

    def test_time_window_filters_samples(self, metadata):
        """min_time/max_time are hours, matching load_ethoscope."""
        with pytest.warns(RuntimeWarning):
            result = load_ethoscope_diagnostics(metadata, max_time=1, progress=False)

        assert len(result) == 2  # the 3661556 ms (61 min) sample is excluded

    def test_missing_table_warns_but_returns_the_rest(self, metadata):
        """Edge case: mixed firmwares must not abort the whole load."""
        with pytest.warns(RuntimeWarning, match="does not record diagnostics"):
            result = load_ethoscope_diagnostics(metadata, progress=False)

        assert not result.empty

    def test_no_diagnostics_anywhere_returns_empty(self, metadata):
        """Edge case: every database predates diagnostics."""
        bare_only = metadata[metadata["machine_name"] == "ETHOSCOPE_001"]

        with pytest.warns(RuntimeWarning):
            result = load_ethoscope_diagnostics(bare_only, progress=False)

        assert result.empty

    def test_empty_metadata_returns_empty(self):
        """Edge case: nothing linked."""
        assert load_ethoscope_diagnostics(pd.DataFrame(), progress=False).empty

    def test_min_time_greater_than_max_time_raises(self, metadata):
        """Failure case: an impossible window is a caller error."""
        with pytest.raises(ValueError, match="min_time is larger than max_time"):
            load_ethoscope_diagnostics(metadata, min_time=10, max_time=1)

    def test_unreadable_database_is_skipped(self, metadata, tmp_path):
        """Failure case: one broken file must not lose the others."""
        metadata = metadata.copy()
        metadata.loc[metadata["machine_name"] == "ETHOSCOPE_001", "path"] = str(
            tmp_path / "does_not_exist.db"
        )

        result = load_ethoscope_diagnostics(metadata, progress=False)
        assert set(result["machine_name"]) == {"ETHOSCOPE_017"}


class TestCachePath:
    """The cache filename must encode everything that changes the cached frame."""

    FILE = {"machine_id": "aaaaaa", "region_id": 3, "date": "2026-08-13"}

    def test_defaults_keep_the_historic_filename(self, tmp_path):
        """Caches written by earlier versions must stay valid."""
        from ethoscopy.load import _cache_path

        path = _cache_path(str(tmp_path), self.FILE, 0, float("inf"), None)
        assert path.name == "cached_aaaaaa_3_2026-08-13.pkl"

    @pytest.mark.parametrize(
        "min_time,max_time,reference_hour",
        [
            (0, float("inf"), 9),
            (3600, float("inf"), None),
            (0, 86400, None),
        ],
    )
    def test_non_default_arguments_get_their_own_file(
        self, tmp_path, min_time, max_time, reference_hour
    ):
        """Failure case guarded: a different window must not reuse the default cache."""
        from ethoscopy.load import _cache_path

        default = _cache_path(str(tmp_path), self.FILE, 0, float("inf"), None)
        other = _cache_path(str(tmp_path), self.FILE, min_time, max_time, reference_hour)
        assert other != default

    def test_different_reference_hours_do_not_collide(self, tmp_path):
        """The bug this guards: two reference hours sharing one cached frame."""
        from ethoscopy.load import _cache_path

        assert _cache_path(str(tmp_path), self.FILE, 0, float("inf"), 9) != _cache_path(
            str(tmp_path), self.FILE, 0, float("inf"), 12
        )
