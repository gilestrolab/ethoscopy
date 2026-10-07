"""
load_ethoscope with an ROI that holds no rows.

SQLite returns an empty table with every column as object dtype. Concatenated
with the other ROIs, it turned their integer columns (x, y, w, h, phi,
xy_dist_log10x1000) into object, and numeric code downstream failed with
"unorderable types for comparison" (motion_qc on five-day recordings).
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

import ethoscopy as etho
from ethoscopy.load import load_ethoscope

INTEGER_COLUMNS = ["x", "y", "w", "h", "phi", "xy_dist_log10x1000"]


def _make_db(path, rows_per_roi):
    """
    Build an ethoscope-like database: SMALLINT columns, one ROI table per entry.

    Args:
        path (Path): Database file to create.
        rows_per_roi (dict): {roi number: rows to write}; 0 leaves the table empty.

    Returns:
        str: The database path.
    """
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE ROI_MAP (roi_idx INT, roi_value INT, x INT, y INT, w INT, h INT)"
    )
    con.execute(
        "CREATE TABLE VAR_MAP (var_name TEXT, sql_type TEXT, functional_type TEXT)"
    )
    con.execute("CREATE TABLE METADATA (field TEXT, value TEXT)")
    con.execute("INSERT INTO METADATA VALUES ('date_time', '1790958651')")
    con.executemany(
        "INSERT INTO VAR_MAP VALUES (?, 'SMALLINT', ?)",
        [
            ("x", "distance"),
            ("y", "distance"),
            ("w", "distance"),
            ("h", "distance"),
            ("phi", "angle"),
            ("xy_dist_log10x1000", "relative_distance_1e6"),
        ],
    )
    rng = np.random.default_rng(0)
    for roi, n in rows_per_roi.items():
        con.execute("INSERT INTO ROI_MAP VALUES (?, ?, 0, 0, 550, 50)", (roi, roi))
        con.execute(
            f"CREATE TABLE ROI_{roi} (id INTEGER PRIMARY KEY, t INTEGER, x SMALLINT, "
            "y SMALLINT, w SMALLINT, h SMALLINT, phi SMALLINT, xy_dist_log10x1000 SMALLINT, "
            "is_inferred BOOLEAN, has_interacted SMALLINT)"
        )
        con.executemany(
            f"INSERT INTO ROI_{roi} (t, x, y, w, h, phi, xy_dist_log10x1000, is_inferred, "
            "has_interacted) VALUES (?, ?, 25, 24, 10, 0, ?, 0, 0)",
            [(i * 500, 100 + int(rng.integers(0, 3)), -3000) for i in range(n)],
        )
    con.commit()
    con.close()
    return str(path)


def _meta(db, rois):
    return pd.DataFrame(
        {
            "path": db,
            "machine_name": "ETHOSCOPE_X",
            "region_id": rois,
            "id": [f"fly_{r}" for r in rois],
        }
    )


@pytest.mark.unit
class TestEmptyRoi:
    """An ROI without rows must not change the other ROIs' columns."""

    def test_integer_columns_stay_numeric(self, tmp_path):
        db = _make_db(tmp_path / "x.db", {1: 300, 2: 0, 3: 300})
        data = load_ethoscope(_meta(db, [1, 2, 3]), progress=False, verbose=False)
        for column in INTEGER_COLUMNS:
            assert pd.api.types.is_numeric_dtype(data[column]), column
        assert sorted(data.id.unique()) == ["fly_1", "fly_3"]

    def test_same_as_loading_only_the_tracked_rois(self, tmp_path):
        db = _make_db(tmp_path / "x.db", {1: 300, 2: 0, 3: 300})
        with_empty = load_ethoscope(_meta(db, [1, 2, 3]), progress=False, verbose=False)
        without = load_ethoscope(_meta(db, [1, 3]), progress=False, verbose=False)
        pd.testing.assert_frame_equal(with_empty, without)

    def test_empty_roi_is_reported(self, tmp_path, capsys):
        db = _make_db(tmp_path / "x.db", {1: 300, 2: 0})
        load_ethoscope(_meta(db, [1, 2]), progress=False, verbose=True)
        assert "ROI_2" in capsys.readouterr().out

    def test_only_empty_rois(self, tmp_path):
        db = _make_db(tmp_path / "x.db", {1: 0, 2: 0})
        assert load_ethoscope(_meta(db, [1, 2]), progress=False, verbose=False).empty

    def test_motion_qc_runs_on_the_whole_load(self, tmp_path):
        db = _make_db(tmp_path / "x.db", {1: 3000, 2: 0})
        data = load_ethoscope(_meta(db, [1, 2]), progress=False, verbose=False)
        if not hasattr(etho, "motion_qc"):
            pytest.skip("motion_qc is not in this version")
        assert len(etho.motion_qc(data)) > 0
