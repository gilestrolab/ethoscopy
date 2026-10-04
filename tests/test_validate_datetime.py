"""
Regression tests for ``validate_datetime`` and its callers.

The lab convention is dd/mm/yyyy. ``validate_datetime`` used to parse the
whole date column with ``pd.to_datetime`` first, which reads '05/01/2024' as
1 May; it only fell back to day-first formats when some day in the column was
above 12. A metadata file whose days were all <= 12 was therefore silently
read month-first, and ``link_meta_index`` then looked for the wrong runs.
"""

import datetime as dt
import ftplib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ethoscopy.load import download_from_remote_dir, link_meta_index
from ethoscopy.misc.validate_datetime import validate_datetime


def _dates(values):
    return validate_datetime(pd.DataFrame({"date": values}))["date"].tolist()


class TestValidateDatetime:
    """dd/mm/yyyy is enforced; other unambiguous forms still work."""

    def test_day_first_when_all_days_are_small(self):
        assert _dates(["05/01/2024", "06/01/2024"]) == ["2024-01-05", "2024-01-06"]

    def test_day_first_mixed_column(self):
        assert _dates(["05/01/2024", "25/03/2024"]) == ["2024-01-05", "2024-03-25"]

    @pytest.mark.parametrize(
        "value",
        [
            "05/01/2024",
            "5/1/2024",
            "05-01-2024",
            "05.01.2024",
            "05/01/24",
            " 05/01/2024 ",
        ],
    )
    def test_day_first_separators_and_years(self, value):
        assert _dates([value]) == ["2024-01-05"]

    @pytest.mark.parametrize(
        "value",
        [
            "2024-01-05",
            "2024-01-05 00:00:00",
            dt.datetime(2024, 1, 5, 9),
            pd.Timestamp("2024-01-05"),
        ],
    )
    def test_iso_and_datetime_values(self, value):
        assert _dates([value]) == ["2024-01-05"]

    @pytest.mark.parametrize("value", ["01/25/2024", "31/02/2024", "not a date"])
    def test_invalid_dates_raise_with_row(self, value):
        with pytest.raises(ValueError, match="Incorrect date format in row 2"):
            _dates(["2024-01-05", value])

    def test_input_not_modified(self):
        data = pd.DataFrame({"date": ["05/01/2024"], "region_id": [1]})
        result = validate_datetime(data)
        assert data["date"].tolist() == ["05/01/2024"]
        assert result["region_id"].tolist() == [1]


def test_link_meta_index_reads_dd_mm(tmp_path):
    """A dd/mm metadata file finds the run started on that day, not month-first."""
    results = tmp_path / "results"
    for stamp in ("2025-01-05_10-00-00", "2025-05-01_10-00-00"):
        run_dir = results / "abc" / "ETHOSCOPE_001" / stamp
        run_dir.mkdir(parents=True)
        (run_dir / f"{stamp}_abc.db").write_text("dummy db content")
    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {"machine_name": ["ETHOSCOPE_001"], "date": ["05/01/2025"], "region_id": [1]}
    ).to_csv(csv_path, index=False)

    result = link_meta_index(str(csv_path), str(results))

    assert result["date"].tolist() == ["2025-01-05"]
    assert "2025-01-05_10-00-00" in result["path"].iloc[0]


class _FakeFTP:
    """Anonymous FTP server over an in-memory tree of dicts (files are bytes)."""

    def __init__(self, tree):
        self.tree = tree
        self.node = tree

    def __call__(self, netloc):  # stands in for the ftplib.FTP class
        return _FakeFTP(self.tree)

    def login(self):
        pass

    def quit(self):
        pass

    def cwd(self, path):
        node = self.tree
        for part in [p for p in str(path).split("/") if p]:
            if not isinstance(node, dict) or part not in node:
                raise ftplib.error_perm(f"550 {path}")
            node = node[part]
        self.node = node

    def nlst(self):
        return list(self.node)

    def size(self, name):
        return len(self.node[name])

    def retrbinary(self, command, callback):
        callback(self.node[command.split(" ", 1)[1]])


def test_download_from_remote_dir_reads_dd_mm(tmp_path, monkeypatch):
    """The FTP download used to discard validate_datetime's result and search
    the server with the raw '05/01/2025' string, finding nothing."""
    runs = {
        stamp: {f"{stamp}_abc.db": b"data"}
        for stamp in ("2025-01-05_10-00-00", "2025-05-01_10-00-00")
    }
    tree = {"results": {"abc": {"ETHOSCOPE_001": runs}}}
    monkeypatch.setattr("ethoscopy.load.ftplib.FTP", _FakeFTP(tree))
    monkeypatch.chdir(tmp_path)  # download_database changes directory
    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame({"machine_name": ["ETHOSCOPE_001"], "date": ["05/01/2025"]}).to_csv(
        csv_path, index=False
    )
    local = tmp_path / "local"
    local.mkdir()

    download_from_remote_dir(
        str(csv_path), "ftp://server/results", str(local), progress=False
    )

    downloaded = sorted(p.name for p in local.rglob("*.db"))
    assert downloaded == ["2025-01-05_10-00-00_abc.db"]
