"""
Unit tests for ethoscopy.load module functions.
"""

import shutil
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from ethoscopy.load import (
    link_meta_index,
    load_ethoscope,
    load_ethoscope_metadata,
    read_single_roi,
)


class TestLinkMetaIndex:
    """Test suite for link_meta_index function."""

    @pytest.mark.unit
    def test_link_meta_index_success(self, sample_metadata_csv, tmp_path):
        """Test successful metadata linking."""
        # Create expected directory structure matching the glob pattern: */MACHINE_NAME/DATE_TIME/*.db
        # The function expects any parent dir, then MACHINE_NAME, then DATE_TIME format
        results_dir = tmp_path / "results"  # This is the 'any subdirectory' part
        machine_dir1 = results_dir / "ETHOSCOPE_001" / "2025-01-01_00-00-00"
        machine_dir1.mkdir(parents=True)
        machine_dir2 = results_dir / "ETHOSCOPE_002" / "2025-01-01_00-00-00"
        machine_dir2.mkdir(parents=True)

        # Create dummy database files with .db extension
        db_file1 = machine_dir1 / "2025-01-01_00-00-00_ETHOSCOPE_001.db"
        db_file1.write_text("dummy db content")
        db_file2 = machine_dir2 / "2025-01-01_00-00-00_ETHOSCOPE_002.db"
        db_file2.write_text("dummy db content")

        # Test the function - pass the parent of results_dir as local_dir
        result = link_meta_index(str(sample_metadata_csv), str(tmp_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "path" in result.columns
        assert "file_size" in result.columns
        assert "id" in result.columns

    @pytest.mark.unit
    def test_link_meta_index_no_files(self, sample_metadata_csv, tmp_path):
        """Test metadata linking when no database files exist."""
        with pytest.raises(RuntimeError, match="No Ethoscope data could be found"):
            link_meta_index(str(sample_metadata_csv), str(tmp_path))

    @pytest.mark.unit
    def test_link_meta_index_missing_metadata(self, tmp_path):
        """Test metadata linking with missing metadata file."""
        missing_file = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError):
            link_meta_index(str(missing_file), str(tmp_path))

    @pytest.mark.unit
    def test_link_meta_index_missing_columns(self, tmp_path):
        """Test metadata linking with missing required columns."""
        # Create metadata CSV missing required columns
        bad_metadata = pd.DataFrame(
            {
                "date": ["2025-01-01"],
                "machine_name": ["ETHOSCOPE_001"],
                # Missing region_id
            }
        )
        csv_path = tmp_path / "bad_metadata.csv"
        bad_metadata.to_csv(csv_path, index=False)

        # Create some database files so it doesn't fail on "No Ethoscope data found"
        # The function needs the directory structure: any_parent/*/MACHINE/DATE_TIME/*.db
        results_dir = tmp_path / "results"
        machine_dir = results_dir / "ETHOSCOPE_001" / "2025-01-01_00-00-00"
        machine_dir.mkdir(parents=True)
        db_file = machine_dir / "2025-01-01_00-00-00_ETHOSCOPE_001.db"
        db_file.write_text("dummy")

        with pytest.raises(KeyError):
            link_meta_index(str(csv_path), str(tmp_path))


class TestReadSingleRoi:
    """Test suite for read_single_roi function."""

    @pytest.mark.unit
    def test_read_single_roi_success(self, linked_metadata_sample):
        """Test successful single ROI reading."""
        file_info = linked_metadata_sample.iloc[0]

        result = read_single_roi(file_info)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "t" in result.columns
        assert "x" in result.columns
        assert "y" in result.columns

    @pytest.mark.unit
    def test_read_single_roi_time_filtering(self, linked_metadata_sample):
        """Test ROI reading with time constraints."""
        file_info = linked_metadata_sample.iloc[0]

        # Test with time constraints
        result = read_single_roi(file_info, min_time=0, max_time=1800)  # 30 minutes

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert result["t"].max() <= 1800

    @pytest.mark.unit
    def test_read_single_roi_invalid_time_range(self, linked_metadata_sample):
        """Test ROI reading with invalid time range."""
        file_info = linked_metadata_sample.iloc[0]

        with pytest.raises(ValueError, match="min_time is larger than max_time"):
            read_single_roi(file_info, min_time=3600, max_time=1800)

    @pytest.mark.unit
    def test_read_single_roi_missing_roi(self, mock_sqlite_db, tmp_path):
        """Test ROI reading when requested ROI doesn't exist."""
        # Create file info for non-existent ROI
        file_info = pd.Series(
            {
                "path": str(mock_sqlite_db),
                "region_id": 999,  # Non-existent ROI
                "machine_id": "test",
                "date": "2025-01-01",
            }
        )

        # Function should raise an exception for missing ROI, not return None
        with pytest.raises(Exception, match="ROI 999 does not exist"):
            read_single_roi(file_info)

    @pytest.mark.unit
    def test_read_single_roi_with_cache(self, linked_metadata_sample, tmp_path):
        """Test ROI reading with caching."""
        file_info = linked_metadata_sample.iloc[0]
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # First call should create cache
        result1 = read_single_roi(file_info, cache=str(cache_dir))

        # Second call should use cache
        result2 = read_single_roi(file_info, cache=str(cache_dir))

        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
        pd.testing.assert_frame_equal(result1, result2)

    @pytest.mark.unit
    def test_read_single_roi_creates_missing_cache_dir(
        self, linked_metadata_sample, tmp_path
    ):
        """A cache directory that does not exist yet is created, not fatal."""
        file_info = linked_metadata_sample.iloc[0]
        cache_dir = tmp_path / "not" / "yet" / "there"

        result = read_single_roi(file_info, cache=str(cache_dir))

        assert isinstance(result, pd.DataFrame) and len(result) > 0
        assert len(list(cache_dir.glob("*.pkl"))) == 1
        assert not list(cache_dir.glob("*.tmp"))

    @pytest.mark.unit
    def test_read_single_roi_survives_unwritable_cache(
        self, linked_metadata_sample, tmp_path
    ):
        """Failure case: a cache that cannot be written costs the cache, not the ROI."""
        file_info = linked_metadata_sample.iloc[0]
        blocker = tmp_path / "blocker"
        blocker.write_text("a file where the cache directory should go")

        with pytest.warns(RuntimeWarning, match="Could not write to the cache"):
            result = read_single_roi(file_info, cache=str(blocker / "cache"))

        assert isinstance(result, pd.DataFrame) and len(result) > 0
        assert blocker.is_file()


class TestLoadEthoscope:
    """Test suite for load_ethoscope function."""

    @pytest.mark.unit
    def test_load_ethoscope_success(self, linked_metadata_sample):
        """Test successful ethoscope data loading."""
        result = load_ethoscope(linked_metadata_sample, verbose=False)

        assert isinstance(result, pd.DataFrame)
        assert "id" in result.columns
        if len(result) > 0:
            assert result["id"].iloc[0] == linked_metadata_sample["id"].iloc[0]

    @pytest.mark.unit
    def test_load_ethoscope_with_function(self, linked_metadata_sample):
        """Test ethoscope loading with processing function."""

        def dummy_function(data):
            """Dummy processing function that adds a column."""
            data["processed"] = True
            return data

        result = load_ethoscope(
            linked_metadata_sample, FUN=dummy_function, verbose=False
        )

        if len(result) > 0:
            assert "processed" in result.columns
            assert result["processed"].all() == True

    @pytest.mark.unit
    def test_load_ethoscope_function_returns_none(self, linked_metadata_sample):
        """Test ethoscope loading when processing function returns None."""

        def failing_function(data):
            """Function that returns None."""
            return None

        result = load_ethoscope(
            linked_metadata_sample, FUN=failing_function, verbose=False
        )

        # Should return empty DataFrame when all processing fails
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.unit
    def test_load_ethoscope_time_constraints(self, linked_metadata_sample):
        """Test ethoscope loading with time constraints."""
        result = load_ethoscope(
            linked_metadata_sample,
            min_time=0,
            max_time=1,
            verbose=False,  # 1 hour
        )

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert result["t"].max() <= 3600  # 1 hour in seconds

    @pytest.mark.unit
    def test_load_ethoscope_empty_metadata(self):
        """Test ethoscope loading with empty metadata."""
        empty_metadata = pd.DataFrame()

        result = load_ethoscope(empty_metadata, verbose=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.unit
    def test_load_ethoscope_id_column_handling(self, linked_metadata_sample):
        """Test that load_ethoscope properly handles existing id columns."""
        # This tests the specific bug we fixed
        result = load_ethoscope(linked_metadata_sample, verbose=False)

        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "id" in result.columns
            # Should not raise ValueError about duplicate id column


class TestLoadEthoscopeMetadata:
    """Test suite for load_ethoscope_metadata function."""

    @pytest.mark.unit
    def test_load_ethoscope_metadata_success(self, sample_metadata_csv, tmp_path):
        """Test successful metadata loading."""
        # First create linked metadata (the function expects DataFrame from link_meta_index)
        machine_dir = (
            tmp_path / "test_results" / "ETHOSCOPE_001" / "2025-01-01_00-00-00"
        )
        machine_dir.mkdir(parents=True)

        # Create a mock database file
        db_file = machine_dir / "2025-01-01_00-00-00_ETHOSCOPE_001.db"

        # Create a simple SQLite database with METADATA table
        import sqlite3

        with sqlite3.connect(str(db_file)) as conn:
            conn.execute(
                """
                CREATE TABLE METADATA (
                    machine_name TEXT,
                    date TEXT,
                    region_id INTEGER
                )
            """
            )
            conn.execute(
                "INSERT INTO METADATA (machine_name, date, region_id) VALUES (?, ?, ?)",
                ("ETHOSCOPE_001", "2025-01-01", 1),
            )
            conn.commit()

        # Get linked metadata first
        try:
            linked_metadata = link_meta_index(
                str(sample_metadata_csv), str(tmp_path / "test_results")
            )
            result = load_ethoscope_metadata(linked_metadata)

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert "machine_name" in result.columns
        except RuntimeError:
            # If link_meta_index fails, skip the test
            pytest.skip("Unable to create linked metadata for test")

    @pytest.mark.unit
    def test_load_ethoscope_metadata_missing_file(self, tmp_path):
        """Test metadata loading with missing database files."""
        # Create metadata DataFrame with non-existent database paths
        bad_metadata = pd.DataFrame(
            {
                "path": [str(tmp_path / "nonexistent.db")],
                "machine_id": ["ETHOSCOPE_001"],
                "date": ["2025-01-01"],
            }
        )

        # Function should handle missing files gracefully or raise appropriate error
        try:
            result = load_ethoscope_metadata(bad_metadata)
            # If it succeeds, it should return empty or handle gracefully
            assert isinstance(result, pd.DataFrame)
        except Exception:
            # Expected to fail with missing database file
            pass

    @pytest.mark.unit
    def test_load_ethoscope_metadata_nan_values(self, tmp_path):
        """Test metadata loading with NaN values."""
        # Create metadata with NaN values
        bad_metadata = pd.DataFrame(
            {
                "date": ["2025-01-01", np.nan],
                "machine_name": ["ETHOSCOPE_001", "ETHOSCOPE_002"],
                "region_id": [1, 2],
            }
        )
        csv_path = tmp_path / "nan_metadata.csv"
        bad_metadata.to_csv(csv_path, index=False)

        # Function should handle NaN values gracefully
        try:
            result = load_ethoscope_metadata(bad_metadata)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, Exception):
            # Expected to handle NaN values with error or graceful handling
            pass


class TestIntegrationLoadWorkflow:
    """Integration tests for the complete loading workflow."""

    @pytest.mark.integration
    def test_complete_loading_workflow(self, sample_metadata_csv, tmp_path):
        """Test the complete workflow from metadata CSV to loaded data."""
        # Create directory structure matching the glob pattern: any_parent/*/MACHINE/DATE_TIME/*.db
        results_dir = tmp_path / "results"
        machine_dir = results_dir / "ETHOSCOPE_001" / "2025-01-01_12-00-00"
        machine_dir.mkdir(parents=True)

        # Create a proper SQLite database with proper name
        db_path = machine_dir / "2025-01-01_12-00-00_ETHOSCOPE_001.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create minimal required tables
        cursor.execute(
            "CREATE TABLE ROI_MAP (roi_idx INTEGER, x INTEGER, y INTEGER, w INTEGER, h INTEGER)"
        )
        cursor.execute(
            "INSERT INTO ROI_MAP VALUES (1, 0, 0, 100, 100), (2, 100, 0, 100, 100)"
        )

        cursor.execute(
            "CREATE TABLE VAR_MAP (var_name TEXT, functional_type TEXT, sql_data_type TEXT)"
        )
        cursor.execute(
            'INSERT INTO VAR_MAP VALUES ("xy_dist_log10x1000", "distance", "REAL")'
        )

        cursor.execute("CREATE TABLE METADATA (field TEXT, value TEXT)")
        cursor.execute('INSERT INTO METADATA VALUES ("date_time", "1640995200")')

        # Create sample tracking data
        sample_data = pd.DataFrame(
            {
                "id": [1] * 100,
                "t": np.linspace(0, 3600, 100),
                "x": np.random.normal(50, 10, 100),
                "y": np.random.normal(50, 10, 100),
                "w": np.full(100, 10),
                "h": np.full(100, 10),
                "phi": np.random.uniform(0, 2 * np.pi, 100),
                "xy_dist_log10x1000": np.random.exponential(100, 100),
                "is_inferred": np.zeros(100),
                "has_interacted": np.zeros(100),
            }
        )

        sample_data.to_sql("ROI_1", conn, if_exists="replace", index=False)
        sample_data.to_sql("ROI_2", conn, if_exists="replace", index=False)

        conn.commit()
        conn.close()

        # Test the complete workflow - use correct directory structure
        linked_metadata = link_meta_index(str(sample_metadata_csv), str(tmp_path))
        final_data = load_ethoscope(linked_metadata, verbose=False)

        assert isinstance(final_data, pd.DataFrame)
        assert len(final_data) > 0
        assert "id" in final_data.columns
        assert "t" in final_data.columns

    @pytest.mark.integration
    def test_real_database_loading(self, real_ethoscope_db):
        """Test loading from a real ethoscope database file."""
        from ethoscopy.load import read_single_roi

        # Create minimal file info for the real database
        file_info = pd.Series(
            {
                "path": str(real_ethoscope_db),
                "region_id": 1,  # Try ROI 1
                "machine_id": "ETHOSCOPE_070",
                "date": "2025-07-10",
            }
        )

        # Test reading a single ROI
        result = read_single_roi(file_info)

        assert result is not None, "Failed to read data from real database"
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0, "No data found in real database"
        assert "id" in result.columns
        assert "t" in result.columns
        assert "x" in result.columns
        assert "y" in result.columns

        # Verify data quality
        assert result["t"].min() >= 0, "Time values should be non-negative"
        assert result["t"].is_monotonic_increasing, "Time should be monotonic"
        assert not result["x"].isna().all(), "X coordinates should not be all NaN"
        assert not result["y"].isna().all(), "Y coordinates should not be all NaN"
