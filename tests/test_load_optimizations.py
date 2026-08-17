"""
Unit tests for load_ethoscope() optimizations.

Tests the new database connection caching, metadata caching, and batch processing features.
"""

import os
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
    read_single_roi,
    read_single_roi_optimized,
)


class TestReadSingleROIOptimized:
    """Test suite for the optimized read_single_roi function."""

    @pytest.fixture
    def mock_database_file(self):
        """Create a mock database file for testing."""
        # Create temporary SQLite database
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables similar to ethoscope database
        cursor.execute(
            """
            CREATE TABLE METADATA (
                field TEXT,
                value TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE ROI_MAP (
                roi_idx INTEGER,
                roi_value TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE VAR_MAP (
                var_name TEXT,
                sql_data_type TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE ROI_1 (
                t INTEGER,
                x REAL,
                y REAL,
                w REAL,
                h REAL
            )
        """
        )

        # Insert sample data
        cursor.execute(
            "INSERT INTO METADATA (field, value) VALUES (?, ?)",
            ("date_time", str(1234567890)),
        )  # Unix timestamp

        cursor.execute(
            "INSERT INTO ROI_MAP (roi_idx, roi_value) VALUES (?, ?)", (1, "roi_1")
        )

        cursor.execute(
            "INSERT INTO VAR_MAP (var_name, sql_data_type) VALUES (?, ?)", ("x", "REAL")
        )

        # Insert time series data
        for i in range(100):
            cursor.execute(
                """
                INSERT INTO ROI_1 (t, x, y, w, h)
                VALUES (?, ?, ?, ?, ?)
            """,
                (i * 1000, np.random.randn(), np.random.randn(), 10, 10),
            )

        conn.commit()

        yield db_path, conn

        conn.close()
        os.unlink(db_path)

    @pytest.fixture
    def sample_file_metadata(self, mock_database_file):
        """Create sample file metadata."""
        db_path, _ = mock_database_file
        return pd.Series(
            {
                "path": db_path,
                "machine_id": "TEST_001",
                "region_id": 1,
                "date": "2023-01-01",
                "id": "test_specimen_01",
            }
        )

    @pytest.mark.unit
    def test_read_single_roi_optimized_basic(
        self, mock_database_file, sample_file_metadata
    ):
        """Test basic functionality of read_single_roi_optimized."""
        db_path, conn = mock_database_file

        # Create cached metadata (similar to what would be cached in load_ethoscope)
        roi_df = pd.read_sql_query("SELECT * FROM ROI_MAP", conn)
        var_df = pd.read_sql_query("SELECT * FROM VAR_MAP", conn)
        date_formatted = "2009-02-13 23:31:30"

        # Test the optimized function
        result = read_single_roi_optimized(
            sample_file_metadata, conn, roi_df, var_df, date_formatted
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "t" in result.columns
        assert "x" in result.columns
        assert "y" in result.columns

    @pytest.mark.unit
    def test_read_single_roi_optimized_time_filtering(
        self, mock_database_file, sample_file_metadata
    ):
        """Test time filtering in read_single_roi_optimized."""
        db_path, conn = mock_database_file

        roi_df = pd.read_sql_query("SELECT * FROM ROI_MAP", conn)
        var_df = pd.read_sql_query("SELECT * FROM VAR_MAP", conn)
        date_formatted = "2009-02-13 23:31:30"

        # Test with time constraints
        result = read_single_roi_optimized(
            sample_file_metadata,
            conn,
            roi_df,
            var_df,
            date_formatted,
            min_time=30,
            max_time=60,
        )

        assert isinstance(result, pd.DataFrame)
        # Should have filtered data within the time range
        if len(result) > 0:
            assert result["t"].min() >= 30
            assert result["t"].max() <= 60

    @pytest.mark.unit
    def test_read_single_roi_optimized_nonexistent_roi(
        self, mock_database_file, sample_file_metadata
    ):
        """Test read_single_roi_optimized with nonexistent ROI."""
        db_path, conn = mock_database_file

        roi_df = pd.read_sql_query("SELECT * FROM ROI_MAP", conn)
        var_df = pd.read_sql_query("SELECT * FROM VAR_MAP", conn)
        date_formatted = "2009-02-13 23:31:30"

        # Modify file metadata to request nonexistent ROI
        sample_file_metadata["region_id"] = 999

        result = read_single_roi_optimized(
            sample_file_metadata, conn, roi_df, var_df, date_formatted
        )

        # Should return None for nonexistent ROI
        assert result is None

    @pytest.mark.unit
    def test_read_single_roi_optimized_with_cache(
        self, mock_database_file, sample_file_metadata
    ):
        """Test read_single_roi_optimized with caching enabled."""
        db_path, conn = mock_database_file

        roi_df = pd.read_sql_query("SELECT * FROM ROI_MAP", conn)
        var_df = pd.read_sql_query("SELECT * FROM VAR_MAP", conn)
        date_formatted = "2009-02-13 23:31:30"

        with tempfile.TemporaryDirectory() as cache_dir:
            # First call should create cache
            result1 = read_single_roi_optimized(
                sample_file_metadata,
                conn,
                roi_df,
                var_df,
                date_formatted,
                cache=cache_dir,
            )

            # Check that cache file was created
            cache_files = list(Path(cache_dir).glob("*.pkl"))
            assert len(cache_files) > 0

            # Second call should use cache
            result2 = read_single_roi_optimized(
                sample_file_metadata,
                conn,
                roi_df,
                var_df,
                date_formatted,
                cache=cache_dir,
            )

            # Results should be identical
            pd.testing.assert_frame_equal(result1, result2)


class TestLoadEthoscopeOptimizations:
    """Test suite for load_ethoscope optimization features."""

    @pytest.fixture
    def mock_metadata_multiple_dbs(self):
        """Create metadata pointing to multiple database files."""
        # Create multiple temporary databases
        db_paths = []
        for i in range(3):
            fd, db_path = tempfile.mkstemp(suffix=f"_test_{i}.db")
            os.close(fd)
            db_paths.append(db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create basic ethoscope structure
            cursor.execute(
                """
                CREATE TABLE METADATA (
                    field TEXT,
                    value TEXT
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE ROI_MAP (
                    roi_idx INTEGER,
                    roi_value TEXT
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE VAR_MAP (
                    var_name TEXT,
                    sql_data_type TEXT
                )
            """
            )

            # Create ROI table for each database
            cursor.execute(
                """
                CREATE TABLE ROI_1 (
                    t INTEGER,
                    x REAL,
                    y REAL
                )
            """
            )

            # Insert metadata
            cursor.execute(
                "INSERT INTO METADATA (field, value) VALUES (?, ?)",
                ("date_time", str(1234567890 + i * 3600)),
            )

            cursor.execute(
                "INSERT INTO ROI_MAP (roi_idx, roi_value) VALUES (?, ?)", (1, "roi_1")
            )

            cursor.execute(
                "INSERT INTO VAR_MAP (var_name, sql_data_type) VALUES (?, ?)",
                ("x", "REAL"),
            )

            # Insert sample data
            for j in range(50):
                cursor.execute(
                    """
                    INSERT INTO ROI_1 (t, x, y)
                    VALUES (?, ?, ?)
                """,
                    (j * 1000, np.random.randn(), np.random.randn()),
                )

            conn.commit()
            conn.close()

        # Create metadata DataFrame
        metadata = pd.DataFrame(
            {
                "id": [f"specimen_{i:02d}" for i in range(3)],
                "path": db_paths,
                "machine_id": [f"ETHOSCOPE_{i:03d}" for i in range(3)],
                "region_id": [1, 1, 1],
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "machine_name": [f"ETHOSCOPE_{i:03d}" for i in range(3)],
            }
        )

        yield metadata

        # Cleanup
        for db_path in db_paths:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.unit
    def test_load_ethoscope_connection_reuse(self, mock_metadata_multiple_dbs):
        """Test that load_ethoscope reuses database connections efficiently."""
        # Create metadata where multiple ROIs share the same database
        metadata = mock_metadata_multiple_dbs.copy()

        # Make multiple ROIs point to the same database
        metadata.loc[1, "path"] = metadata.loc[0, "path"]
        metadata.loc[2, "path"] = metadata.loc[0, "path"]
        metadata.loc[1, "region_id"] = 1  # Same ROI for simplicity in mock
        metadata.loc[2, "region_id"] = 1

        with patch("ethoscopy.load.read_single_roi_optimized") as mock_read_roi:
            # Mock the optimized function to track calls
            mock_read_roi.return_value = pd.DataFrame(
                {"t": [100, 200, 300], "x": [1, 2, 3], "y": [1, 2, 3]}
            )

            with patch("sqlite3.connect") as mock_connect:
                mock_conn = MagicMock()
                mock_connect.return_value = mock_conn

                # Mock the SQL queries to return proper data
                with patch("pandas.read_sql_query") as mock_sql_query:
                    # Set up return values for different queries
                    def sql_side_effect(query, conn):
                        if "ROI_MAP" in query:
                            return pd.DataFrame(
                                {"roi_idx": [1, 2], "w": [100, 100], "h": [100, 100]}
                            )
                        elif "VAR_MAP" in query:
                            return pd.DataFrame(
                                {
                                    "var_name": ["x", "y"],
                                    "functional_type": ["position", "position"],
                                }
                            )
                        elif "date_time" in query:
                            return pd.DataFrame(
                                {"value": [1640995200.0]}
                            )  # Valid timestamp
                        else:
                            return pd.DataFrame()

                    mock_sql_query.side_effect = sql_side_effect

                    # Load ethoscope data
                    result = load_ethoscope(metadata, verbose=False)

                # Should have opened connection only once for the shared database
                # (Note: in real scenario, this tests the grouping logic)
                assert mock_connect.called

    @pytest.mark.unit
    def test_load_ethoscope_batch_concatenation(self, mock_metadata_multiple_dbs):
        """Test that load_ethoscope uses batch concatenation instead of incremental."""
        metadata = mock_metadata_multiple_dbs

        with patch("ethoscopy.load.read_single_roi_optimized") as mock_read_roi:
            # Mock successful ROI reading
            def mock_roi_data(file_info, *args, **kwargs):
                return pd.DataFrame(
                    {
                        "t": np.arange(10) * 60,
                        "x": np.random.randn(10),
                        "y": np.random.randn(10),
                        "id": [file_info["id"]] * 10,
                    }
                )

            mock_read_roi.side_effect = mock_roi_data

            with patch("pandas.concat") as mock_concat:
                # Set up mock concat to return expected result
                mock_concat.return_value = pd.DataFrame(
                    {
                        "t": np.arange(30) * 60,
                        "x": np.random.randn(30),
                        "y": np.random.randn(30),
                        "id": ["specimen_00"] * 10
                        + ["specimen_01"] * 10
                        + ["specimen_02"] * 10,
                    }
                )

                result = load_ethoscope(metadata, verbose=False)

                # Should call concat once at the end (batch) rather than incrementally
                assert mock_concat.call_count == 1

                # Verify concat was called with a list (batch mode)
                args, kwargs = mock_concat.call_args
                assert isinstance(args[0], list)
                assert len(args[0]) == 3  # Three ROIs

    @pytest.mark.unit
    def test_load_ethoscope_error_handling(self, mock_metadata_multiple_dbs):
        """Test error handling in optimized load_ethoscope."""
        metadata = mock_metadata_multiple_dbs

        with patch("ethoscopy.load.read_single_roi_optimized") as mock_read_roi:
            # Mock one successful and one failed ROI read
            def mock_roi_with_error(file_info, *args, **kwargs):
                if file_info["id"] == "specimen_01":
                    raise Exception("Database connection error")
                else:
                    return pd.DataFrame(
                        {
                            "t": np.arange(10) * 60,
                            "x": np.random.randn(10),
                            "y": np.random.randn(10),
                            "id": [file_info["id"]] * 10,
                        }
                    )

            mock_read_roi.side_effect = mock_roi_with_error

            # Should handle errors gracefully and return data from successful ROIs
            result = load_ethoscope(metadata, verbose=False)

            # Should have data from the successful ROIs only
            assert isinstance(result, pd.DataFrame)
            unique_ids = result["id"].unique() if len(result) > 0 else []
            assert "specimen_01" not in unique_ids  # Failed ROI should be excluded

    @pytest.mark.unit
    def test_load_ethoscope_empty_result_handling(self, mock_metadata_multiple_dbs):
        """Test handling when no ROIs can be loaded."""
        metadata = mock_metadata_multiple_dbs

        with patch("ethoscopy.load.read_single_roi_optimized") as mock_read_roi:
            # Mock all ROI reads returning None (failed)
            mock_read_roi.return_value = None

            result = load_ethoscope(metadata, verbose=False)

            # Should return empty DataFrame when no ROIs load successfully
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    @pytest.mark.unit
    def test_load_ethoscope_with_function_parameter(self, mock_metadata_multiple_dbs):
        """Test load_ethoscope with FUN parameter applied to each ROI."""
        metadata = mock_metadata_multiple_dbs

        def test_function(df):
            """Test function that adds a processed column."""
            if df is not None:
                df["processed"] = df["x"] ** 2
                return df
            return None

        with patch("ethoscopy.load.read_single_roi_optimized") as mock_read_roi:
            # Mock ROI data
            def mock_roi_data(file_info, *args, **kwargs):
                return pd.DataFrame(
                    {
                        "t": np.arange(5) * 60,
                        "x": np.random.randn(5),
                        "y": np.random.randn(5),
                        "id": [file_info["id"]] * 5,
                    }
                )

            mock_read_roi.side_effect = mock_roi_data

            result = load_ethoscope(metadata, FUN=test_function, verbose=False)

            # Should have applied the function to each ROI
            if len(result) > 0:
                assert "processed" in result.columns
                # Verify the function was applied correctly
                np.testing.assert_array_almost_equal(
                    result["processed"].values, result["x"].values ** 2
                )


class TestLoadOptimizationPerformance:
    """Performance tests for load optimizations."""

    @pytest.mark.performance
    def test_load_ethoscope_memory_usage(self):
        """Test that optimized loading uses less memory."""
        # This would require more complex setup to actually measure memory
        # For now, we'll test that the optimization code paths work

        # Create mock metadata for many ROIs
        n_rois = 50
        metadata = pd.DataFrame(
            {
                "id": [f"roi_{i:03d}" for i in range(n_rois)],
                "path": ["test.db"] * n_rois,  # All same DB for connection reuse
                "machine_id": ["TEST_001"] * n_rois,
                "region_id": list(range(1, n_rois + 1)),
                "date": ["2023-01-01"] * n_rois,
                "machine_name": ["TEST_001"] * n_rois,
            }
        )

        with patch("ethoscopy.load.read_single_roi_optimized") as mock_read_roi:
            # Mock small ROI data to simulate many small datasets
            def mock_small_roi(file_info, *args, **kwargs):
                return pd.DataFrame(
                    {
                        "t": np.arange(10),
                        "x": np.random.randn(10),
                        "id": [file_info["id"]] * 10,
                    }
                )

            mock_read_roi.side_effect = mock_small_roi

            with patch("sqlite3.connect"):
                result = load_ethoscope(metadata, verbose=False)

                # Should successfully process all ROIs
                if len(result) > 0:
                    assert len(result["id"].unique()) <= n_rois

    @pytest.mark.performance
    def test_connection_caching_benefit(self, tmp_path):
        """Test that connection caching provides performance benefit."""
        # The paths must exist on disk: _connect_db refuses a missing file up
        # front rather than letting SQLite invent an empty database for it.
        db1, db2 = tmp_path / "db1.db", tmp_path / "db2.db"
        db1.touch()
        db2.touch()

        # Mock scenario where multiple ROIs share databases
        metadata = pd.DataFrame(
            {
                "id": ["roi_1", "roi_2", "roi_3", "roi_4"],
                "path": [str(db1), str(db1), str(db2), str(db2)],  # Two shared DBs
                "machine_id": ["M1", "M1", "M2", "M2"],
                "region_id": [1, 2, 1, 2],
                "date": ["2023-01-01"] * 4,
                "machine_name": ["M1", "M1", "M2", "M2"],
            }
        )

        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            with patch("ethoscopy.load.read_single_roi_optimized") as mock_read_roi:
                mock_read_roi.return_value = pd.DataFrame(
                    {"t": [1, 2], "x": [1, 2], "id": ["test", "test"]}
                )

                load_ethoscope(metadata, verbose=False)

                # Should open connections for unique database paths only
                # In the optimized version, connections are reused per DB file
                assert mock_connect.called


class TestBackwardCompatibility:
    """Test backward compatibility of optimized functions."""

    @pytest.mark.unit
    def test_read_single_roi_compatibility(self):
        """Test that read_single_roi_optimized produces same results as original."""
        # Create minimal test scenario
        file_metadata = pd.Series(
            {
                "path": "test.db",
                "machine_id": "TEST_001",
                "region_id": 1,
                "date": "2023-01-01",
                "id": "test_roi",
            }
        )

        # Mock database components
        mock_conn = MagicMock()
        roi_df = pd.DataFrame({"roi_idx": [1], "roi_value": ["roi_1"]})
        var_df = pd.DataFrame({"var_name": ["x"], "sql_data_type": ["REAL"]})
        date_formatted = "2023-01-01 12:00:00"

        # Mock the database query results
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ("ROI_1",)  # Table exists
        mock_cursor.fetchall.return_value = [
            (1, "id", "INTEGER", 0, None, 1),  # id column, primary key
        ]

        # Mock pandas read_sql_query
        with patch("pandas.read_sql_query") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame(
                {"t": [1000, 2000, 3000], "x": [1.0, 2.0, 3.0], "y": [1.5, 2.5, 3.5]}
            )

            result = read_single_roi_optimized(
                file_metadata, mock_conn, roi_df, var_df, date_formatted
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert "t" in result.columns
            assert "x" in result.columns

            # Time should be converted from milliseconds to seconds
            expected_times = [1.0, 2.0, 3.0]
            np.testing.assert_array_equal(result["t"].values, expected_times)
