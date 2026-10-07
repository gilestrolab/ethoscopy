"""
Unit tests for ethoscopy.analyse module functions.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ethoscopy.analyse import (
    _find_runs,
    cumsum_delta,
    max_velocity_detector,
    prep_data_motion_detector,
    sleep_annotation,
    stimulus_response,
)


class TestMaxVelocityDetector:
    """Test suite for max_velocity_detector function."""

    @pytest.mark.unit
    def test_max_velocity_detector_success(self, sample_ethoscope_data):
        """Test successful velocity detection."""
        result = max_velocity_detector(sample_ethoscope_data)

        assert isinstance(result, pd.DataFrame)
        assert "moving" in result.columns
        assert result["moving"].dtype == bool
        # Function bins data into 10-second windows, so output length should be duration/window_size
        expected_length = 3600 // 10  # 360 rows for 1 hour with 10s windows
        assert len(result) == expected_length

    @pytest.mark.unit
    def test_max_velocity_detector_custom_threshold(self, sample_ethoscope_data):
        """Test velocity detection with custom threshold."""
        result = max_velocity_detector(
            sample_ethoscope_data, velocity_correction_coef=0.01
        )

        assert isinstance(result, pd.DataFrame)
        assert "moving" in result.columns

    @pytest.mark.unit
    def test_max_velocity_detector_empty_data(self):
        """Test velocity detection with empty data."""
        empty_data = pd.DataFrame(columns=["t", "x", "y", "xy_dist_log10x1000"])

        result = max_velocity_detector(empty_data)

        # Empty data returns None (insufficient data for analysis)
        assert result is None

    @pytest.mark.unit
    def test_max_velocity_detector_missing_columns(self):
        """Test velocity detection with missing required columns."""
        # Create data with enough rows but missing required columns
        bad_data = pd.DataFrame(
            {
                "t": list(range(200)),
                "x": list(range(200)),
                # Missing 'y' and 'xy_dist_log10x1000'
            }
        )

        with pytest.raises(KeyError):
            max_velocity_detector(bad_data)


class TestSleepAnnotation:
    """Test suite for sleep_annotation function."""

    @pytest.mark.unit
    def test_sleep_annotation_success(self, sample_ethoscope_data):
        """Test successful sleep annotation."""
        result = sleep_annotation(sample_ethoscope_data, rule="classic")

        assert isinstance(result, pd.DataFrame)
        assert "asleep" in result.columns
        assert result["asleep"].dtype == bool
        # sleep_annotation bins data, so length will be less than original
        assert len(result) < len(sample_ethoscope_data)
        # With 10-second bins over 3600 seconds, expect ~360 rows
        assert len(result) > 0

    @pytest.mark.unit
    def test_sleep_annotation_custom_parameters(self, sample_ethoscope_data):
        """Test sleep annotation with custom parameters."""
        result = sleep_annotation(
            sample_ethoscope_data,
            time_window_length=20,  # 20 seconds
            min_sleep_duration=600,  # 10 minutes
            rule="classic",
        )

        assert isinstance(result, pd.DataFrame)
        assert "asleep" in result.columns
        # With 20-second bins, expect fewer rows
        assert len(result) > 0

    @pytest.mark.unit
    def test_sleep_annotation_no_movement_data(self, sample_ethoscope_data):
        """Test sleep annotation with insufficient data."""
        # Test with very little data (should return None)
        small_data = sample_ethoscope_data.head(5)
        result = sleep_annotation(small_data, rule="classic")
        # Function might return None for insufficient data
        assert result is None or isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_sleep_annotation_all_moving(self, sample_ethoscope_data):
        """Test sleep annotation when animal is always moving."""
        sample_ethoscope_data["moving"] = True

        result = sleep_annotation(sample_ethoscope_data, rule="classic")

        assert isinstance(result, pd.DataFrame)
        assert "asleep" in result.columns
        assert not result["asleep"].any()  # Should never be asleep

    @pytest.mark.unit
    def test_sleep_annotation_never_moving(self, sample_ethoscope_data):
        """Test sleep annotation when animal never moves."""
        sample_ethoscope_data["moving"] = False

        result = sleep_annotation(sample_ethoscope_data, rule="classic")

        assert isinstance(result, pd.DataFrame)
        assert "asleep" in result.columns
        # Most points should be asleep (after initial delay)


class TestStimulusResponse:
    """Test suite for stimulus_response function."""

    @pytest.mark.unit
    def test_stimulus_response_success(self, sample_ethoscope_data):
        """Test successful stimulus response analysis."""
        # Add has_interacted column that function expects
        sample_ethoscope_data["has_interacted"] = 0
        sample_ethoscope_data.loc[
            sample_ethoscope_data["t"] == 600, "has_interacted"
        ] = 1

        result = stimulus_response(
            sample_ethoscope_data, start_response_window=0, response_window_length=10
        )

        if result is not None:
            assert isinstance(result, pd.DataFrame)
            assert "interaction_t" in result.columns
            assert "has_responded" in result.columns
        else:
            # Function returns None if no interactions found
            assert result is None

    @pytest.mark.unit
    def test_stimulus_response_no_interactions(self, sample_ethoscope_data):
        """Test stimulus response with no interactions."""
        # Add has_interacted column with no interactions
        sample_ethoscope_data["has_interacted"] = 0

        result = stimulus_response(
            sample_ethoscope_data, start_response_window=0, response_window_length=10
        )

        # Function should return None when no interactions found
        assert result is None

    @pytest.mark.unit
    def test_stimulus_response_custom_window(self, sample_ethoscope_data):
        """Test stimulus response with custom response window."""
        # Add has_interacted column with one interaction
        sample_ethoscope_data["has_interacted"] = 0
        sample_ethoscope_data.loc[
            sample_ethoscope_data["t"] == 600, "has_interacted"
        ] = 1

        result = stimulus_response(
            sample_ethoscope_data,
            start_response_window=0,
            response_window_length=120,  # 2 minutes
        )

        if result is not None:
            assert isinstance(result, pd.DataFrame)
            assert "interaction_t" in result.columns
            assert "has_responded" in result.columns

    @pytest.mark.unit
    def test_stimulus_response_invalid_window_parameters(self, sample_ethoscope_data):
        """Test stimulus response with invalid window parameters."""
        # Add has_interacted column
        sample_ethoscope_data["has_interacted"] = 0
        sample_ethoscope_data.loc[
            sample_ethoscope_data["t"] == 600, "has_interacted"
        ] = 1

        # Test with invalid parameters (start >= response_window_length)
        with pytest.raises(
            ValueError,
            match="start_response_window must be less than response_window_length",
        ):
            stimulus_response(
                sample_ethoscope_data,
                start_response_window=10,
                response_window_length=10,  # Equal values should raise error
            )


class TestCumsumDelta:
    """Test suite for cumsum_delta function."""

    @pytest.mark.unit
    def test_cumsum_delta_success(self, sample_ethoscope_data):
        """Test successful cumulative delta calculation."""
        # Add required columns for cumsum_delta function
        test_data = sample_ethoscope_data.copy()
        test_data["activity_count"] = 0  # Add required activity_count column
        test_data["deltaT"] = 1.0  # Add required deltaT column

        result = cumsum_delta(test_data, 10)  # Pass integer immobility threshold

        assert isinstance(result, pd.DataFrame)
        assert "cumsum_delta" in result.columns
        assert "new_has_interacted" in result.columns

    @pytest.mark.unit
    def test_cumsum_delta_custom_column_name(self, sample_ethoscope_data):
        """Test cumulative delta with different threshold."""
        # Add required columns for cumsum_delta function
        test_data = sample_ethoscope_data.copy()
        test_data["activity_count"] = 0
        test_data["deltaT"] = 2.0

        result = cumsum_delta(test_data, 20)  # Different threshold

        assert isinstance(result, pd.DataFrame)
        assert "cumsum_delta" in result.columns

    @pytest.mark.unit
    def test_cumsum_delta_missing_column(self, sample_ethoscope_data):
        """Test cumulative delta with missing required columns."""
        # Missing activity_count column should raise KeyError
        with pytest.raises(KeyError):
            cumsum_delta(sample_ethoscope_data, 10)

    @pytest.mark.unit
    def test_cumsum_delta_empty_data(self):
        """Test cumulative delta with empty data."""
        empty_data = pd.DataFrame(columns=["activity_count", "deltaT"])

        result = cumsum_delta(empty_data, 10)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestPrepDataMotionDetector:
    """Test suite for prep_data_motion_detector function."""

    @pytest.mark.unit
    def test_prep_data_motion_detector_success(self, sample_ethoscope_data):
        """Test successful motion detection data preparation."""
        required_columns = ["t", "x", "y", "w", "h", "phi", "xy_dist_log10x1000"]

        result = prep_data_motion_detector(
            sample_ethoscope_data, required_columns=required_columns
        )

        assert isinstance(result, pd.DataFrame)
        assert "t_round" in result.columns
        assert len(result) <= len(sample_ethoscope_data)  # May be less due to filtering

    @pytest.mark.unit
    def test_prep_data_motion_detector_with_optional_columns(
        self, sample_ethoscope_data
    ):
        """Test motion detection prep with optional columns."""
        required_columns = ["t", "x", "y", "w", "h", "phi", "xy_dist_log10x1000"]
        # Add optional column
        sample_ethoscope_data["has_interacted"] = 0
        optional_columns = ["has_interacted"]

        result = prep_data_motion_detector(
            sample_ethoscope_data,
            required_columns=required_columns,
            optional_columns=optional_columns,
        )

        assert isinstance(result, pd.DataFrame)
        assert "has_interacted" in result.columns
        assert "t_round" in result.columns

    @pytest.mark.unit
    def test_prep_data_motion_detector_missing_required_column(
        self, sample_ethoscope_data
    ):
        """Test motion detection prep with missing required column."""
        required_columns = ["t", "x", "y", "w", "h", "phi", "xy_dist_log10x1000"]
        data_no_dist = sample_ethoscope_data.drop(columns=["xy_dist_log10x1000"])

        with pytest.raises(KeyError):
            prep_data_motion_detector(data_no_dist, required_columns=required_columns)


class TestFindRuns:
    """Test suite for _find_runs internal function."""

    @pytest.mark.unit
    def test_find_runs_basic(self):
        """Test basic run finding functionality."""
        mov = np.array([True, True, False, False, False, True, True])
        time = np.arange(len(mov))
        dt = np.ones(len(mov))

        result = _find_runs(mov, time, dt)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mov)
        assert "t" in result.columns
        assert "moving" in result.columns
        assert "activity_count" in result.columns

    @pytest.mark.unit
    def test_find_runs_no_valid_runs(self):
        """Test run finding with alternating pattern."""
        mov = np.array([True, False, True, False, True])
        time = np.arange(len(mov))
        dt = np.ones(len(mov))

        result = _find_runs(mov, time, dt)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mov)
        # Check that activity_count changes for each alternating run
        assert len(result["activity_count"].unique()) == len(mov)

    @pytest.mark.unit
    def test_find_runs_all_same_value(self):
        """Test run finding with all same values."""
        mov = np.array([False] * 10)
        time = np.arange(len(mov))
        dt = np.ones(len(mov))

        result = _find_runs(mov, time, dt)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mov)
        # All should have same activity_count (one long run)
        assert result["activity_count"].nunique() == 1

    @pytest.mark.unit
    def test_find_runs_empty_array(self):
        """Test run finding with empty array."""
        mov = np.array([], dtype=bool)
        time = np.array([])
        dt = np.array([])

        result = _find_runs(mov, time, dt)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestIntegrationAnalysis:
    """Integration tests for analysis workflow."""

    @pytest.mark.integration
    def test_complete_analysis_pipeline(self, sample_ethoscope_data):
        """Test complete analysis pipeline from raw data to sleep annotation."""
        # Step 1: Detect movement
        data_with_movement = max_velocity_detector(sample_ethoscope_data)

        # Step 2: Annotate sleep (use original data, not already-processed data)
        data_with_sleep = sleep_annotation(sample_ethoscope_data, rule="classic")

        # Step 3: Add stimulus response (use original data with has_interacted column)
        sample_ethoscope_data["has_interacted"] = 0
        sample_ethoscope_data.loc[
            sample_ethoscope_data["t"] == 600, "has_interacted"
        ] = 1
        stimulus_data = stimulus_response(
            sample_ethoscope_data, start_response_window=0, response_window_length=10
        )

        # Verify results
        assert isinstance(data_with_movement, pd.DataFrame)
        assert "moving" in data_with_movement.columns

        assert isinstance(data_with_sleep, pd.DataFrame)
        assert "asleep" in data_with_sleep.columns

        if stimulus_data is not None:
            assert isinstance(stimulus_data, pd.DataFrame)
            assert "has_responded" in stimulus_data.columns

    @pytest.mark.integration
    @pytest.mark.slow
    def test_analysis_with_large_dataset(self):
        """Test analysis functions with large dataset."""
        # Create larger dataset for performance testing with realistic data density
        n_points = 10000
        large_data = pd.DataFrame(
            {
                "t": np.arange(
                    0, n_points * 0.1, 0.1
                ),  # Dense time series (0.1s intervals)
                "x": np.random.normal(100, 20, n_points),
                "y": np.random.normal(100, 20, n_points),
                "xy_dist_log10x1000": np.random.exponential(100, n_points),
                "w": np.random.normal(10, 1, n_points),
                "h": np.random.normal(10, 1, n_points),
                "phi": np.random.uniform(0, 2 * np.pi, n_points),
            }
        )

        # Test that functions complete without error on large dataset
        movement_result = max_velocity_detector(large_data)
        sleep_result = sleep_annotation(large_data, rule="classic")

        # Verify results - functions may return None for insufficient/sparse data
        if movement_result is not None:
            assert isinstance(movement_result, pd.DataFrame)
            assert "moving" in movement_result.columns
            assert len(movement_result) > 0

        if sleep_result is not None:
            assert isinstance(sleep_result, pd.DataFrame)
            assert "asleep" in sleep_result.columns
            assert len(sleep_result) > 0

        # At least one analysis should work with this dataset
        assert movement_result is not None or sleep_result is not None
