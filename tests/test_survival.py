"""
Tests for survival analysis: death detection, Kaplan-Meier estimation,
survival tables and the plots built on them.

The retrocompatibility tests matter most here: curate_dead_animals() has been
part of the public API for years, and death detection must keep giving the same
answer for data recorded at any sampling interval.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from ethoscopy.behavpy import behavpy
from ethoscopy.survival import (
    kaplan_meier,
    median_sampling_interval,
    sliding_window_death,
    survival_table,
    zero_run_death,
)

DAY = 24 * 3600


def make_specimen(interval, hours, dies_at=None, seed=0, gap=None):
    """
    Build one specimen's movement record.

    Args:
        interval (int): Sampling interval in seconds.
        hours (float): Length of the recording in hours.
        dies_at (float, optional): Hour after which the specimen stops moving.
        seed (int): Seed for the movement noise.
        gap (tuple, optional): (start_hour, end_hour) of data to drop.

    Returns:
        pd.DataFrame: Columns t, moving, asleep.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, int(hours * 3600), interval)
    moving = rng.random(len(t)) < 0.4
    if dies_at is not None:
        moving[t >= dies_at * 3600] = False
    if gap is not None:
        keep = ~((t >= gap[0] * 3600) & (t < gap[1] * 3600))
        t, moving = t[keep], moving[keep]
    return pd.DataFrame({"t": t, "moving": moving, "asleep": ~moving})


def make_behavpy(specs, canvas="seaborn", meta_extra=None):
    """
    Assemble a behavpy object from a dict of specimen id to DataFrame.

    Args:
        specs (dict): Specimen id mapped to its DataFrame.
        canvas (str): Plotting backend.
        meta_extra (dict, optional): Extra metadata columns per specimen id.

    Returns:
        behavpy: Linked data and metadata.
    """
    data = pd.concat([df.assign(id=name) for name, df in specs.items()]).set_index("id")
    meta = pd.DataFrame(
        [{"id": name, **(meta_extra or {}).get(name, {})} for name in specs]
    ).set_index("id")
    return behavpy(data, meta, check=True, canvas=canvas)


class TestKaplanMeier:
    """The estimator itself, against values that can be computed by hand."""

    @pytest.mark.unit
    def test_matches_hand_computed_survival(self):
        """Textbook example: deaths at 1, 3, 4 with censoring at 2 and 5."""
        km = kaplan_meier([1, 2, 3, 4, 5], [1, 0, 1, 1, 0])

        assert km["time"].tolist() == [1, 2, 3, 4, 5]
        np.testing.assert_allclose(
            km["survival"].to_numpy(),
            [0.8, 0.8, 0.8 * 2 / 3, 0.8 * 2 / 3 * 0.5, 0.8 * 2 / 3 * 0.5],
        )
        assert km["n_at_risk"].tolist() == [5, 4, 3, 2, 1]
        assert km["n_events"].tolist() == [1, 0, 1, 1, 0]
        assert km["n_censored"].tolist() == [0, 1, 0, 0, 1]

    @pytest.mark.unit
    def test_greenwood_confidence_interval(self):
        """The band is the log-transformed Greenwood interval."""
        km = kaplan_meier([1, 2, 3], [1, 1, 1])

        # After the first death: S = 2/3, var(log S) = 1 / (3 * 2).
        se_log = np.sqrt(1 / (3 * 2))
        expected_lower = (2 / 3) * np.exp(-1.959963984540054 * se_log)
        assert km["ci_lower"].iloc[0] == pytest.approx(expected_lower)
        assert km["ci_upper"].iloc[0] == pytest.approx(
            min(1.0, (2 / 3) * np.exp(1.959963984540054 * se_log))
        )
        # The band is degenerate once survival reaches zero.
        assert km["survival"].iloc[-1] == 0
        assert km["ci_lower"].iloc[-1] == 0
        assert km["ci_upper"].iloc[-1] == 0

    @pytest.mark.unit
    def test_ties_are_handled_together(self):
        """Two deaths at the same time drop the curve in one step."""
        km = kaplan_meier([5, 5, 9], [1, 1, 1])

        assert len(km) == 2
        assert km["n_events"].iloc[0] == 2
        assert km["survival"].iloc[0] == pytest.approx(1 / 3)

    @pytest.mark.unit
    def test_all_censored_keeps_survival_at_one(self):
        """With no deaths the curve stays flat rather than vanishing."""
        km = kaplan_meier([3, 6, 9], [0, 0, 0])

        assert (km["survival"] == 1.0).all()
        assert km["time"].tolist() == [3, 6, 9]

    @pytest.mark.unit
    def test_empty_input_returns_empty_frame(self):
        """No subjects gives an empty frame with the expected columns."""
        km = kaplan_meier([], [])

        assert len(km) == 0
        assert "survival" in km.columns


class TestDeathDetection:
    """The sliding window and zero-run rules."""

    @pytest.mark.unit
    def test_finds_start_of_sustained_immobility(self):
        """Death is reported at the start of the qualifying window."""
        spec = make_specimen(60, hours=120, dies_at=48)
        death = sliding_window_death(
            spec["t"].to_numpy(),
            spec["moving"].to_numpy(),
            time_window_s=DAY,
            step=3600,
            prop_immobile=0.01,
        )

        assert death == pytest.approx(48 * 3600, abs=3600)

    @pytest.mark.unit
    def test_returns_none_for_a_living_specimen(self):
        """An animal that keeps moving is never declared dead."""
        spec = make_specimen(60, hours=120)
        death = sliding_window_death(
            spec["t"].to_numpy(),
            spec["moving"].to_numpy(),
            time_window_s=DAY,
            step=3600,
            prop_immobile=0.01,
        )

        assert death is None

    @pytest.mark.unit
    def test_min_coverage_ignores_a_sparse_window(self):
        """
        A short stretch of quiet data at a recording stop is not death.

        Without min_coverage the trailing sliver qualifies, which is what made
        machine restarts read as deaths.
        """
        t = np.concatenate(
            [np.arange(0, 72 * 3600, 60), np.arange(72 * 3600, 72 * 3600 + 600, 60)]
        )
        moving = np.ones(len(t), dtype=float)
        moving[t >= 72 * 3600] = 0.0

        args = dict(time_window_s=DAY, step=3600, prop_immobile=0.01)
        assert sliding_window_death(t, moving, **args) is not None
        assert sliding_window_death(t, moving, min_coverage=0.75, **args) is None

    @pytest.mark.unit
    def test_zero_run_detects_contiguous_immobility(self):
        """A long enough run of zeros reports its own start."""
        t = np.arange(0, 48 * 3600, 60)
        moving = np.ones(len(t))
        moving[(t >= 10 * 3600) & (t < 24 * 3600)] = 0

        assert zero_run_death(t, moving, zero_run_s=12 * 3600) == pytest.approx(
            10 * 3600
        )
        assert zero_run_death(t, moving, zero_run_s=20 * 3600) is None

    @pytest.mark.unit
    def test_median_sampling_interval(self):
        """The interval is read from the data, not assumed."""
        assert median_sampling_interval(np.arange(0, 1000, 10)) == 10
        assert median_sampling_interval(np.array([5.0])) is None


class TestCurateRetrocompatibility:
    """curate_dead_animals() must behave as it always has."""

    @pytest.mark.parametrize("interval", [10, 60, 300])
    @pytest.mark.unit
    def test_death_detected_at_every_sampling_interval(self, interval):
        """
        Detection must not assume 10 second sampling.

        Binned data (60 s, 300 s) has fewer points per window; a rule keyed to a
        hardcoded interval silently stops detecting death there.
        """
        df = make_behavpy({"fly1": make_specimen(interval, hours=120, dies_at=48)})
        curated = df.curate_dead_animals()

        assert len(curated) < len(df)
        assert curated["t"].max() == pytest.approx(48 * 3600, abs=2 * 3600)

    @pytest.mark.unit
    def test_living_specimen_is_left_untouched(self):
        """Nothing is trimmed when no death is detected."""
        df = make_behavpy({"fly1": make_specimen(60, hours=120)})

        assert len(df.curate_dead_animals()) == len(df)

    @pytest.mark.unit
    def test_min_coverage_is_opt_in(self):
        """
        The coverage rule changes results, so it must not apply by default.

        The specimen stops moving only in a short tail after a gap, which the
        default reads as death and the coverage rule rejects.
        """
        t = np.concatenate(
            [np.arange(0, 72 * 3600, 60), np.arange(96 * 3600, 96 * 3600 + 600, 60)]
        )
        moving = np.ones(len(t), dtype=bool)
        moving[t >= 96 * 3600] = False
        spec = pd.DataFrame({"t": t, "moving": moving})
        df = make_behavpy({"fly1": spec})

        assert len(df.curate_dead_animals()) < len(df)
        assert len(df.curate_dead_animals(min_coverage=0.75)) == len(df)

    @pytest.mark.unit
    def test_invalid_resolution_raises(self):
        """Resolution is validated as before."""
        df = make_behavpy({"fly1": make_specimen(60, hours=48)})

        with pytest.raises(ValueError):
            df.curate_dead_animals(resolution=0)
        with pytest.raises(ValueError):
            df.curate_dead_animals(time_window=12, resolution=24)


class TestBaselineRetrocompatibility:
    """baseline() shifts time without changing the column's type."""

    @pytest.mark.unit
    def test_shift_is_applied_per_specimen(self):
        """Only the specimens with a non-zero baseline move."""
        df = make_behavpy(
            {"fly1": make_specimen(60, hours=24), "fly2": make_specimen(60, hours=24)},
            meta_extra={"fly1": {"baseline": 0}, "fly2": {"baseline": 2}},
        )
        shifted = df.baseline("baseline")

        assert shifted.xmv("id", "fly2")["t"].min() == 2 * DAY
        assert shifted.xmv("id", "fly1")["t"].min() == 0

    @pytest.mark.unit
    def test_whole_day_string_shift_keeps_integer_timestamps(self):
        """
        A baseline read from a string stays integral.

        Numeric baselines are converted through float and promote the column, as
        they always have; string ones must not.
        """
        df = make_behavpy(
            {"fly1": make_specimen(60, hours=24), "fly2": make_specimen(60, hours=24)},
            meta_extra={
                "fly1": {"baseline": "0"},
                "fly2": {"baseline": "2 days"},
            },
        )
        shifted = df.baseline("baseline")

        assert shifted["t"].dtype == df["t"].dtype
        assert shifted.xmv("id", "fly2")["t"].min() == 2 * DAY

    @pytest.mark.unit
    def test_no_shift_leaves_timestamps_identical(self):
        """An all-zero baseline column is a no-op."""
        df = make_behavpy(
            {"fly1": make_specimen(60, hours=24)},
            meta_extra={"fly1": {"baseline": 0}},
        )

        pd.testing.assert_series_equal(df.baseline("baseline")["t"], df["t"])


class TestSurvivalTable:
    """Per-subject time-to-event tables."""

    @pytest.mark.unit
    def test_records_death_and_censoring(self):
        """A dead specimen is an event, a living one is censored."""
        df = make_behavpy(
            {
                "dead": make_specimen(60, hours=120, dies_at=48, seed=1),
                "alive": make_specimen(60, hours=120, seed=2),
            }
        )
        table = df.survival_table()

        assert table.loc["dead", "E"] == 1
        assert table.loc["dead", "T"] == pytest.approx(48, abs=2)
        assert table.loc["alive", "E"] == 0
        assert table.loc["alive", "T"] == pytest.approx(120, abs=1)

    @pytest.mark.unit
    def test_segments_are_counted_and_all_searched(self):
        """
        Death in an early segment is found, not only in the last one.

        A specimen re-recorded after a stop looks dead from the start of the
        later segment; searching only that segment dates the death wrongly.
        """
        spec = make_specimen(60, hours=120, dies_at=48, gap=(60, 90), seed=3)
        df = make_behavpy({"fly1": spec})
        table = df.survival_table()

        assert table.loc["fly1", "n_segments"] == 2
        assert table.loc["fly1", "E"] == 1
        assert table.loc["fly1", "T"] == pytest.approx(48, abs=2)

    @pytest.mark.unit
    def test_subject_cols_merge_sessions_of_one_animal(self):
        """Two recordings of the same ROI count as one animal, not two."""
        first = make_specimen(60, hours=48, seed=4)
        second = make_specimen(60, hours=48, dies_at=24, seed=5)
        second["t"] += 72 * 3600  # a later session on the same machine and ROI
        df = make_behavpy(
            {"session_a": first, "session_b": second},
            meta_extra={
                "session_a": {"machine_name": "ETHOSCOPE_001", "region_id": 1},
                "session_b": {"machine_name": "ETHOSCOPE_001", "region_id": 1},
            },
        )

        assert len(df.survival_table()) == 2
        merged = df.survival_table(subject_cols=["machine_name", "region_id"])
        assert len(merged) == 1
        assert merged["n_segments"].iloc[0] == 2
        assert merged["E"].iloc[0] == 1

    @pytest.mark.unit
    def test_overlapping_sessions_raise(self):
        """Merging needs a shared clock; overlapping sessions cannot be merged."""
        df = make_behavpy(
            {
                "session_a": make_specimen(60, hours=48, seed=6),
                "session_b": make_specimen(60, hours=48, seed=7),
            },
            meta_extra={
                "session_a": {"machine_name": "ETHOSCOPE_001", "region_id": 1},
                "session_b": {"machine_name": "ETHOSCOPE_001", "region_id": 1},
            },
        )

        with pytest.raises(ValueError, match="overlap in time"):
            df.survival_table(subject_cols=["machine_name", "region_id"])

    @pytest.mark.unit
    def test_missing_columns_raise(self):
        """Unknown data or metadata columns are reported clearly."""
        df = make_behavpy({"fly1": make_specimen(60, hours=48)})

        with pytest.raises(KeyError):
            df.survival_table(mov_column="not_a_column")
        with pytest.raises(KeyError):
            df.survival_table(subject_cols=["not_a_column"])

    @pytest.mark.unit
    def test_second_movement_column_triggers_detection(self):
        """Either movement column can declare death."""
        spec = make_specimen(60, hours=120, seed=8)
        spec["micro"] = spec["moving"].astype(float)
        spec.loc[spec["t"] >= 48 * 3600, "micro"] = 0.0
        df = make_behavpy({"fly1": spec})

        assert df.survival_table().loc["fly1", "E"] == 0
        combined = df.survival_table(second_mov_column="micro")
        assert combined.loc["fly1", "E"] == 1

    @pytest.mark.unit
    def test_zero_run_option_detects_earlier_death(self):
        """A run of complete immobility can precede the windowed detection."""
        t = np.arange(0, 120 * 3600, 60)
        moving = np.ones(len(t))
        moving[(t >= 20 * 3600) & (t < 40 * 3600)] = 0
        df = make_behavpy({"fly1": pd.DataFrame({"t": t, "moving": moving})})

        assert df.survival_table().loc["fly1", "E"] == 0
        with_run = df.survival_table(zero_run_hours=12)
        assert with_run.loc["fly1", "E"] == 1
        assert with_run.loc["fly1", "T"] == pytest.approx(20, abs=1)


class TestDeathTable:
    """The user-facing list of deaths."""

    @pytest.mark.unit
    def test_lists_dead_subjects_with_metadata(self):
        """Only deaths are listed, ordered by time, with requested metadata."""
        df = make_behavpy(
            {
                "dead": make_specimen(60, hours=120, dies_at=48, seed=9),
                "alive": make_specimen(60, hours=120, seed=10),
            },
            meta_extra={"dead": {"genotype": "mutant"}, "alive": {"genotype": "wt"}},
        )
        deaths = df.km_death_table(meta_cols=["genotype"])

        assert deaths["id"].tolist() == ["dead"]
        assert deaths["genotype"].tolist() == ["mutant"]
        assert deaths["T"].iloc[0] == pytest.approx(48, abs=2)

    @pytest.mark.unit
    def test_days_convert_from_hours(self):
        """Time can be reported in days."""
        df = make_behavpy({"fly1": make_specimen(60, hours=120, dies_at=48, seed=11)})

        hours = df.km_death_table()["T"].iloc[0]
        days = df.km_death_table(time_unit="days")["T"].iloc[0]
        assert days == pytest.approx(hours / 24)

    @pytest.mark.unit
    def test_invalid_time_unit_raises(self):
        """An unknown unit is rejected rather than silently ignored."""
        df = make_behavpy({"fly1": make_specimen(60, hours=48)})

        with pytest.raises(ValueError, match="time_unit"):
            df.km_death_table(time_unit="fortnights")


class TestSurvivalPlots:
    """Both canvases must draw the curves they are given."""

    @staticmethod
    def _dataset(canvas):
        specs, meta = {}, {}
        for i in range(6):
            genotype = "mutant" if i % 2 else "wt"
            specs[f"fly{i}"] = make_specimen(
                60, hours=120, dies_at=48 if genotype == "mutant" else None, seed=i
            )
            meta[f"fly{i}"] = {"genotype": genotype}
        return make_behavpy(specs, canvas=canvas, meta_extra=meta)

    @pytest.mark.parametrize("canvas", ["seaborn", "plotly"])
    @pytest.mark.unit
    def test_km_survival_plot_draws_one_curve_per_group(self, canvas):
        """Faceting produces a curve per genotype on either backend."""
        fig = self._dataset(canvas).km_survival_plot(facet_col="genotype")

        if canvas == "seaborn":
            assert len(fig.axes[0].get_lines()) >= 2
        else:
            names = [tr.name for tr in fig.data if tr.name]
            assert any("mutant" in n for n in names)
            assert any("wt" in n for n in names)

    @pytest.mark.parametrize("canvas", ["seaborn", "plotly"])
    @pytest.mark.unit
    def test_km_survival_plot_without_faceting(self, canvas):
        """A single pooled curve is drawn when no facet column is given."""
        assert self._dataset(canvas).km_survival_plot() is not None

    @pytest.mark.parametrize("canvas", ["seaborn", "plotly"])
    @pytest.mark.unit
    def test_km_survival_plot_rejects_bad_time_unit(self, canvas):
        """The time unit is validated before any work is done."""
        with pytest.raises(ValueError, match="time_unit"):
            self._dataset(canvas).km_survival_plot(time_unit="weeks")

    @pytest.mark.parametrize("canvas", ["seaborn", "plotly"])
    @pytest.mark.unit
    def test_plot_sleep_bouts(self, canvas):
        """Bout histograms are drawn for each facet group."""
        fig = self._dataset(canvas).plot_sleep_bouts(
            facet_col="genotype", bin_size=5, max_bins=20
        )

        if canvas == "plotly":
            assert len(fig.data) == 2
        else:
            assert len(fig.axes[0].patches) > 0

    @pytest.mark.unit
    def test_plot_sleep_bouts_without_bouts_raises(self):
        """A dataset with no qualifying bouts says so rather than drawing nothing."""
        t = np.arange(0, 12 * 3600, 60)
        df = make_behavpy(
            {"fly1": pd.DataFrame({"t": t, "asleep": np.zeros(len(t), dtype=bool)})}
        )

        with pytest.raises(ValueError, match="No data to plot"):
            df.plot_sleep_bouts()


class TestSurvivalTableFunction:
    """The pure function behind the behavpy method."""

    @pytest.mark.unit
    def test_subject_series_controls_grouping(self):
        """Specimens mapped to one key are treated as a single subject."""
        data = pd.concat(
            [
                make_specimen(60, hours=24, seed=12).assign(id="a"),
                make_specimen(60, hours=24, seed=13).assign(id="b"),
            ]
        ).set_index("id")
        data.loc["b", "t"] = data.loc["b", "t"] + 48 * 3600

        one = survival_table(data, pd.Series({"a": "same", "b": "same"}))
        two = survival_table(data, pd.Series({"a": "a", "b": "b"}))

        assert len(one) == 1
        assert len(two) == 2
