"""
Tests for plot_hmm_response: response to stimuli per decoded HMM state.

Synthetic flies alternate 20-min active and still blocks and receive a true
stimulus (has_interacted == 1) or a mock one (== 2) every 5 minutes. The
expected per-fly response rates are computed independently, one fly at a time.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from hmmlearn.hmm import CategoricalHMM

import ethoscopy as etho
from ethoscopy.analyse import sleep_annotation, stimulus_response

LABELS = ["Deep sleep", "Light sleep", "Quiet awake", "Active awake"]
INTERACTIONS = {1: "True Stimulus", 2: "Spon. Mov."}
STILL, MOVE = -3000, -2000  # xy_dist_log10x1000 for velocity ~0.33 and ~3.3
IDS = ["fly_1", "fly_2", "fly_3", "fly_4"]
T_BIN = 60


def _raw_fly(seed: int, hours: int = 4) -> pd.DataFrame:
    """
    Simulate one fly's raw tracking at 1 frame per second.

    Args:
        seed (int): Seed for the random generator.
        hours (int, optional): Length of the recording. Default is 4.

    Returns:
        pd.DataFrame: Raw frames with the columns stimulus_response and
            sleep_annotation need.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, hours * 3600, dtype=float)
    active = (t // 1200) % 2 == 0
    xy = np.where(active & (rng.random(len(t)) < 0.5), MOVE, STILL)
    has_interacted = np.zeros(len(t), dtype=int)
    for k, start in enumerate(np.arange(150, len(t) - 20, 300).astype(int)):
        kind = 1 if k % 2 == 0 else 2
        has_interacted[start] = kind
        # Reason: still flies answer true stimuli more often than mock ones, so the states differ.
        if rng.random() < (0.6 if kind == 1 else 0.1):
            xy[start + 2] = MOVE
    return pd.DataFrame(
        {
            "t": t,
            "x": 0.5 + rng.normal(0, 1e-3, len(t)),
            "y": np.full(len(t), 0.5),
            "w": np.full(len(t), 0.05),
            "h": np.full(len(t), 0.02),
            "phi": np.zeros(len(t)),
            "xy_dist_log10x1000": xy,
            "has_interacted": has_interacted,
        }
    )


@pytest.fixture(scope="module")
def raw_flies():
    """Raw frames for each synthetic fly, keyed by id."""
    return {fid: _raw_fly(seed) for seed, fid in enumerate(IDS)}


@pytest.fixture(scope="module")
def hmm():
    """A fixed 4-state model: two sleep states, two awake states."""
    model = CategoricalHMM(n_components=4, n_features=2)
    model.startprob_ = np.full(4, 0.25)
    model.transmat_ = np.full((4, 4), 0.1 / 3) + np.eye(4) * (0.9 - 0.1 / 3)
    model.emissionprob_ = np.array([[0.99, 0.01], [0.8, 0.2], [0.4, 0.6], [0.05, 0.95]])
    return model


def _behavpy_pair(raw_flies, canvas):
    """Build the response and movement behavpy objects for one canvas."""
    meta = pd.DataFrame({"id": IDS, "genotype": ["A", "A", "B", "B"]}).set_index("id")
    resp = pd.concat(
        [stimulus_response(d.copy()).assign(id=fid) for fid, d in raw_flies.items()]
    ).set_index("id")
    mov = pd.concat(
        [sleep_annotation(d.copy(), rule="classic").assign(id=fid) for fid, d in raw_flies.items()]
    ).set_index("id")
    return (
        etho.behavpy(resp, meta, check=True, canvas=canvas),
        etho.behavpy(mov, meta, check=True, canvas=canvas),
    )


def _expected(raw_flies, hmm):
    """
    Per-fly response rate by state and interaction type, one fly at a time.

    The state of the 60-s bin holding the stimulus is used only if the response
    fell in a later bin; otherwise the previous bin's state is used, so the
    response does not decide the state it is credited to.
    """
    rows = []
    for fid, d in raw_flies.items():
        mov = sleep_annotation(d.copy(), rule="classic")
        moving = mov.groupby(mov.t // T_BIN * T_BIN)["moving"].max().astype(int)
        _, states = hmm.decode(moving.to_numpy().reshape(-1, 1))
        decoded = pd.DataFrame(
            {
                "bin": moving.index,
                "state": states,
                "previous_state": np.r_[np.nan, states[:-1]],
            }
        )
        resp = stimulus_response(d.copy())
        resp["bin"] = resp.interaction_t // T_BIN * T_BIN
        merged = resp.merge(decoded, on="bin")
        later = (merged.interaction_t + merged.t_rel) // T_BIN * T_BIN > merged.bin
        merged["used"] = np.where(later, merged.state, merged.previous_state)
        rate = merged.dropna(subset=["used"]).groupby(["used", "has_interacted"])[
            "has_responded"
        ]
        for (state, kind), value in rate.mean().items():
            rows.append((fid, LABELS[int(state)], INTERACTIONS[kind], value))
    return pd.DataFrame(rows, columns=["id", "state", "interaction", "rate"])


def _observed(grouped_data):
    """Reduce plot_hmm_response's table to the columns of _expected()."""
    out = grouped_data.reset_index()[
        ["id", "state", "has_interacted", "has_responded_mean"]
    ]
    return out.rename(
        columns={"has_interacted": "interaction", "has_responded_mean": "rate"}
    )


def _sorted(df):
    return df.sort_values(["id", "state", "interaction"]).reset_index(drop=True)


@pytest.mark.unit
@pytest.mark.parametrize("canvas", ["plotly", "seaborn"])
class TestPlotHmmResponse:
    """plot_hmm_response on both backends."""

    def teardown_method(self):
        plt.close("all")

    def test_rates_match_per_fly_calculation(self, raw_flies, hmm, canvas):
        resp, mov = _behavpy_pair(raw_flies, canvas)
        fig, grouped = resp.plot_hmm_response(mov, hmm, labels=LABELS)

        assert isinstance(fig, go.Figure if canvas == "plotly" else plt.Figure)
        observed, expected = _sorted(_observed(grouped)), _sorted(
            _expected(raw_flies, hmm)
        )
        # Reason: guard the fixture itself; both kinds of stimulus and several states must occur.
        assert set(expected.interaction) == set(INTERACTIONS.values())
        assert expected.state.nunique() >= 2
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False)

    def test_facets_keep_t_bin_and_labels_apart(self, raw_flies, hmm, canvas):
        resp, mov = _behavpy_pair(raw_flies, canvas)
        _, grouped = resp.plot_hmm_response(
            mov,
            hmm,
            labels=LABELS,
            facet_col="genotype",
            facet_arg=["A", "B"],
            facet_labels=["geno A", "geno B"],
            t_bin=T_BIN,
        )

        assert set(grouped["genotype"]) <= {
            f"geno {g} {kind}" for g in "AB" for kind in INTERACTIONS.values()
        }
        by_id = grouped.reset_index().groupby("id")["genotype"].first()
        assert by_id["fly_1"].startswith("geno A") and by_id["fly_4"].startswith(
            "geno B"
        )

    def test_one_hmm_per_facet(self, raw_flies, hmm, canvas):
        resp, mov = _behavpy_pair(raw_flies, canvas)
        _, grouped = resp.plot_hmm_response(
            mov,
            [hmm, hmm],
            labels=LABELS,
            facet_col="genotype",
            facet_arg=["A", "B"],
            t_bin=[T_BIN, T_BIN],
        )

        # Same model and bin for both groups, so the rates equal the single-model ones.
        pd.testing.assert_frame_equal(
            _sorted(_observed(grouped)),
            _sorted(_expected(raw_flies, hmm)),
            check_dtype=False,
        )

    def test_missing_response_column_raises(self, raw_flies, hmm, canvas):
        resp, mov = _behavpy_pair(raw_flies, canvas)
        with pytest.raises(KeyError):
            resp.plot_hmm_response(mov, hmm, response_col="not_a_column")

    def test_several_hmms_need_a_facet(self, raw_flies, hmm, canvas):
        resp, mov = _behavpy_pair(raw_flies, canvas)
        with pytest.raises(RuntimeError):
            resp.plot_hmm_response(mov, [hmm, hmm], t_bin=[T_BIN, T_BIN])
