# Ethoscopy — task log

## 2026-04-21 — Fix missing tutorial pickles on PyPI installs

**Problem.** Users installing ethoscopy 2.0.4 from PyPI hit
`FileNotFoundError: Tutorial data files not found in: …/ethoscopy/misc/tutorial_data`
when running `get_tutorial('overview')`. Root cause: `.gitignore` line 9
contains `*.pkl`; Hatchling's default VCS plugin honours `.gitignore` when
selecting wheel contents, so every tracked pickle was silently dropped from
the 2.0.4 wheel (confirmed by inspecting the 144 KB artifact on PyPI).

**Decision.** Keep the pickles *out* of the wheel by design
(`overview_data.pkl` alone is ~31 MB — ~200× the code payload). Instead,
ship an explicit fetch path and document it clearly everywhere.

### Changes

- [x] `pyproject.toml`: add explicit `[tool.hatch.build.targets.wheel] exclude`
      for `src/ethoscopy/misc/tutorial_data/*.pkl` so intent is independent of
      `.gitignore`.
- [x] `src/ethoscopy/misc/get_tutorials.py`: add `download_tutorial_data()`
      (stdlib `urllib`, idempotent, overwrite flag) and rewrite the
      `FileNotFoundError` with a copy-paste recovery snippet + GitHub URL.
- [x] `src/ethoscopy/misc/get_HMM.py`: update its `FileNotFoundError` to point
      at the same helper (the 4-state HMM pickles live in the same folder).
- [x] `src/ethoscopy/__init__.py`: re-export `download_tutorial_data` and
      `get_tutorial` at the top level so users call
      `etho.download_tutorial_data()`.
- [x] `tests/test_get_tutorials.py`: add 5 tests for the new helper (URL
      coverage, full download, skip-if-present, overwrite, network error).
      15 tests pass.
- [x] Six tutorial notebooks (`1_Overview`, `2_HMM`, `3_Circadian`,
      `5_Ethoscopy_catch22`, `6_Ethoscopy_to_hctsa`, `notebook_paper`): insert
      one markdown + one code cell before each first `get_tutorial(...)`
      explaining the one-time fetch.
- [x] `README.md`: add a "Tutorial data" section with the one-liner plus a
      fallback manual-download path.

### Follow-up (same session)

After shipping the first cut, a second concern surfaced: the default
destination (`<site-packages>/ethoscopy/misc/tutorial_data/`) is only
writable for the user who installed ethoscopy. In system-wide installs,
conda base envs, or Docker images where the package was installed by
root, a non-root user calling `etho.download_tutorial_data()` would hit
`PermissionError`. Fixed by:

- [x] Default `download_tutorial_data(dest_dir=...)` to
      `~/.cache/ethoscopy/tutorial_data/` (always user-writable).
- [x] `get_tutorial` / `get_HMM` now consult three locations in order:
      (1) package dir, (2) `$ETHOSCOPY_TUTORIAL_DATA_DIR`, (3) user cache.
      `_missing_files_message()` prints the full search list so users
      know exactly where ethoscopy looked.
- [x] `PermissionError` on `mkdir` surfaces a wrapped message pointing
      at `dest_dir=` / the env override.
- [x] New tests: search-path ordering, env-var override, cache fallback,
      permission-error wrapping, default-dest-is-user-cache. 22/22 pass.
- [x] `Docker/Dockerfile` now runs
      `download_tutorial_data(dest_dir=package_tutorial_data_dir())`
      during build so JupyterHub users inherit the pickles from the
      root-owned package dir — no runtime download ever needed.
- [x] `README.md` "Tutorial data" section documents the lookup order.

### Verification

- [x] `python -m build --wheel` → 146 KB wheel, 24 files, 0 `.pkl`.
- [x] Live network check of `etho.download_tutorial_data()` against
      GitHub raw URLs — will be exercised by the Docker build.
- [x] Build the Docker image with
      `docker compose build` — in-flight.

### Ship checklist (session of 2026-04-21)

- [x] `pyproject.toml` bumped 2.0.4 → 2.0.5.
- [x] Ruff + black both clean on `src/` and `tests/`.
- [x] Bookstack "Getting started" page (book 1 / page 1) updated with
      new "Tutorial data" section; dependency list refreshed; editor
      switched from `wysiwyg` → `markdown` as a side effect of using
      the markdown field.
- [x] GitHub issue #7 opened and auto-closed by the commit footer.
- [x] Commit `c858b84` pushed to `main` (13 files, +593 / −37).
- [x] Tag `v2.0.5` pushed.
- [x] GitHub release v2.0.5 published.
- [x] CI `release.yml` failed on both tag-push and release events —
      **pre-existing** `tests/test_load_optimizations.py` failures
      (`No date_time found in METADATA table`), same pattern as 2.0.4.
      Unrelated to this change.
- [x] Manual `twine upload` to PyPI succeeded: 2.0.5 wheel (147 KB)
      and sdist (14 MB) live at
      <https://pypi.org/project/ethoscopy/2.0.5/>.
- [ ] Docker image `ggilestro/ethoscope-lab:1.2` build + push.

### Side debt spotted during this session

- `tests/test_load_optimizations.py::TestLoadOptimizationPerformance`
  (`test_load_ethoscope_memory_usage`, `test_connection_caching_benefit`)
  fails on GitHub Actions for both 2.0.4 and 2.0.5 tag / release runs.
  Raises `ValueError: No date_time found in METADATA table` from
  `load.py:524`. The test fixture likely builds a SQLite DB without the
  METADATA row that `load_ethoscope` now requires after
  a6a1473 (`harden load_ethoscope_metadata against firmware quirks`).
  Fix: update the test fixture to seed a `date_time` value in METADATA.
- `.github/workflows/release.yml` also runs broader-than-unit tests on
  tag push (`pytest tests/ -v --cov=ethoscopy -m "not slow"`), which is
  stricter than `ci.yml`. This is why the `test` job blocks automated
  PyPI publish. Either tighten the selection to `-m unit` or fix the
  underlying test.
- `.gitignore`'s blanket `*.pkl` is still present. It's currently
  harmless because `pyproject.toml` has an explicit wheel exclusion,
  but a future maintainer unaware of the wheel-build behaviour could
  be surprised.

### Discovered during work

- `.gitignore` has a blanket `*.pkl` rule that's also generating noise (it
  affects the behaviour of `git add` for tutorial pickles and the Hatchling
  wheel). The explicit `exclude` in `pyproject.toml` now removes any ambiguity
  about packaging, but the `.gitignore` rule could be tightened to a
  user-output pattern (e.g. `tutorial_dataframe.pkl`) in a future pass.
- The project-local `.venv` had a stale shebang (pointing to
  `/home/gg/Data/...`) and had to be rebuilt.

### Review

- Net diff intentionally small: one new helper (~40 lines), one packaging
  guard, one README section, one notebook preamble (templated, six files).
- No behaviour change for users who already have the pickles on disk; the
  error path now self-documents recovery.
- PyPI release 2.0.5 should carry these changes — the 2.0.4 wheel remains
  broken for new installers until then.

## 2026-08-17 — DIAGNOSTICS support + cache-key correctness

**Context.** Alice's `2026_08_13_222_fix_v2` experiment: four ethoscopes, 20 tubes
each, ~95 h. Two machines (017, 018) ran the firmware carrying the "222 fix",
two (001, 039) did not. Only the fixed firmware writes the new device-level
`DIAGNOSTICS` table (t, fps, image_noise, sharpness, jitter, n_rois_sampled,
cpu_temp, frame_noise), so ethoscopy had no way to read it.

### Changes

- [x] `load.py`: add `load_ethoscope_diagnostics()` — reads the per-device
      DIAGNOSTICS table, one row per sample tagged with machine_id/machine_name/
      date, t in seconds, honouring min_time/max_time/reference_hour. Databases
      whose firmware never wrote the table are skipped with a RuntimeWarning
      instead of aborting the load (mixed-firmware experiments are the norm).
- [x] `load.py`: extract `_rebase_time()` — the ms→s + reference_hour block was
      duplicated verbatim in `read_single_roi` and `read_single_roi_optimized`.
- [x] `load.py`: extract `_one_row_per_database()` — device-level tables must be
      read once per recording, not once per ROI.
- [x] `load.py`: **bug fix** — `_cache_path()` now keys the ROI cache on
      `reference_hour`, `min_time` and `max_time` as well as machine/ROI/date.
      Previously loading the same ROI with a different reference hour or time
      window silently returned the *first* cached frame, i.e. wrong timestamps
      or a short frame with no warning. Default arguments reproduce the historic
      filename so existing caches stay valid.
- [x] `__init__.py`: export `load_ethoscope_diagnostics`.
- [x] `tests/test_load_diagnostics.py`: 15 tests (real SQLite files, not mocks)
      covering expected use, mixed firmware, time windows, reference hour,
      unreadable files, and cache-key separation. Suite: 245 passed, 11 skipped.

### Discovered During Work

- `sleep_annotation()` sets `moving = False` on interpolated (untracked) bins but
  leaves `micro` and `walk` as **NaN**. A plain `.mean()` on `micro` is therefore
  conditioned on tracked bins only. For dead flies, 68–86 % of bins are
  interpolated, so this inflates apparent micromovement ~5-fold and can make
  `micro` exceed `moving`, which is impossible by construction. Worth deciding
  whether `sleep_annotation` should fill these consistently.
- Analysis scripts live outside the repo in `~/Downloads/Alice/analysis/`.

### Issue #222 field test — result

Alice's experiment is a field test of gilestrolab/ethoscope#222 (FPS ceiling
caps exposure -> sensor noise -> spurious max-velocity spikes -> lost sleep).
017/018 run commit dc8ee5f with `exposure_decoupled = True`; 001/039 run older
and *different* commits. Findings, in the order they were established:

- Sleep 34.3 % (unpatched) -> 40.2 % (patched); bout-length distributions
  identical, so it is scoring, not behaviour.
- Per-frame p50 matches across arms (-2674 / -2679) exactly as the issue says,
  but the p99 tail is *heavier* on the patched machines. Pooling active and
  quiet frames makes this statistic uninterpretable.
- Conditioned on quiet bins (median per-frame displacement at the noise floor),
  the false-positive rate halves in the siesta: 21.3 % -> 11.9 %.
- Per-fly false-positive rate predicts per-fly siesta sleep at r = -0.84, and
  -0.82..-0.91 *within* each machine. Partly structural (both use the same
  `max_velocity > threshold`), so it evidences sensitivity, not causation.
- Decisive statistic: ratio of "just over threshold" (-2523..-2000, noise-like)
  to "well over" (> -2000, real movement) separates the arms with no overlap —
  2.27 / 1.70 unpatched vs 1.02 / 0.96 patched.
- `frame_noise` in the light phase is 0.64 (E017) and 0.60 (E018), matching the
  0.63 the issue quotes for a well-exposed machine. Unmeasurable on the
  unpatched arm — DIAGNOSTICS ships only with the firmware under test.

Caveats: 2 machines/arm, firmware confounded with machine, unpatched pair not on
matched commits, group separation leans heavily on ETHOSCOPE_001.

**Dead flies do not work as a noise floor.** `AdaptiveBGModel` absorbs a
motionless object into the background, so 68-86 % of dead-fly bins carry no
tracking data and the survivors are a biased sample of tracker failures. Rates
ranged 4.7-81.9 % within one firmware group. Do not design future controls
around dead animals.

### 2025 legacy controls added

Three 2025-03-24 databases (ETHOSCOPE_004, _007, _014; 14 flies) added as a
pre-regression reference. They run `OurPiCameraAsync` on the **pre-libcamera
picamera stack** (kernel 6.1.14, no picamera2), so the `FrameRate` ->
`FrameDurationLimits` -> `ExposureTime` coupling that #222 describes does not
exist there. Truncated to the first 96 h to match the 2026 window.

Their firmware records no lighting info and no DIAGNOSTICS, so the light cycle
was recovered from `IMG_SNAPSHOTS` mean luminance — validated by re-deriving the
2026 cycle from snapshots (08:55-08:57) against the frame_noise answer (08:55).
Legacy is 09:03 UTC on, 21:03 off, 12:12 LD.

**Headline: the fix restores the pre-regression baseline.** Siesta
false-positive rate: legacy 12.2 %, patched 11.9 %, unpatched 21.3 %.

**Two corrections to the earlier conclusions:**

1. The threshold-band *ratio* (just-over / well-over) does NOT transfer across
   cohorts. Legacy scores 1.65, with the unpatched pair, despite having the
   lowest noise numerator of any arm. The denominator tracks genuine locomotor
   activity, which differs between fly cohorts. Valid within one experiment
   only; the numerator alone is the transferable part (legacy 7.8 % < fix 9.1 %
   < no_fix 11.0 %).
2. The false-positive effect was overstated. The arms sample unequally — 47.4
   frames/10 s bin unpatched vs 31.2 patched — and a bin fails if *any* frame
   crosses threshold. Subsampling all arms to 25 frames/bin (real subsampling;
   the binomial 1-(1-p)^n model overestimates badly because noisy frames are
   temporally correlated) gives 18.2 % vs 11.5 %. So a 37 % reduction, not 44 %.

The residual sampling effect is a *second* causal pathway, not an artefact:
decoupling exposure lowers the frame rate (unpatched pinned at 4.96 fps = its
5 fps ceiling; patched 4.06; legacy 3.30), and fewer frames per bin genuinely
means fewer false positives — but also less sensitivity to real brief movements.
Issue option 1 (throttle CPU in software, not via sensor FrameRate) would
separate the two.

Motion blur ruled out as an explanation for the patched arm's heavier
"well over" tail: tracked blob area is slightly *smaller* during movement than
at rest, by the same factor in every arm (0.89 / 0.91 / 0.92).

Suggested for the ethoscope side: log `ExposureTime` into DIAGNOSTICS — every
exposure inference here uses achieved frame rate as a proxy.

### Decimation study (the test unsolved.md called decisive)

Run as `12_decimation.py`: frames dropped from already-recorded runs, then
rescored. Changes sampling without touching image quality, so it separates the
fix's two pathways.

| arm | native | 3.5 Hz | 3.0 Hz |
|---|---|---|---|
| unpatched | 38.4 % | 39.8 % | 40.6 % |
| fix 222 | 51.1 % | 51.4 % | 51.8 % |
| difference | +12.7 | +11.6 | +11.3 |

**~9 % of the sleep effect is the lower frame rate; ~91 % is image quality.**
Supersedes the earlier within-bin estimate of "about a fifth" — that was a
cruder proxy for the same question.

Report: analysis scripts + published report in `~/Downloads/Alice/analysis/`.

### Potential Agents

- None proposed.

## 2026-09-17 — Land PR #12 (Kaplan–Meier survival + plot_sleep_bouts) on branch `pr12-survival`

**Context.** PR #12 (Lei Guo) merges cleanly but changes results of existing
functions and has correctness problems (see review in session). Decision:
Option A — fix on our own branch, merge with a merge commit so the author's
commits survive, then close the PR pointing at the merge.

**Hard constraint: no behaviour change for existing users** — `baseline()`
and `curate_dead_animals*()` must give byte-identical results to `main` with
default arguments. New behaviour only behind new, opt-in parameters.

### Plan
- [x] `baseline()`: keep vectorised version but restore main's dtype semantics
      (integer `t` stays integer when no specimen is shifted).
- [x] `_wrapped_curate_dead_animals`: revert the hardcoded 75 %/10 s coverage
      rule. Add opt-in `min_coverage: float | None = None` to
      `curate_dead_animals()` / `curate_dead_animals_interactions()`; expected
      samples per window derived from the specimen's own median sampling
      interval, never a hardcoded 10 s.
- [x] New module `src/ethoscopy/survival.py` (pure functions, numpy): sliding
      window death detection (shared with curate), zero-run detection,
      Kaplan–Meier with Greenwood CI, per-subject survival table.
- [x] Subject identity: default one subject per `id` (no merging). Optional
      `subject_cols=[...]` merges recording segments across restarts using
      metadata columns (e.g. `machine_name`, `region_id`), instead of parsing
      the id string. Merging requires non-overlapping `t` → raise otherwise.
- [x] Drop `count_unique_flies` and the "def1/def2 n" heuristic (only needed
      because identity was parsed wrongly); drop dead `_wrapped_km_death_detect`.
- [x] `facet_col` taken from metadata via `_check_lists` (no hardcoded
      `species`); colours via `_get_colours`/`_check_grey` like `survival_plot`.
- [x] Shared `_km_curves()` in core; thin `km_survival_plot()` per backend.
- [x] `plot_sleep_bouts`: reuse `sleep_bout_analysis(as_hist=True)`; drop the
      unsubstantiated "2.2.0 constructor bug" workaround after testing single-id.
- [x] Google-style docstrings, no in-function imports, no magic numbers.
- [x] Tests: `tests/test_survival.py` (KM against hand-computed values, table,
      merging, coverage opt-in, curate identical to main on tutorial data,
      baseline dtype), plot smoke tests both canvases.
- [x] Verify: full suite green; diff curate/baseline results vs `main` on
      `overview` tutorial data.

### Review

**Retrocompatibility, verified rather than asserted.** A fingerprint harness ran
`curate_dead_animals()` (default, `time_window=12`, `prop_immobile=0.05`) and
`baseline()` on `main` and on this branch and compared shape, dtypes and a hash
of every row. Identical on the real `overview` tutorial dataset (327,031 rows)
and on synthetic data at 10 s, 60 s and 300 s sampling, with gaps, with and
without deaths, for all-zero, numeric and string baseline columns, and for
unsorted input.

**What changed relative to PR #12**

- The 75 % coverage rule no longer applies by default. It is the opt-in
  `min_coverage` argument on both curate methods, and expected samples come from
  each specimen's own median sampling interval instead of a hardcoded 10 s. As
  written in the PR it silently disabled death detection for anything not
  sampled at 10 s.
- `baseline()` keeps the vectorised shift but reproduces the previous dtype and
  ordering exactly: an all-integer shift keeps an integer `t`, and data that did
  not arrive sorted by id is still sorted, which `groupby` used to do as a side
  effect.
- Animal identity comes from metadata via `subject_cols`, not from parsing the id
  string. The PR's parser extracted a file-name hash as the "machine number" and
  merged different experiments that happened to share a machine and ROI. Merging
  now raises if the sessions overlap in time, since it needs a shared clock.
- Death detection, the Kaplan-Meier estimator and the survival table moved to
  `src/ethoscopy/survival.py` as pure functions, so `curate_dead_animals()` and
  the survival table share one definition of death.
- Dropped `count_unique_flies()` and the "def1 vs def2" n heuristic, which only
  existed to paper over the identity bug, and the unused `_wrapped_km_death_detect`.
- `facet_col` now comes from the metadata through `_check_lists`, so faceting
  works on any column rather than only `species`. Colours come from
  `_get_colours`/`_check_grey` like every other plot.
- Plot bodies are thin: `_km_curves()` and `_km_steps()` in core, and
  `_bout_hist_data()` in `behavpy_draw`, are shared by both backends.
- The "2.2.0 internal constructor bug" workaround is gone.
  `sleep_bout_analysis(as_hist=True)` was tested with single-specimen data and
  works, so `plot_sleep_bouts()` calls the public method.
- Bar width in `plot_sleep_bouts()` now scales with `bin_size`; the previous
  default drew hairlines for any bin size above 1 minute. The plotly backend
  groups bars like seaborn instead of hiding one group behind the other.

**Verification.** 315 tests pass, 11 skipped. `tests/test_survival.py` adds 39,
including the Kaplan-Meier estimate and Greenwood band against hand-computed
values, death detection at three sampling intervals, `min_coverage` being
opt-in, session merging and its overlap error, and plot smoke tests on both
canvases. All four figures were rendered and inspected.

**Left alone.** The three plot files remain far over the 500-line guideline;
they were already 3,000-4,000 lines before this work and splitting them is a
separate job. Ruff reports the same error categories as `main` (`Optional[...]`
style), since the new code follows the conventions of the files it sits in.

### Discovered During Work

- `pytest.ini` still declares `[tool:pytest]`, so markers such as
  `@pytest.mark.unit` are unregistered and every test file emits
  `PytestUnknownMarkWarning`. Renaming the section switches on
  `--cov-fail-under=70` at the same time, so it needs checking separately.

## 2026-09-28 — Noise-calibrated movement threshold (auto threshold)

**Problem.** Movement is scored as `max_velocity > 1.0`, a constant tuned for one
imaging setup. Alice's DBS experiment (`~/Downloads/Alice_DBS_092026`) exposed it:
the same species gave 55% night sleep under the old IR light and 14% under a new
double-band IR. Analysis showed neither number is right:
- Per observed still bin (position unchanged ±0.5 px over ~1 min), 15–36% of bins
  cross 1.0 in *both* setups — tracker jitter / one-frame snap-back glitches.
- Sleep needs 30 consecutive clean bins, so a p=15% false-positive rate leaves
  (0.85)^30 ≈ 0.8% of true rests intact. Sleep is dominated by the FP rate.
- The old IR only looked better because it *lost* still flies (21% of quiet night
  time untracked) and `sleep_annotation` scores untracked bins as immobile.
- Prototype: per-fly, per-light-phase threshold = 99th percentile of max_velocity in
  still bins (floored at 1.0) → night sleep 83% vs 84%, day 70% vs 66%; FP fixed at
  1% by construction. Prototype scripts: session scratchpad (`adapt.py`, `evalm.py`).

**Design.** Threshold becomes a property estimated from each fly's own noise, per
light phase, instead of a constant. "Still" bins are identified *independently of
velocity* by positional stability, so the estimate is not circular.

### Plan
- [x] New module `src/ethoscopy/motion_calibration.py`: `find_still_bins`,
      `estimate_velocity_threshold` (per light phase, fallback recording → floor with
      warning), `find_spikes`, `motion_qc`, `light_phase`, `default_still_shift`.
- [x] `max_velocity_detector(velocity_threshold="auto", threshold_quantile, threshold_floor,
      day_length, lights_off, remove_spikes)`; `velocity_threshold` column with "auto";
      micro/walk stay a partition of moving when the threshold exceeds walk_threshold.
- [x] `sleep_annotation(velocity_threshold=, untracked="immobile"|"break")`.
- [x] behavpy `motion_detector` / `sleep_contiguous`: same parameters.
- [x] `motion_qc(...)`: thresholds, untracked and spike fraction, FP rate at 1.0, rest survival.
- [x] Tests `tests/test_motion_calibration.py` (24).
- [x] `scripts/validate_motion_threshold.py` for the ground-truth nights (dead flies).
- [x] Port to rethomics `sleepr` (`~/Code/ethoscope_project/rethomics/sleepr`, cloned
      from rethomics/sleepr): `R/motion-calibration.R`, detector + `sleep_annotation`
      options, roxygen docs, `tests/testthat/test-motion_calibration.R` incl. parity
      fixtures exported from ethoscopy (binned + frame level).
- [x] README section.
- [ ] Ground-truth recordings (lab): dead/anaesthetised flies under each IR → run
      `scripts/validate_motion_threshold.py`.
- [ ] Decide default switch after ground truth (currently opt-in).
- [x] Still-bin tolerance default 0.5 px -> 1 px (Giorgio, 2026-09-30), both packages:
      `STILL_SHIFT_PIXELS`/`pixel_size()` (Python), `pixel_size()` (R, exported);
      spike rule restated in whole pixels (3 px jump, 1 px return; unchanged).
      Parity fixture now has sub-pixel jitter so it distinguishes 0.5 vs 1 px.
      Alice re-validation: night sleep old/new IR 88.0% / 87.2%; one old-IR fly
      night threshold 7.4 (outlier, not inspected).
- [x] Tracker issues (never-detected still flies, fragment tracking) and the
      SQLite end-of-run data loss sent to the ethoscope session (ethoscope-63).
- [x] Real-time sleep deprivation: replayed InactivityTrigger on ETHOSCOPE_044 (47 h)
      against pixel truth. Per-frame v>1.0 (device today): 79% of true >=2.5-min still
      periods stimulated, 16% of stimuli after real movement. Per-10-s-window v>1.0: 89%,
      4.6%. Giorgio approved per-window (2026-09-30); spec sent to ethoscope-63.
      Online calibration deferred (mixed benefit in real time).
      CORRECTION (ethoscope-63, confirmed): at the same threshold per-window sees the
      same crossings as per-frame; per-frame with 130 s scores identically (89.0%,
      5.4%). The "gain" is only ~10 s later firing. Remaining case for per-window:
      one definition shared with post-hoc scoring. Excluding is_inferred rows (2%)
      changed nothing. ethoscope-63 has it implemented, uncommitted, awaiting Giorgio.
- [ ] Replay the per-window rule on Giorgio's overnight recording (started 2026-09-30,
      lab-standard milder IR) once it is available.
- [ ] Human annotation of micro-movements on current hardware (thesis protocol;
      flyscorer on turing) to decide which movements should break sleep.

### Design decisions taken
- Opt-in `velocity_threshold="auto"` (Giorgio, 2026-09-28); 1.0 stays default.
- Still bin = before/after 30-s median positions agree AND trimmed range (2nd-lowest
  to 2nd-highest of the 7-bin window) ≤ ½ px. Medians alone were fooled by pacing
  (alternating bin means) — a walking fly scored as still.
- Spike removal is required for "auto": 1-frame jumps of 12–69 px that land back on the
  same pixel hit up to 4% of a still fly's bins and pushed q99 to 55. Enabled with
  "auto" by default, opt-in (`remove_spikes=True`) for the fixed threshold.
- Quantile/median arithmetic is linear (numpy default = R type 7) in both packages,
  vectorised (`_row_quantile` / `row_quantile`) — nanquantile/apply were ~100x slower.

### Results
- Alice DBS, days 2–3, final code: night sleep fixed 1.0 → old IR 56% / new IR 14%;
  auto → 86.3% / 86.2%. Day: 5%/2% → 68%/61%. Auto thresholds night median 1.41/1.53,
  day 1.88/1.91; one old-IR fly reaches 5.6 at night (worth inspecting).
  motion_qc: FP at 1.0 on still bins 14–33%, 5-min rest survival < 1%.
- R vs Python on 4 real flies: thresholds identical or within 0.2–4% (pre-existing
  detector difference, see below).

### Validation against tracker-independent pixel motion (2026-09-30)
Ground truth: pixel changes > 20 grey levels (~14x empty-tube noise) in a crop around
the fly; validated on 5 dead flies (0.0% of frames), offline tracking with the
device's AdaptiveBGModel. Scripts + data on turing: /mnt/cache/claude_motion_calibration.
- Alice ETHOSCOPE_012 (1 h, 15->3.75 fps, new IR), dark phase: fixed 1.0 FP 11.3%;
  auto 0.5 px FP 4.3% (threshold 0.83x oracle); auto 1 px FP 0.9% (1.02x oracle).
- ETHOSCOPE_044 (2024, 47 h, 6.2 fps, dim IR, 2 nights, 20 flies, 103 fly-blocks):
  fixed 1.0 FP 7.7% dark / 12.8% light; auto 1 px FP 1.0% / 1.4%, threshold 1.00x
  oracle (spearman 0.60); auto 0.5 px 0.97x. => default max_shift 1 px is supported.
- Sleep amount depends on what movement breaks sleep: pixel-truth night sleep is 29%
  if any single-frame twitch counts, 62% if >= 8 active frames per 10 s; auto 1 px
  gives 63%. This is a definition, to be set with human-annotated micro-movements.
- Light phase in Alice's video: partial-fly (fragment) tracking by AdaptiveBGModel
  gives phantom 5-30 px jumps; not fixable by thresholds.
- Dead/still-from-start flies are never detected by AdaptiveBGModel.

### Related work handed to other sessions (2026-09-30)
- ethoscope-63 (jenner, ethoscope repo): SQLite end-of-run data loss FIXED (dev e7e0f8ca);
  windowed sleep-dep trigger implemented, uncommitted, awaiting Giorgio (gain is only
  ~10 s later firing, see correction above); still-fly seeding (#2) and second-blob
  hopping (#3) diagnosed; GPIO listener busy loop (100% of a core on every device).
- ethoscope_DL_tracking (turing): learned per-tube fly tracker, plan sent; Pi 3
  feasibility measured on ETHOSCOPE000 (tiny CNN 44.5 ms/20 tubes; live 9.1 fps).

### Discovered During Work
- `beam_cross` compares `x` to 0.5, but `x` arrives in pixels from load_ethoscope
  — likely never fires correctly; check.
- `masking_duration` masks only beam_cross, not `moving`: a stimulator pulse can
  still register as movement.
- Per-frame distance is fps-dependent (velocity = dist/coef, no dt): runs at 2.8
  vs 4 fps are scored on different scales. Auto threshold absorbs the noise part.
- `load_ethoscope` (read_single_roi_optimized) returns x/y in **pixels**; the legacy
  `read_single_roi` divides distance variables by ROI width. beam_cross (x vs 0.5) and
  `feeding(dist_from_food=0.05)` assume normalised x → both likely broken on data
  loaded with load_ethoscope.
- sleepr's detector drops the first frame of each window from max_velocity
  (`velocity_corrected[2:.N]`) and zeroes velocity during masking; ethoscopy keeps it
  and masks only beam_cross. Explains the 0–4% R/Python threshold gap; pre-existing.
- analyse.py is 800+ lines (limit 500); split motion detection into its own module.
- ~/R system library has stale compiled packages (stringi vs ICU 78, vctrs vs R);
  rebuilt stringi, vctrs, purrr in ~/R/library.
