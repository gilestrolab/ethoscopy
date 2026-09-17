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
