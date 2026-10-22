# Changelog

## Unreleased

### Fixed
- `load_ethoscope_light_schedule` returned the recorded `lights_on`, a local clock
  time, as `reference_hour`, while `load_ethoscope` counts from the recording's start
  in UTC. In British Summer Time this put ZT0 an hour late. `reference_hour` is now
  UTC: the offset is read off the light switches in `LIGHT_EVENTS`, so 09:00 BST
  gives 8.0. A new `utc_offset_h` column reports the offset used. Recordings that span
  a clock change raise a warning.
- `load_ethoscope(cache=...)` discarded every ROI when the cache directory did not
  exist or could not be written to. The directory is now created as needed. If the
  write still fails, the data load uncached and a single warning is raised. Cache
  files are written atomically, so a reader sharing the cache never sees a partial
  pickle.

### Added
- `timezone=` in `load_ethoscope_light_schedule`, an IANA name such as
  `"Europe/London"`, for firmware without `LIGHT_EVENTS` (before August 2026). Without
  either, `lights_on` is returned unconverted (`source="recorded"`), with a warning.

### Changed
- The docstrings of `load_ethoscope`, `load_ethoscope_diagnostics` and
  `read_single_roi` now say that `reference_hour` is a UTC hour.

## 3.0.0 (2026-10-07)

**Breaking: `sleep_annotation` needs a sleep rule.** To reproduce earlier results, add
`rule="classic"` to your calls, or declare it once at the top of the notebook with
`etho.set_sleep_rule("classic")`, or set the environment variable
`ETHOSCOPY_SLEEP_RULE=classic` to re-run old notebooks unchanged. With
`rule="classic"` the output is byte-identical to 2.4.0. Without any rule,
`sleep_annotation` raises an error explaining the choice. See "Choosing a sleep rule"
in the README.

### Added
- `rule="k"`: sleep scored from walking and sustained movement events, ignoring
  tracking noise (flickers, sub-pixel jitter and isolated micro-movements). `k=3` by
  default; `"k2"` is stricter. It matches video ground truth at night, and flies it
  scores asleep respond to air puffs like sleeping flies. Recommended for new analyses,
  especially on DeepTubeTracker or exposure-first recordings.
- `set_sleep_rule()` / `get_sleep_rule()` and `ETHOSCOPY_SLEEP_RULE`: declare the rule
  once, as with matplotlib's rcParams. An explicit argument always wins.
- `untracked=` in `sleep_annotation` and `sleep_contiguous`: `"immobile"` (default, as
  before) or `"break"`, which stops windows without tracked data from counting as sleep.
- `motion_qc()`: per fly and light phase, how often a still fly crosses the classic
  threshold, and how much of the recording is untracked.
- `scripts/validate_k_rule.py`: parity of the k-rule with the reference implementation.

### Fixed
- `plot_hmm_response` (plotly and seaborn) failed on every call since 2.0.1; faceted
  plots also failed in `facet_merge`.
- `load_ethoscope` no longer turns integer columns into object dtype when one ROI has
  no rows in the requested time window.
