# Lessons

## 2026-09-28 — noise-calibrated threshold work

- **Format only the files you changed.** `black src/ethoscopy/` reformatted the
  untouched `load.py`. Pass explicit paths: `black <file> <file>`.
- **Scripted text edits: replace exactly once.** Python `str.replace(old, new)` hits
  every occurrence; an anchor like `n_still = sum(mask),` existed in two functions and
  a line landed in the wrong one. Use `str.replace(old, new, 1)` after
  `assert s.count(old) == 1`.
- **Don't filter output with `grep -v "^  "`.** pandas prints index-continuation rows
  with leading spaces; the filter silently removed half a results table and looked
  like a bug. Send warnings to `2>/dev/null` or `python -W ignore` instead.
- **Don't hide stderr on a script whose output you depend on.** A failed export
  under `2>/dev/null` surfaced later as a confusing missing-file error.
- **Robust statistics can be fooled by structure.** A median/MAD over a window
  tolerates up to 50% outliers, so a 4:3 alternating pattern (pacing) reads as
  stillness. When "robust to one glitch" is the goal, trim exactly one value per end.
- **Check the tails, not just medians.** Median thresholds on real data looked right
  while a few flies had q99 = 55 from tracker spikes; the smoke run's per-fly max
  exposed it.
- **Seeded thresholds in tests must reflect the design, not one seed's luck.**
  A 1% FP target gives rest sleep of 0.90–0.98 depending on where FPs land; assert
  the FP rate itself and a bound valid across seeds.
- **Measure positions in figures; don't eyeball them.** Reading a montage, I forgot
  the 70-px label column and "saw" a 90-px walk that the data showed was a still fly.
  Before claiming movement from an image, confirm it from the tracked coordinates.
- **`pgrep -f PATTERN` matches the shell running it** whenever the pattern appears in
  that shell's own command line, in wait loops and in one-off checks over ssh alike
  (it happened twice). Find a service's process with
  `systemctl show -p MainPID --value UNIT`, or wait on a log line.
- **Unsilence warnings when a result looks off.** `warnings.simplefilter('ignore')` hid
  that "auto" had fallen back to the floor on a 1-h video, which made it look inert.
- **Before crediting a mechanism, test the simplest alternative.** I called per-window
  judging "the clearest win" for sleep deprivation; the gain was just firing ~10 s
  later (per-frame at 130 s scored identically). A metric with a hard boundary (fires
  "within 120 s of movement") rewards any rule that fires later. Replay the baseline
  with the obvious knob changed before recommending a new rule.
- **Long remote jobs: pull results home as they come, and don't trust one waiter.**
  A 5-h soak test on a Pi logged to RAM-backed /tmp; my single background waiter
  (capped at 1 h) ended silently, and the Pi was unreachable two days later, so the
  data were lost. Copy partial results off the device periodically (or write them
  to persistent storage), and re-arm or check long waits explicitly.
- **A file you changed is not necessarily black-clean.** `black load.py` after a 12-line
  fix reflowed three untouched functions (black 26 against code formatted by an older
  black). Run `black --check --diff FILE` first; if it would touch more than your hunk,
  leave the file alone and keep your hunk in its style by hand.
