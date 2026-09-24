# EXP-001 smoke evidence

Single N=5/seed=11 validation pipeline check, not a final project result. See [the experiment log](../../docs/EXPERIMENTS.md).

- `metadata.json`: configuration, environment, label access, hashes, provenance, timing, and completion status.
- `metrics.json`: validation metrics for the lexical specialist and dummy comparator.
- `samples.json`: exact training/validation row IDs and ordered labels.
- `predictions.json`: validation labels and both models' predictions, sufficient to recompute quality metrics.
- `protocol.md`: the exact plan snapshot taken before the run.

Raw data and larger artifacts are excluded from Git. `metadata.json` also hashes local-only files; those hashes are provenance, not a claim that every artifact is stored in this directory. Full artifacts and source snapshot remain at `artifacts/exp001-smoke-n5-s11/` on the originating machine. Regenerate into a new directory using the README's setup/prepare commands and:

```sh
.venv/bin/python -m baseline run --shots 5 --seed 11 --output artifacts/exp001-smoke-n5-s11-reproduction
```

This reproduction has **not** been run. Source hashes identify the pre-commit implementation; post-run documentation changes are expected. Timing and serialized-model hashes may differ between environments even when samples, predicted labels, and metrics agree. BANKING77 attribution and license are in the repository README and source manifest.


Protocol note (2026-09-24): this record uses the legacy 20-validation-examples-per-class split. Reproduction requires code at `717b7e6` or the saved source snapshot. Current CLI defaults use `banking77-val10-v2` (10 validation examples/class) and **do not reproduce this run**. The legacy metrics, sampled IDs, metadata, and frozen protocol remain unchanged; do not aggregate this score with v2 scores.
