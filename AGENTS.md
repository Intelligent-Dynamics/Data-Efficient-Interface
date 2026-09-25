# Data-Efficient Specialist Inference

An Intelligent Dynamics research project. Operating principle: **Measure → understand → improve → re-measure.**

## Start each session

Canonical checkout: `/Users/Andrew/Developer/data-efficient-inference`. Use this directory for all work; do not recreate the former Documents/ChatGPT path. Verify the Git top-level before making changes.

Read `docs/PROJECT_CONTEXT.md`, `docs/CURRENT_PLAN.md`, `docs/EXPERIMENTS.md`, and `docs/RESULTS.md` before making changes. Inspect the working tree and preserve existing work. Follow the current milestone; do not infer that the entire system should be built.

## Research rules

- Never fabricate metrics, coverage, cost savings, or completed experiments. Mark hypotheses, plans, externally reported facts, and our measurements distinctly.
- Start with simple baselines. No frontend, routing service, or unnecessary infrastructure at this stage.
- Keep test data sealed from feature fitting, tuning, calibration, prompt design, threshold selection, and qualitative development. Fit TF-IDF only on the sampled training texts.
- Count all labels used for training, validation, calibration, and prompt examples. An N-shot training claim must disclose additional development labels.
- Pin dataset revisions, dependencies, split/sample manifests, seeds, configuration, code revision, and commands. Retain predictions and artifacts sufficient to reproduce claims.
- Separate actual experiment spend, benchmark cost estimates, and projected deployment savings. Local inference is not automatically free. Never equate fallback reduction with cost reduction.
- Preserve failed, null, and negative experiments. Log deviations and limitations, not just successful runs.
- Put only verified, reproducible project results in `docs/RESULTS.md`; use `docs/EXPERIMENTS.md` for planned/running/failed/unverified work.
- Update project context and the current plan when major decisions change. Record the reason and date; do not silently rewrite experimental history.

## Current scope

EXP-004 is complete: the frozen MiniLM 5/10/20-shot × five-seed validation curve and 15 independent local re-encoding/refits all matched exactly. 64 tests pass; no failed model runs or warnings. EXP-001/002/003 and their implementations remain preserved. See `RESULTS.md` and the current plan; do not duplicate completed experiments. The proposed next milestone is paired error analysis from saved 20-shot predictions, with no new fitting. Reuse exact v2 IDs and pinned settings. Cached classifier-only timings are not end-to-end inference speed. No test evaluation, fine-tuning, routing, paid APIs, or UI.

Use `uv sync --locked --cache-dir .cache/uv`, then `.venv/bin/python -m pytest -q`. CLI commands are documented in `README.md`. Commit only source, lockfiles, documentation, and compact evidence; raw data and large generated artifacts stay ignored. Confirm the Git top-level is this project before Git writes or recording provenance.
