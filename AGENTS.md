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

M1a is implemented and tested; one N=5/seed=11 validation smoke experiment has run. It is not an independently reproduced final result. Read `docs/CURRENT_PLAN.md` for the next experiment (N=10/seed=11, not yet run) under protocol `banking77-val10-v2`: fixed validation=10 per class and training N=5/10/20 across all 77 classes. N=50 is unsupported. Keep the legacy validation=20 smoke record separate; its scores cannot be pooled with the new protocol. Never oversample, drop classes, or reuse test data to disguise this limit. Strong-model calls, calibration, routing, deployment, and UI remain later work.

Use `uv sync --locked --cache-dir .cache/uv`, then `.venv/bin/python -m pytest -q`. CLI commands are documented in `README.md`. Commit only source, lockfiles, documentation, and compact evidence; raw data and large generated artifacts stay ignored. Confirm the Git top-level is this project before Git writes or recording provenance.
