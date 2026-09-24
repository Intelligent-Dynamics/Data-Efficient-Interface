# Data-Efficient Specialist Inference

An Intelligent Dynamics research project. Operating principle: **Measure → understand → improve → re-measure.**

## Start each session

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

Planning is complete; the next milestone is a local BANKING77 few-shot TF-IDF + logistic-regression baseline, specified in `docs/CURRENT_PLAN.md`. Strong-model calls, confidence routing, deployment, and UI are later milestones. No experiments have run as of initialization on 2026-09-24.
