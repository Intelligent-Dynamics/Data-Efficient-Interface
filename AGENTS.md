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

**Bounded v1 scope (2026-09-27) supersedes immediate EXP-007 live execution.** The user reports an earlier authorized mechanical test preflight; do not say the official test was never opened. Test predictions and scores remain unmeasured. No further test access, encoding, inference or scoring is authorized. Work is limited to existing train/validation data, the frozen seed-11 CPU benchmark (EXP-008), and the single retrieved-example Luna companion (EXP-009). No new fitting/calibration/dataset, paid pilot, GPU serving or frontend. Preserve original EXP-007 protocol bytes/hash. Companion approval is separate and cannot inherit EXP-007 or earlier dollar-cap discussions. Retrieve exactly 20 examples from only the existing 1,540 seed-11 training IDs; disclose all those labels plus 770 development labels. Describe complementary errors, quality and call fraction; no stronger-overall Luna or total-system-savings claim.

EXP-006R live recovery is complete (52/52, zero unresolved), and EXP-006 + EXP-006R offline evaluation is complete (770 resolved IDs, all 90 combinations). Verified compact evidence is in `experiments/exp006-completed-validation/`. Preserve the original EXP-006 status/history and every original/recovery artifact. EXP-007 fixed-threshold preparation is complete; **OFFICIAL TEST NOT RUN**. Use the five frozen 20-shot classifiers and per-seed thresholds in `experiments/exp007-fixed-threshold-preparation/protocol.json`. Route each request independently with unrounded binary64 confidence >= threshold; never rank test requests or force 90% coverage. Do not refit, recalibrate, change the prompt/schema/model, or tune on test outcomes. Test unsealing/evaluation requires separate explicit user authorization. Paid test collection additionally requires current pricing, approval of the new protocol and a numeric cap; earlier approvals do not transfer. No key or documented command is permission. Keep raw data/responses, model weights, credentials and large caches out of Git. Reuse one all-case Luna test response set across all five seeds. Preserve failures and report actual observed coverage/all-seed results. No production threshold guarantee, total-system savings or unmeasured specialist-cost/latency claim. No frontend or fine-tuning.

The guarded final-test runner is `baseline.final_test`: `dry-run` needs no authorization and stays sealed. `preflight` requires exact protocol approval and explicit test access only; it cannot perform inference/API calls, needs no cap or pricing acknowledgement, and may compare an optional proposed cap. `live` still requires exact protocol approval, explicit test/live authorization, a positive approved cap and current pricing acknowledgement; preflight grants none of those live permissions. See `experiments/exp007-runner-preparation/README.md`. Runner preparation is reviewed and included in this checkpoint; execution still requires separate authorization. Keep the single frozen output directory and intact caches/ledgers for resume; never start another directory to reset attempts. No test/result or paid collection occurred during runner implementation.

Use `uv sync --locked --cache-dir .cache/uv`, then `.venv/bin/python -m pytest -q`. CLI commands are documented in `README.md`. Commit only source, lockfiles, documentation, and compact evidence; raw data and large generated artifacts stay ignored. Confirm the Git top-level is this project before Git writes or recording provenance.
