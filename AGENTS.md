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

**Latest status (2026-09-29 UTC): v1 is DONE.** EXP-007 and EXP-009 official-test evaluations are complete; the official test is no longer unscored. Preserve both completed artifact trees, manifests, response/attempt histories, reservation ledgers and source-compatibility receipts. Do not resume completed runs, reset attempts, rerun inference or tune thresholds, prompts, retrieval, classifiers or models from test outcomes. Finalization uses saved evidence and offline checks only.

EXP-007's five-seed mean specialist/hybrid accuracy is **85.435065% / 86.487013%**, versus zero-shot Luna **81.363636%**, at **9.012987% mean fallback**. Five seeds share one 3,080-case test population and one Luna set; seed SD is not a confidence interval. See `experiments/exp007-official-test/`.

EXP-009 is a **single seed-11 / single training-pool companion**, not a five-seed result. Retrieved hybrid accuracy/macro-F1 is **88.084416% / 87.982622%**, using Luna for **268/3,080 cases (8.701299%)**, versus matching specialist **85.584416% / 85.347125%**. All-request retrieved Luna is higher at **92.175325% / 92.134813%**. Compare with matching seed-11 controls, not five-seed means. All 3,080 final retrieved outputs are OK; usage-priced charges are **$0.576343155**, bounded by **$0.576343155–$0.584791155** with three unknown-usage attempts. Actual invoice spend is unknown/unreconciled. See `experiments/exp009-official-test/`.

Each 20-shot model uses **1,540 task-training labels plus 770 reused development labels**, with external MiniLM pretraining. Retrieval uses only that same seed-11 pool. Both API studies collected all test cases; routed call fractions describe policy replay, not actual study call counts. This is a quality-versus-LLM-usage tradeoff, not a production-dollar-savings claim or evidence that Luna is universally stronger. Specialist deployment cost and total-system savings remain unmeasured.

Frozen protocols remain EXP-007 `0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00` and EXP-009 `b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334`. The independent scalar rule remains full-precision `confidence >= threshold`; no ranking or forced coverage. Historical pending/resume entries describe their dates and do not authorize another execution.

EXP-006R recovery and merged validation evaluation are complete, with original failures preserved. Validation findings remain separate from official-test findings. EXP-008 CPU timing is an existing validation-input measurement, not a production cost estimate. Keep raw customer/test text, request/response bodies, provider request IDs, secrets, weights and caches out of Git; commit only compact verified evidence.

**V2 is deferred:** calibration; SetFit/fine-tuned specialist; CLINC150/second dataset; self-hosted vLLM/H100; production cost study. Do not begin these, a paid pilot, further inference/training/test preflight, frontend or infrastructure without a new user task.

Use `uv sync --locked --cache-dir .cache/uv`, then `.venv/bin/python -m pytest -q`. CLI commands are documented in `README.md`. Commit only source, lockfiles, documentation, and compact evidence; raw data and large generated artifacts stay ignored. Confirm the Git top-level is this project before Git writes or recording provenance.
