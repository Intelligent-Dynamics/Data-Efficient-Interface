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

**Latest status (2026-09-28 UTC): EXP-007 official-test evaluation is complete.** All 3,080 Luna outputs resolved in 3,081 attempts; one HTTP-503 attempt lacks usage. Verified compact evidence is in `experiments/exp007-official-test/`. Mean specialist/hybrid accuracy is 85.435065%/86.487013%, versus Luna-only 81.363636%; hybrid gains 1.051948 points at 9.012987% mean Luna fallback. This is complementary routing, not evidence that Luna is universally stronger. Each specialist uses 1,540 fitting labels plus 770 reused development labels. Five seeds share one test population/one Luna set; seed SD is not a confidence interval. Usage-priced charges are $0.332622650 with bounded interval $0.332622650–$0.334866150, not an invoice-reconciled bill. All-case study spend differs from hypothetical routed charges; total production cost/savings remain unmeasured.

Preserve the completed EXP-007 artifacts, original manifests, response/attempt histories and source-compatibility receipts. Do not resume a completed run, reset attempts or create a replacement directory. Do not rerun test inference or tune thresholds, prompts, retrieval, classifiers or models from test outcomes. EXP-007 protocol SHA-256 remains `0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`. The fixed scalar rule uses full-precision confidence >= threshold independently per request; no rank or forced coverage. Historical preparation/repair write-ups describe their original statuses and are retained; the official test is no longer unscored.

EXP-009 retrieved-example test evaluation is **pending**, under separate protocol `b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334`. The existing test bundle now has 3,080 prepared requests and 826 successful live responses. Preserve that progress; do not repeat preflight or reset collection. The cooldown-only port reuses the reviewed EXP-007 helper, but the existing manifest correctly rejects changed source/Git identity. Follow `docs/CURRENT_PLAN.md`: separately approved exact-source compatibility support is required before resume; no manifest bypass or rewrite is allowed. Later paid execution separately requires explicit live authorization, current official model/settings/pricing verification and a sufficient approved numeric cap. EXP-007 approvals and a locally present key do not authorize EXP-009. Use exactly 20 retrieved examples from the existing 1,540 seed-11 training IDs; no prompt/k/model/threshold search or new fitting. Preserve both frozen protocols.

EXP-006R recovery and merged validation evaluation are complete; the earlier 718-success/52-HTTP-429 history remains intact. Validation findings remain exploratory and separate from official-test findings. EXP-008's existing CPU timing used validation inputs and one seed-11 model; no new timing study is implied. Keep raw customer text/data, request/response bodies, provider request IDs, secrets, weights and caches out of Git. Commit only compact verified evidence. No frontend, paid pilot, new dataset, calibration, fine-tuning or serving infrastructure is authorized by this checkpoint.

Use `uv sync --locked --cache-dir .cache/uv`, then `.venv/bin/python -m pytest -q`. CLI commands are documented in `README.md`. Commit only source, lockfiles, documentation, and compact evidence; raw data and large generated artifacts stay ignored. Confirm the Git top-level is this project before Git writes or recording provenance.
