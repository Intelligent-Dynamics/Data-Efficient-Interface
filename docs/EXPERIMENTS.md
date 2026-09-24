# Experiment log

Preserve completed, failed, null, and negative runs. Do not replace historical records with a better result. Verified measurements are summarized separately in `RESULTS.md`.

## 2026-09-24 — initialization

- Repository inspected: empty working tree apart from Git metadata; no commits or experiment artifacts.
- Persistent project context created; publisher dataset documentation reviewed.
- No experiment executed. Published dataset counts are reference facts, not project measurements.

## EXP-001 — BANKING77 few-shot lexical baseline

- Status: **planned; not implemented or run**.
- Hypothesis: lexical features can learn some useful intent distinctions with limited labels; the magnitude is unknown.
- Protocol: M1 in `CURRENT_PLAN.md`, dated 2026-09-24. Freeze a copy/hash of the protocol in run artifacts before execution.
- Comparison: most-frequent-class dummy versus fixed TF-IDF + logistic regression.
- Planned matrix: N = 5, 10, 20, 50; seeds = 11, 22, 33, 44, 55; fixed validation = 20 examples/class.
- Primary endpoint: validation macro-F1; secondary quality and resource measurements are specified in the plan.
- Official test access: none. Strong-model calls: none.
- Metrics, costs, artifacts, and outcome: unavailable; no measurements exist.

## Required record for each executed experiment

Record experiment/run ID, date, status, hypothesis, frozen protocol, code revision/dirty patch, exact command, environment/dependencies, dataset revision/checksums/license, label map, split/sample manifests, seeds, and complete model/prompt settings.

Include training/validation/calibration/prompt/evaluation label budgets, all attempted runs, artifact paths, predictions, measured metrics and uncertainty method, hardware/timing settings, warnings/errors, and reproducibility check. For model API work, retain usage and pricing provenance and distinguish actual spend from benchmark estimates and deployment projections.

Finish with observations, limitations, deviations, negative findings, interpretation, and the next decision. Clearly separate an observation from a hypothesis. Log every test access and whether the evaluation remains confirmatory. If a result is superseded, link its replacement and preserve the original record.
