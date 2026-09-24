# Experiment log

Preserve completed, failed, null, and negative runs. Do not replace historical records with a better result. Verified measurements are summarized separately in `RESULTS.md`.

## 2026-09-24 — initialization

- Repository inspected: empty working tree apart from Git metadata; no commits or experiment artifacts.
- Persistent project context created; publisher dataset documentation reviewed.
- No experiment executed. Published dataset counts are reference facts, not project measurements.

## EXP-001 — BANKING77 few-shot lexical baseline

- Initial status: **planned; not implemented or run**. Updated status: one smoke run completed below; full study and independent reproduction remain pending.
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

## 2026-09-24 — EXP-001 preparation and protocol revision

- The implementation-session instruction limits execution to one N=5/seed=11 smoke experiment; the original full matrix remains deferred.
- Initial preparation stopped with `Insufficient unique training examples for contactless_not_working: 35`, before model fitting. The original 70-example prerequisite was not met. With 20 validation examples per class, N=20 and N=50 are infeasible; retain this negative feasibility finding.
- Revised protocol: preserve 77 classes and validation=20; prepare the split when N=5 is possible, report availability for each regime, and reject infeasible requests without oversampling or dropping classes. See `CURRENT_PLAN.md` for class counts.
- Duplicate audit: 7 training rows matched test hashes; 4 within-training duplicates were removed; no conflicting-label groups. Test rows remain intact. Test access consisted only of mechanical schema/count checks, text hashing for contamination checks, and file-integrity hashing; no test-label use, predictions, metrics, or error analysis.
- Setup issues: sandbox network restrictions required network-enabled downloads. Initial dependency resolution exposed a NumPy 2.5/joblib serialization deprecation; NumPy 2.3.3 and SciPy 1.16.2 were pinned before the real experiment, with the complete environment in `uv.lock`.
- During implementation the original local Git metadata and documentation disappeared (cause unknown). The exact initial commit `649f7e106dd88f31101b910eb17783cb6b779e6d` was verified on the private remote and restored into the current directory, preserving the new code. No unrelated parent-repository files were changed.

## 2026-09-24 — EXP-001 smoke run: `exp001-smoke-n5-s11`

**Status: completed pipeline check; not a final project result or independently reproduced experiment.** No other real-data training regime was run this session. Synthetic unit-test fits are separate from this experiment.

Command:

```sh
.venv/bin/python -m baseline run --shots 5 --seed 11 --output artifacts/exp001-smoke-n5-s11
```

- Dataset revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`; source file hashes and upstream license recorded in the metadata.
- Training: 5 examples × 77 classes = 385 labels. Validation: 20 × 77 = 1,540 labels. Total fitting/validation budget: 1,925 unique examples. Calibration/prompt/test-evaluation labels: none. All 10,003 source-training labels were mechanically read for stratification and duplicate auditing; this is simulated limited-data training, not an annotation-cost claim.
- Model: fixed word unigram/bigram TF-IDF with sublinear term frequency, smoothed IDF and L2 normalization; multinomial logistic regression, L2, C=1, lbfgs, max_iter=2000, tol=1e-4. No tuning or calibration. Training seed 11, validation seed 20260924.
- Environment: Python 3.13.0; scikit-learn 1.7.2, NumPy 2.3.3, SciPy 1.16.2, joblib 1.5.2, threadpoolctl 3.6.0; macOS 26.5.1 arm64; one compute thread. Complete package lock, native library info, and timing procedure are recorded with the run.
- Code provenance: base commit `649f7e106dd88f31101b910eb17783cb6b779e6d` plus the uncommitted implementation snapshot. Per-file source hashes, snapshot hash, and working-tree patch hash are in `metadata.json`; the run's source files are retained locally. The milestone commit includes those source files unchanged, with post-run documentation updates. The frozen pre-run plan is retained as `protocol.md` in the compact evidence directory.

| Validation measurement | TF-IDF + logistic regression | Most-frequent-class dummy |
| --- | ---: | ---: |
| Accuracy (fraction) | 0.5253246753246753 | 0.012987012987012988 |
| Macro-F1 (fraction) | 0.5138837328503768 | 0.000333000333000333 |
| Evaluated examples | 1,540 | 1,540 |

Per-class precision/recall/F1/support and confusion counts are stored in `metrics.json`. The dummy's tied-class choice is recorded in metadata. No multi-seed uncertainty or test-set performance is asserted.

Local resource observations: fit time 0.047240125015378 seconds; 2,998 TF-IDF features; 4 optimizer iterations; serialized model 1,932,612 bytes. Prediction timings include one warmup and five measured passes for batch size 1 and full-validation batches; raw times and units are in metadata. These are single-machine observations, not inference-cost or production-latency claims. No warnings occurred.

Validation performed: 22 unit tests passed with no warnings; artifact SHA-256 checks passed; exact per-class budgets and split isolation passed; probability dimensions/order and row sums passed; both models' complete metrics recomputed exactly from saved predictions without refitting. The official test file's pinned checksum was verified unchanged. There was no test evaluation or qualitative test inspection.

- Split manifest SHA-256: `94afed93106efc82a0ab3bf143e18c6ae19c823061c956fe28c14a51fffe6305`.
- Sample manifest SHA-256: `e2db7786f0526e12ff1e0643bfa37fe6adc92cb1c413abf1780330f766919865`.
- Versioned compact evidence: [`experiments/exp001-smoke-n5-s11/`](../experiments/exp001-smoke-n5-s11/) contains metadata, metrics, sampled IDs, predicted labels, and the frozen protocol. Full probabilities, model, split audit, source snapshot, and patch remain in ignored `artifacts/exp001-smoke-n5-s11/` and can be regenerated from the pinned source and recorded command.

Interpretation: the pipeline fits and evaluates successfully and outperforms the dummy on this validation sample. One training sample does not establish robustness, calibrated confidence, useful routing coverage, or cost savings. Preserve the infeasible-budget finding above. Next: N=10/seed=11 with unchanged settings, then an independent reproduction and the feasible multi-seed study described in `CURRENT_PLAN.md`.
