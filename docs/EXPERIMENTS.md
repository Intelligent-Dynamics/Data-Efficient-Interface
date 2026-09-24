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


## 2026-09-24 — canonical checkout and protocol v2 (no model experiment)

**Status: protocol change implemented and tested; no additional BANKING77 model run.** This dated entry supersedes the earlier validation=20 plan, without changing the legacy smoke metrics or artifacts.

- The existing checkout was moved intact to `/Users/Andrew/Developer/data-efficient-inference`. Before edits, `pwd` and Git top-level matched that path, the working tree was clean at `717b7e6` (one commit ahead of remote), and origin remained `https://github.com/Intelligent-Dynamics/data-efficient-inference.git`. The former Documents/ChatGPT directory no longer exists. Git history, ignored data, and local artifacts moved with the checkout.
- At the user's direction, protocol `banking77-val10-v2` fixes validation/calibration development data at **10 examples per class**, with split seed 20260924, and uses **5/10/20-shot** training for all 77 intents. Training seeds remain 11/22/33/44/55. The same held-out IDs apply to every regime/seed, with nested deterministic training samples. Calibration is not implemented; the held-out pool is currently used for validation.
- Fifty-shot is removed, not approximated through oversampling, class exclusion, or test reuse. The smallest source-training class has 35 usable examples and retains 25 after the new holdout. Any future reconsideration requires valid additional training data and a documented protocol.
- No source or duplicate-cleaning changes: 9,992 cleaned official-training rows become **770 validation rows + 9,222 pool rows**. The 3,080 official test rows and their checksum remain unchanged. Test access was limited to mechanical schema/count/text-hash auditing and integrity checks; no test labels, model predictions, evaluation metrics, or qualitative test inspection.
- Versioning: schema-2 manifest at `data/processed/banking77-val10-v2/manifest.json`; explicit protocol metadata checked by the training loader. Old or changed metadata is rejected; existing manifests are never overwritten with a different split. The old manifest and all original smoke-artifact hashes were checked unchanged.
- Tests: `.venv/bin/python -m pytest -q` completed with **29 passed**, no warnings. Real-data sampling checks passed for **15 configurations** (three sizes × five seeds): all class counts exact, deterministic and nested sampling, fixed validation, and ID/normalized-text isolation. No BANKING77 fit was needed; unit-test fits used synthetic data only.
- Machine-readable check report: [`experiments/protocol-val10-v2-check.json`](../experiments/protocol-val10-v2-check.json). V2 split manifest SHA-256: `f07ac5a03a3ae444db042dd064db920daa3f6462b1e92c9df36ad6118196030a`. The report contains per-configuration sample hashes and the validation hash.
- Per-run fitting/validation budgets are 1,155 / 1,540 / 2,310 unique labels for 5/10/20 shots, including 770 validation labels. Source labels read for stratification/auditing remain separately disclosed; no annotation-cost claim.
- Interpretation: the new budget is suitable for coarse early model comparisons but weak for small differences, per-class metrics, and later high-confidence calibration. Per-class recall has 10-percentage-point granularity. Repeated tuning can overfit this holdout; independent calibration/evaluation or a suitable cross-fitting design will be needed before calibration claims.
- The v2 validation set is a deterministic subset of the old holdout. Earlier smoke scores remain legacy observations and must not be pooled with v2 results or treated as evidence on a fresh confirmatory sample. Reproduce old runs using code at `717b7e6` (or their saved source snapshot), not the new CLI defaults.
- Exact next experiment, still pending: N=10/seed=11 under v2 in `artifacts/exp001-v2-n10-s11`, followed by an independent reproduction and a full v2 multi-seed matrix including fresh 5-shot runs. No routing, SetFit, embeddings, LLM, or calibration work was started.
