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


## 2026-09-24 — EXP-002 complete validation learning curve

**Status: completed and independently reproduced locally.** The user's full-matrix request supersedes the earlier next-single-run plan. The original legacy smoke and v2 protocol-check records remain unchanged.

EXP-002 uses the unchanged word unigram/bigram TF-IDF + L2 multinomial logistic-regression pipeline at clean Git commit `30645360724d7fb8fe0c5a11d7dd7e02f3ac339b`. Dataset: BANKING77 revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Split protocol: `banking77-val10-v2`, seed 20260924, all 77 classes, the same 770 validation examples. Training seeds: 11, 22, 33, 44, 55; N=5/10/20. No model settings, splits, or seeds were selected using these scores.

- Primary IDs: `exp002-v2-n{5,10,20}-s{11,22,33,44,55}`; separate artifacts for all 15.
- Reproduction IDs append `-reproduction`; another 15 local refits in a separate root, excluded from aggregates. No seed selection or hyperparameter tuning occurred.
- All required per-run metadata, configurations, per-class metrics, predictions, timestamps, timings, and versioned split IDs are in [the study evidence](../experiments/exp002-learning-curve/README.md). All 30 fits used clean commit `3064536`, before analysis code/dependencies were added.

| Shots/class | Train/run | Accuracy mean ± SD (%) | Macro-F1 mean ± SD (%) | Macro-F1 min–max (%) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 385 | 51.84 ± 1.69 | 50.41 ± 1.58 | 48.77–52.59 |
| 10 | 770 | 62.49 ± 1.60 | 61.58 ± 1.81 | 60.05–64.14 |
| 20 | 1,540 | 69.95 ± 0.61 | 69.19 ± 0.63 | 68.18–69.80 |

Macro-F1 per seed (%):

| Shots/class | Seed 11 | Seed 22 | Seed 33 | Seed 44 | Seed 55 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 51.13 | 49.01 | 50.55 | 48.77 | 52.59 |
| 10 | 62.84 | 60.52 | 60.37 | 60.05 | 64.14 |
| 20 | 69.80 | 69.07 | 69.37 | 69.55 | 68.18 |

All 15 primary runs and 15 independent local refits completed without warnings or failures. Refits used separate clean output directories, reproduced sampled IDs, predicted labels, and complete quality metrics exactly, and had zero maximum absolute difference in classifier coefficients/intercepts. Only the 15 primary runs enter the aggregates. This is local reproducibility, not independent external or cross-platform replication.

The verifier checked every required artifact and recorded hash, dataset/split/configuration agreement, exact N-per-class training membership, train/validation/test ID and normalized-text isolation, and shared validation order/labels. Each saved TF-IDF vocabulary and IDF vector exactly matched a vectorizer rebuilt using only its intended training texts. Saved models reproduced their validation predictions/probabilities. Metrics were recomputed from all saved predictions. The aggregate summary, error-analysis JSON, and per-run CSV were independently regenerated byte-for-byte from the versioned records. No official test rows were parsed during this study; test access was limited to pinned byte-integrity checks. No test evaluation occurred.

The new aggregation/verification/plotting module is `baseline/learning_curve.py`. It replays validation predictions and reconstructs train-only TF-IDF to check artifacts; it does not fit a new classifier or evaluate test requests. Matplotlib 3.10.7 and its locked dependencies were added after model fitting, only for reproducible figures. Figures were visually checked. Aggregate JSON, error analysis, and per-run CSV reproduced byte-for-byte from the versioned primary records. Test suite: **35 passed**, no warnings.

Unique training labels across the matrix: 5,394; including the fixed validation set: 6,164. Refits added no new labeled examples. Per-regime training unions across seeds were 5-shot: 1,758, 10-shot: 3,177, 20-shot: 5,394. Source-label audit access is recorded separately. No failed or discarded model run occurred.

Both increments improved accuracy and macro-F1 for every paired seed. Mean macro-F1 gains were **+11.17 percentage points** from 5→10 (seed gains 9.81–11.71) and **+7.61 points** from 10→20 (4.04–9.50). Accuracy gains were +10.65 and +7.45 points. These are substantial, consistent development-set gains; no formal population significance claim is made.

Macro-F1 seed SD is **1.58 → 1.81 → 0.63 percentage points**: variance does not decrease monotonically, but is much lower at 20 shots. Accuracy SD is **1.69 → 1.60 → 0.61 points**. Seed variability is not validation-sampling uncertainty.

The slope is flattening, but the curve does **not yet show a clear plateau**: 10→20 still adds 7.61 macro-F1 points on average. Diminishing returns are especially visible per added example/class (approximately 2.23 F1 points for 5→10 versus 0.76 for 10→20). Three budgets cannot establish an asymptote.

Error-analysis findings: across all 15 primary models, `topping_up_by_card`, `supported_cards_and_currencies`, and `transfer_fee_charged` each averaged 26% recall. At 20 shots, weakest mean recall was `transfer_fee_charged` (30%), followed by `unable_to_verify_identity` and `supported_cards_and_currencies` (32%). The highest recall was 100% for `apple_pay_or_google_pay`, `lost_or_stolen_phone`, and `verify_source_of_funds` on this small holdout. The most common 20-shot directed confusion was `get_disposable_virtual_card` → `disposable_card_limits` (19/50 prediction events involving 4/10 distinct validation cases), followed by `getting_virtual_card` → `virtual_card_not_working` (17/50; 6/10 cases). Complete class/confusion tables are retained, not only selected examples.

These are validation results on only 10 fixed examples per class, reused across training seeds and regimes. SD uses `ddof=1` across the five training seeds; it is neither a confidence interval nor an estimate of uncertainty over new traffic. The legacy validation=20 smoke is excluded. Class-level 100% recall is an observation on a tiny holdout, not a reliability guarantee. Confusion counts repeat the same requests across models; 50 prediction events per class/regime represent only 10 distinct validation requests. Training subsets overlap across seeds, and validation was available in earlier development, so these are development comparisons rather than a fresh confirmatory evaluation. Near-duplicate leakage beyond the implemented normalized-exact audit remains untested. No production-quality, routing, calibration, or cost-savings claim is supported.

**Recommended next experiment: a 25-shot extension with the same TF-IDF + logistic regression, the same 770 validation IDs, and seeds 11/22/33/44/55.** This is the largest common budget supported by the current cleaned pool (minimum 25 per class), and tests whether the remaining 20→25 gain is still useful before changing models. First document/test a protocol extension that permits 25 while preserving the current validation IDs and each seed's 5/10/20 training prefixes; do not modify or relabel EXP-002. Compare paired 20→25 gains and their seed variability. The smallest class will use its entire 25-example pool for every seed, so disclose that reduced sampling variation. No 25-shot run, model change, routing, embedding, LLM, calibration, cost model, or official-test evaluation has been performed.
