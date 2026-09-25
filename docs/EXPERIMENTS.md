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


## 2026-09-25 — EXP-003 frozen MiniLM smoke: `exp003-minilm-v2-n5-s11`

**Status: completed implementation and one validation smoke check; not a final research result.** At the user's direction this milestone supersedes the proposed 25-shot extension. No other seed/regime, new split, fine-tuning, routing, calibration, paid inference API, GPU execution, or frontend was introduced. The existing TF-IDF pipeline and all EXP-001/EXP-002 records remain unchanged.

| 5-shot, seed 11; same v2 IDs | Accuracy (%) | Macro-F1 (%) |
| --- | ---: | ---: |
| TF-IDF + logistic regression (`exp002-v2-n5-s11`) | 52.21 | 51.13 |
| Frozen MiniLM + unchanged logistic regression | 75.84 | 74.46 |
| Difference (percentage points) | +23.64 | +23.33 |

Source: [complete compact evidence and reproduction instructions](../experiments/exp003-minilm-v2-n5-s11/README.md). Accuracy/F1 are computed from 770 saved validation predictions; the comparison uses the exact matching EXP-002 v2 run, not the old validation=20 smoke or the five-seed mean. The difference is +23.64 accuracy points and +23.33 macro-F1 points for this single training sample; no seed variance or significance estimate is available.

Model: `sentence-transformers/all-MiniLM-L6-v2`, immutable revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0. Native mean pooling and normalization; CPU float32, 384 dimensions, max length 256, batch 32, no prompts, eager attention. Frozen encoder in eval/inference mode. The 22,713,216 parameters have no gradients or trainable entries; before/after state hashes match. Only logistic regression was fitted. Its original L2/C=1/lbfgs/max_iter=2000/tol=1e-4/intercept/no-class-weight settings are unchanged, with no hyperparameter search. It converged in 9 iterations with no warnings. The encoder comes with substantial external pretraining; task few-shot counts do not describe its pretraining data.

Packages are pinned: sentence-transformers 5.1.1, torch 2.8.0, transformers 4.56.2; existing numerical pins preserved. Full package inventory, model-file hashes, hardware, configuration, timestamps, and base Git commit plus source/lockfile snapshot are in metadata. Run base commit: `059cd9d65f6dde72955f214cbfe8bb75d489e4b7`. The frozen pre-run protocol is retained in `protocol.md`. Setup downloaded 11 pinned inference files before the experiment; no failed/discarded model run occurred.

Exact train/validation IDs were read directly from the versioned reference artifact and checked against the existing `banking77-val10-v2` manifest (SHA-256 `f07ac5a03a3ae444db042dd064db920daa3f6462b1e92c9df36ad6118196030a`). No preparation or fresh sampling occurred. Sample SHA-256 `018f481053a04611d7171083b1235e8108a26a1a4ac8d4a3eb3cba210e481af8` matches EXP-002 exactly. The original split loader recomputes deterministic holdout membership solely to verify the existing manifest; it does not write or select a new split.

Budget: **385 training + 770 additional validation labels = 1,155 unique examples**, reused from the TF-IDF comparator, so zero new unique task labels. All 10,003 official-training labels remain mechanically accessed for audit/stratification, separately disclosed. Calibration/prompt/test-evaluation labels: zero. Official test access was byte-integrity checking only; no test rows parsed or evaluated.

Resource observations on Apple M1 Pro / 32 GiB / macOS 26.5.1 / Python 3.13.0, one CPU compute thread: training encoding 0.674077s; validation encoding 1.102255s; classifier fitting 0.018407s; classifier prediction 0.000910s. Cache resolution/model loading are recorded separately. These are single passes, not a warmed repeated latency benchmark; downloads, initial library imports, hashing and serialization are excluded. No inference-cost or savings claim follows.

Validation: **47 tests passed**. Synthetic tests cover aligned IDs/texts/labels, unchanged frozen encoder state and absent gradients, exact reference reuse despite reordered input pools, rejection of changed/legacy references and metric inconsistencies, train-only classifier inputs, serialization, and recorded convergence/mutation failures. Saved artifact/source hashes, sample budgets, ID/text isolation, feature alignment, frozen-state hashes, LR configuration, prediction/probability replay, and recomputed metrics/comparison all passed. Replaying the saved classifier did not refit or execute the encoder and is not an independent training reproduction.

Interpretation: the frozen representation is promising on this one paired development sample. A 770-case reused holdout with only 10 examples/class and one training seed cannot establish robustness, reliable per-class performance, calibration, unseen-traffic quality, or public-benchmark independence from pretraining. `RESULTS.md` is deliberately unchanged. The next recommended experiment is an independent reproduction of this exact smoke in a fresh directory before any multi-seed extension. It was not run; this milestone ends after commit/push.


## 2026-09-25 — EXP-004 full frozen-MiniLM learning curve

**Status: completed; all 15 primary configurations independently reproduced.** The user's full-matrix request supersedes the proposed single EXP-003 reproduction. The complete attempt ledger contains 30 successful real-data classifier fits, with no warnings, failures, discarded seeds, tuning or unreported retry. Synthetic tests are separate from these experiments. No dependencies or prior experiment artifacts changed.

BANKING77 publisher revision `57ec275d8078af65b7731c2a98be812d844a6d6b`; `banking77-val10-v2`, split seed 20260924, all 77 classes. Each ordered training/validation ID list was read from its matching EXP-002 artifact and checked against the preexisting v2 protocol report and manifest. No preparation, resampling, holdout changes or hyperparameter tuning occurred.

Encoder: `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with the exact EXP-003 model-file hashes. CPU float32, max length 256, batch 32, mean pooling, L2 normalization, 384 dimensions, no prompts, eager attention, eval/inference mode, deterministic operations, seed 11 for encoding, all weights frozen. Encoder state remained identical to EXP-003 before/after both cache stages. Runtime pins are unchanged: sentence-transformers 5.1.1, torch 2.8.0, transformers 4.56.2, scikit-learn 1.7.2, NumPy 2.3.3, SciPy 1.16.2; all installed versions are recorded.

Only logistic regression was trained: unchanged L2, C=1, lbfgs, max_iter=2000, tol=1e-4, intercept, no class weighting, multinomial loss. Each fit receives only its exact N-shot rows/labels; encoding the shared union with a frozen model does not fit on the union. No encoder updates, learned scaler or calibration.

Primary IDs: `exp004-minilm-v2-n{5,10,20}-s{11,22,33,44,55}`; independent-refit IDs append `-reproduction`. All ran from `8dcffc549ce170c875ec15adaf3b78b886e06836` plus the same recorded new-runner/test/protocol snapshot. Started 2026-09-25T18:09:01.341366+00:00; finished 2026-09-25T18:09:29.047274+00:00. [The evidence directory](../experiments/exp004-minilm-learning-curve/README.md) includes exact commands, the pre-run plan, compact primary/refit records, both cache metadata sets, verification, summary, CSV and figure. Full binaries/caches/source snapshots remain ignored locally.

| Shots/class | MiniLM accuracy (%) | MiniLM macro-F1 (%) | TF-IDF accuracy (%) | TF-IDF macro-F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 74.13 ± 1.30 | 72.60 ± 1.34 | 51.84 ± 1.69 | 50.41 ± 1.58 |
| 10 | 80.05 ± 0.93 | 79.19 ± 1.03 | 62.49 ± 1.60 | 61.58 ± 1.81 |
| 20 | 83.77 ± 0.66 | 83.35 ± 0.68 | 69.95 ± 0.61 | 69.19 ± 0.63 |

Per-seed macro-F1 gain over the matching TF-IDF configuration, in percentage points:

| Shots/class | Seed 11 | Seed 22 | Seed 33 | Seed 44 | Seed 55 | Mean gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | +23.33 | +22.74 | +20.39 | +24.01 | +20.49 | +22.19 |
| 10 | +16.87 | +17.64 | +17.62 | +20.01 | +15.87 | +17.60 |
| 20 | +13.86 | +13.59 | +13.36 | +13.86 | +16.11 | +14.16 |

MiniLM improves macro-F1 for **all 15 matched training-budget/seed pairs**. Mean advantages over TF-IDF are +22.19, +17.60, and +14.16 percentage points at 5/10/20 shots. Absolute differences for every seed are retained above; no best seed was selected.

The MiniLM curve gains **6.58 macro-F1 points from 5→10** and **4.16 points from 10→20**. Returns diminish, but these three budgets do not establish a plateau. MiniLM macro-F1 seed SD decreases **1.34 → 1.03 → 0.68 points**, and accuracy SD decreases **1.30 → 0.93 → 0.66 points**. MiniLM's seed SD is lower than TF-IDF's at 5/10 shots but slightly higher at 20 shots. Macro-F1 ranges across all five seeds: 5-shot 70.94–74.46%; 10-shot 77.99–80.06%; 20-shot 82.65–84.29%.

All 15 primary runs and all 15 independent local refits completed without warnings or failures. The reproduction stage reloaded the pinned frozen encoder and recomputed all 6,164 required vectors into a separate cache, then fitted new logistic-regression instances for every configuration. Embedding maximum absolute difference was **0.0** (predeclared tolerance 1e-6, relative tolerance zero). Every pair had identical sampled IDs, predicted labels, full metrics, and classifier coefficients/intercepts; maximum parameter difference was **0.0** for all 15. The 5-shot seed-11 run also reproduced EXP-003's predictions and metrics exactly. Only primary runs enter aggregates.

This is independent local re-encoding and refitting using the same implementation and machine, not independent external replication or uncertainty over new traffic. Saved classifier replay is an additional integrity check, not counted as another independent fit. Artifact/cache/source hashes, reference sample hashes, exact per-class training counts, fixed validation IDs/labels, normalized-text/ID isolation, feature alignment, unchanged encoder state, and classifier probability/prediction replay all passed. A separate NumPy calculation agreed with all reported means and sample SDs to maximum absolute difference 1.11e-16. Summary JSON, per-seed CSV and plot were regenerated byte-for-byte from compact primary records. **64 tests passed**; the figure was visually inspected.

The primary cache reused EXP-003's 1,155 validated vectors and encoded 5,009 additional training texts in **7.492437s**. The independent cache encoded 385 / 770 / 5,009 rows in **0.618963s / 1.141228s / 7.422240s**. Encoder model loading/cache resolution is recorded separately. These are single component wall-time observations on Apple M1 Pro, 32 GiB RAM, macOS 26.5.1, Python 3.13.0, CPU float32, one compute thread.

Each run records classifier fitting and prediction time on cached embeddings. **These are classifier-only measurements, not end-to-end inference speed.** Cache hits record no new encoding time (`null`); end-to-end inference time is unmeasured (`null`). Encoding, cache I/O, hashing, serialization and probability generation are excluded from classifier timers. No warmed repeated latency comparison or monetary model was run. Do not amortize these cache observations into a production savings claim.

Every run uses **770 additional validation labels** (10 per class), besides 385/770/1,540 training labels for 5/10/20 shots. Per-run totals are 1,155/1,540/2,310. Across the study, 5,394 unique training rows plus 770 validation rows give 6,164 unique task labels. These are the same examples used by EXP-002; the new representation and refits added no new unique task labels. All 10,003 official-training labels were mechanically read for audit/stratification and are separately disclosed. The encoder benefits from substantial external pretraining; this is simulated few-shot task adaptation, not total training-data or annotation-cost accounting.

The official BANKING77 test set stayed sealed. Test access was limited to byte-integrity checks; no test rows were parsed, labeled, inspected, encoded, tuned on, or evaluated. Existing stored test IDs/text hashes were used for isolation checks. The dataset and split manifest are unchanged.

SD is **sample SD across five training seeds (`ddof=1`)**, not a confidence interval or uncertainty over new validation samples. All runs share the same previously used 770-case holdout, with only ten examples per class; seeds share overlapping training data. These are paired development comparisons, not confirmatory unseen-test or production results. Public-benchmark exposure in upstream pretraining and semantic near-duplicate leakage cannot be ruled out. No calibration, routing, production-quality, or cost-savings claim is established.

The unchanged test checksum is `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`; split-manifest SHA-256 is `f07ac5a03a3ae444db042dd064db920daa3f6462b1e92c9df36ad6118196030a`. Verified validation findings were appended to `RESULTS.md`; earlier sections/artifacts remain intact.

Next recommended work: paired error analysis of the saved 20-shot predictions across all five seeds, identifying intent/request-level improvements and regressions against TF-IDF. This requires no new fitting or test access. No fine-tuning, routing, paid APIs or frontend was added.


## 2026-09-25 — EXP-005 confidence/selective-prediction diagnostic

**Status: completed exploratory analysis of all 15 existing EXP-004 primary runs.** The user's request supersedes the proposed paired 20-shot error analysis. No new model was trained or executed; no probabilities needed regeneration. Dataset, sample IDs, package/model pins and earlier artifacts are unchanged. The official test remained sealed, with no raw dataset or test-byte access in this milestone.

Source provenance, sample IDs/labels, probability row/class order, normalization, maximum-probability/prediction agreement, full metrics and saved artifact/cache/source hashes all passed before ranking. Confidence is **UNCALIBRATED maximum class probability**. Rank descending, with exact ties broken by SHA-256 of UTF-8 sample ID then ID, without consulting labels. No exact confidence ties occurred. All 771 prefixes per run are reconstructible; zero accepted accuracy/risk is `null`. Full coverage exactly matches every corresponding EXP-004 accuracy.

The requested 25/50/75/90/100% landmarks accept 193/385/578/693/770 cases using `ceil(target * 770)`. Below are mean ± sample SD across five seeds; coverage/accepted-count SD is zero. Zero-acceptance intent counts include mean ± SD and range.

| Shots/class | Actual coverage (%) | Accepted/run | Accepted accuracy (%) | Selective risk (%) | Errors/run | Intents with zero accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 25.065 | 193 | 92.12 ± 1.44 | 7.88 ± 1.44 | 15.20 ± 2.77 | 31.60 ± 2.30 (28–34) |
| 5 | 50.000 | 385 | 87.90 ± 0.91 | 12.10 ± 0.91 | 46.60 ± 3.51 | 8.20 ± 2.39 (5–11) |
| 5 | 75.065 | 578 | 82.49 ± 1.65 | 17.51 ± 1.65 | 101.20 ± 9.55 | 0.80 ± 0.84 (0–2) |
| 5 | 90.000 | 693 | 78.67 ± 1.25 | 21.33 ± 1.25 | 147.80 ± 8.64 | 0.00 ± 0.00 (0–0) |
| 5 | 100.000 | 770 | 74.13 ± 1.30 | 25.87 ± 1.30 | 199.20 ± 10.03 | 0.00 ± 0.00 (0–0) |
| 10 | 25.065 | 193 | 97.72 ± 0.70 | 2.28 ± 0.70 | 4.40 ± 1.34 | 29.80 ± 2.05 (28–32) |
| 10 | 50.000 | 385 | 93.25 ± 1.31 | 6.75 ± 1.31 | 26.00 ± 5.05 | 8.00 ± 1.58 (6–10) |
| 10 | 75.065 | 578 | 88.72 ± 1.00 | 11.28 ± 1.00 | 65.20 ± 5.76 | 0.80 ± 1.30 (0–3) |
| 10 | 90.000 | 693 | 84.36 ± 0.88 | 15.64 ± 0.88 | 108.40 ± 6.11 | 0.00 ± 0.00 (0–0) |
| 10 | 100.000 | 770 | 80.05 ± 0.93 | 19.95 ± 0.93 | 153.60 ± 7.13 | 0.00 ± 0.00 (0–0) |
| 20 | 25.065 | 193 | 99.27 ± 0.28 | 0.73 ± 0.28 | 1.40 ± 0.55 | 27.00 ± 1.58 (25–29) |
| 20 | 50.000 | 385 | 96.68 ± 0.34 | 3.32 ± 0.34 | 12.80 ± 1.30 | 6.00 ± 1.22 (5–8) |
| 20 | 75.065 | 578 | 92.60 ± 0.55 | 7.40 ± 0.55 | 42.80 ± 3.19 | 0.00 ± 0.00 (0–0) |
| 20 | 90.000 | 693 | 88.05 ± 0.30 | 11.95 ± 0.30 | 82.80 ± 2.05 | 0.00 ± 0.00 (0–0) |
| 20 | 100.000 | 770 | 83.77 ± 0.66 | 16.23 ± 0.66 | 125.00 ± 5.10 | 0.00 ± 0.00 (0–0) |

[Complete evidence, curve, interpretation and reproduction commands](../experiments/exp005-selective-diagnostic/README.md) preserve all 15 individual diagnostics, 75 landmark rows, 5,775 per-intent acceptance/error rows, and every seed's values. High confidence does identify more reliable subsets on this reused validation set. Every non-full landmark is more accurate than its own model's full result, but risk is not strictly monotonic at every prefix.

Class composition is a material limitation: at 50%, 20-shot runs omit all examples of 5–8 intents, and at 25% omit 25–29. `cash_withdrawal_not_recognised` is entirely rejected at 50% in all 15 runs; `topping_up_by_card` is entirely rejected in all five 20-shot runs. Every intent is represented by 75% coverage for 20 shots, while some 5/10-shot runs still omit up to 2/3. Preserve these omissions alongside the favorable aggregate accuracy.

Implementation: `baseline/selective.py` reads audited saved predictions/probabilities and supports compact-record regeneration. A separate standard-library checker independently reproduces rankings, every prefix's error counts, landmarks and per-class counts from the original probabilities; aggregate maximum absolute difference is 1.78e-15. All individual records, summary, CSV tables, complete aggregate curves and PNG regenerate byte-for-byte. Plot visually inspected. **99 tests passed**, including new ordering/tie/count/risk, zero-null, alignment, provenance-data consistency and compact-record corruption cases. No diagnostic execution failed or was discarded.

Analysis began `2026-09-25T18:47:05.905393+00:00` and completed `2026-09-25T18:47:08.388105+00:00`, from base commit `e1b428c606567b19145ef058e0f96f77f75fb060` plus hashed analysis/tests/pre-run protocol. Exact commands, source/artifact hashes and verification are retained in the evidence directory. EXP-001–004 and `RESULTS.md` remain preserved; the diagnostic is recorded as exploratory evidence here, not promoted to a production claim.

The same **770 additional validation labels** are reused; zero new labels are added. Five seeds do not create independent validation datasets or confidence intervals. The ten examples per intent and repeated holdout use limit high-confidence/rare-error conclusions. No calibration model or production threshold was fitted or selected. Rejected requests have no assumed LLM outcome. No combined-system accuracy, speed, dollar savings, fine-tuning, routing, paid APIs or frontend was introduced.

Next proposed experiment, not executed: a saved-prediction diagnostic of persistent intent rejection and confident errors. Use the same 15 records and predefined EXP-005 prefixes to report all-intent acceptance/error tables and per-request recurrence across seeds; compare the same requests against the existing TF-IDF predictions. No new labels, fitting, threshold optimization or test access. Any later calibration experiment requires a separately specified label-budgeted fitting/evaluation protocol first. Stop after this milestone's commit/push.
