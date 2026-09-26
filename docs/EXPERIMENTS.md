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


## 2026-09-25 — EXP-006 general-model/fallback preparation

**Status: preparation implemented and tested; PAID EXPERIMENT NOT RUN.** User authorization covers runner development, synthetic/mock tests, a frozen protocol and a no-inference cost estimate only. No paid endpoint, general-model validation prediction, or real combined-system evaluation occurred. Actual API spend for this preparation: **$0**. No new model, split, embedding, calibration, tuning, or official-test access. EXP-001–005 and `RESULTS.md` remain unchanged.

Candidate **`gpt-6-luna`** was verified from official OpenAI model/pricing/Responses documentation on 2026-09-25. No dated snapshot is listed; alias/underlying-version limitations are explicit. Freeze reasoning `none`, 128 maximum output tokens, Standard/default processing, strict one-intent JSON enum, no tools/history, and one zero-shot prompt containing all 77 official names. Ground truth, specialist output/confidence, row IDs and demonstrations are absent from API payloads. [Exact prompt, schema, protocol and sources](../experiments/exp006-general-model-preparation/README.md). Protocol SHA-256: `d7f824ee761ca90d4dc3a848dd19e3928475c799d6713e03e336dc2a80c9ca68`.

The offline input audit verified the exact existing 770 IDs/texts/labels, pinned training CSV and categories, existing v2 manifest, 15 EXP-004/005 evidence pairs, sample separation and stored sealed-test hashes without opening the official test file. No examples were qualitatively inspected for prompt tuning. All 770 validation labels were already used in development; zero additional unique labels or demonstrations are introduced.

One future 770-response set serves all 15 specialists and 90 combinations (the five existing EXP-005 prefixes plus all fallback). At most two attempts per request gives 1,540 maximum attempts. The runner saves request/response hashes, provenance, model/tier, timestamps, usage details and timing; bounded transport retries never depend on correctness. Refusals/incomplete/invalid outputs remain terminal failures. Cached successes are reused. A durable reservation ledger prevents deleted/truncated cache files from resetting attempted-call budgets, including during read-only evaluation. Live CLI and engine require explicit user authorization, numeric cap, protocol hash and fresh pricing acknowledgement.

No-inference dry-run estimates: **$0.124057 nominal** for 770 first attempts; **$1.7313625 conservative** with the full output/input envelope; **$3.462725** with every permitted retry. The proposed future cap is **$4.00, not approved**. Nominal input uses serialized UTF-8 bytes/4 and 32 output tokens. Reservation uses twice serialized bytes plus 8,192 input tokens, full 128-output cap and all input at cache-write pricing. These are token-count heuristics, not measured usage or guaranteed invoices. Frozen Standard per-million input/cached/write/output rates are $0.10/$0.01/$0.125/$0.50; sources/date and exclusions are preserved.

Future general and combined accuracy/macro-F1 use all 770 requests, counting unresolved responses as failures. Rejected-subset fallback accuracy is separate; all-fallback/full-specialist invariants are tested, with undefined fallback accuracy at zero rejected requests. Preserve each combination and mean/sample SD across specialist seeds, which share the same validation data and general responses. No production threshold is selected and the fallback is not assumed better.

Actual collection usage-priced charges (including failures and unknown-usage reservations) are kept separate from hypothetical API-only routed replay estimates. Unknown accounting stays unknown; model/tier or token-envelope violations halt and invalidate unsupported cost bounds. Replayed cache behavior is an assumption, not new spend. Specialist deployment cost, total-system dollars/savings and production latency remain unmeasured/null.

Implementation comprises `baseline/general_protocol.py`, `baseline/general.py`, `baseline/general_metrics.py` and synthetic tests, using existing packages plus the Python standard-library HTTP client. Synthetic review identified and fixed malformed response/usage crashes, missing cached-response evidence, duplicate payload keys, mismatched-model pricing, and evaluation ledger checks; these were test/development defects, not discarded API experiments. Test-only mock setup issues were corrected without accessing live services. **243 tests passed**, including 144 new synthetic tests. A final guarded dry run rejected any possible HTTP/socket/credential access and completed with zero such calls; its 770 unique IDs/cache keys, frozen prompt and independent Decimal cost recalculation all matched. Both dry runs yielded the same cost/configuration; no response/evaluation artifact was produced. Final source and dry-run hashes are recorded in the preparation verification artifact.

Next milestone: only after explicit approval of the frozen protocol and a numeric spending cap, collect/evaluate the single 770-response set. First recheck official pricing/model availability if needed, then use the reviewed command. Preserve all failures and independently reproduce scores from saved responses. Live account/model compatibility, token usage, general accuracy, fallback quality and actual paid cost remain unknown. This session stops after commit/push of preparation.


## 2026-09-26 — EXP-006R HTTP-429 recovery preparation

**Status: PREPARATION ONLY; PAID RECOVERY NOT RUN. Actual preparation API spend: $0.** No recovered predictions, quality metrics or merged research result are claimed. The earlier EXP-006 preparation entry remains historical; its later saved live execution is the immutable source of this new preparation.

Read-only source audit: `artifacts/exp006-gpt6-luna-validation-v1/execution.json` has **718 ok, 52 http_429, status finished_with_unresolved**. All 52 eligible IDs have exactly two original HTTP-429 attempts and no prediction. Across all original requests there are 823 saved attempts: 718 successful, 105 HTTP-429. All 105 errors explicitly report `insufficient_quota` / `credit_balance_exhausted`; original response caches contain no Retry-After headers. Restore credits/quota before future recovery; repeated backoff cannot resolve the original cause. No account/billing endpoint or credential was inspected. Original source inventory: **775 files, 16,688,254 bytes**, tree SHA-256 `bfd8ad140577004bb436a4c89a977721a3286dd358470fa05e5cf1351037a7cb`. Before/after hashes prove preservation, including hidden files and every old attempt.

Original protocol `d7f824ee761ca90d4dc3a848dd19e3928475c799d6713e03e336dc2a80c9ca68` stays unchanged. Separate EXP-006R protocol **`99500454a24e5b01ba3c1bba16d5076101d06a5a3d448fe400c32af716c16161`** records all 52 exact IDs/cache keys and the source inventory digest. It reuses the exact saved `gpt-6-luna` request bodies, original prompt/schema/class names, reasoning none, 128-output cap, service tier/default, store false and all remaining settings. No 718-success resubmission, model retraining, dataset reading, split regeneration, changed specialist output or changed evaluation definition. The same **770 additional validation labels** are reused; zero new labels. Official BANKING77 test remains sealed with no test-file access.

Implementation: `baseline/recovery_source.py` audits immutable original records against frozen source/EXP-004 provenance; `baseline/recovery.py` provides guarded dry-run/live/offline-merge commands with a separate durable ledger/cache; `baseline/recovery_merge.py` preserves all 770 IDs and reuses the unchanged EXP-006 evaluator for standalone and all 90 specialist/fallback combinations. Output is labeled **EXP-006 + EXP-006R recovery**, never the untouched original experiment. Synthetic tests cover exact eligibility, row alignment, byte preservation, no-network/no-test access, explicit authorization/numeric caps, ledger and source corruption, bounded retries, cooldown persistence, quota stops, failures and full-denominator metrics.

Frozen new retry policy: serial requests; five-second minimum gap after completion; at most four recovery attempts per eligible ID; exponential waits 30/60/120 seconds, with a 240-second global cooldown after an exhausted fourth transient failure. Respect Retry-After seconds/dates and optional millisecond hints across IDs/restarts. A single wait above one hour pauses instead of sending early. Quota/billing errors halt the collection immediately. Refusal/incomplete/invalid or valid-but-wrong answers are never retried for correctness. Interrupted or missing-usage attempts retain their spending reservations.

No-network dry-run estimates for **52 unique requests / at most 208 attempts**: nominal one-attempt **$0.008374250**, conservative one-attempt **$0.11689400**, conservative all-attempt **$0.46757600**. Proposed future recovery cap **$0.50 is NOT AUTHORIZED**; original EXP-006's approval does not carry over. Token counts use the unchanged byte heuristic, full output allowance and all-input cache-write pricing without a cache-read discount. The estimate is conditional on its envelope, not a guaranteed invoice. Official rates reverified 2026-09-26 match original Standard short-context rates. Missing usage stays unknown; original/recovery charges and API-only replay estimates remain separate. No total-system savings or production-latency claim.

**Final full suite: 371 passed in 22.23s**, including 128 new recovery tests. The real-source dry run completed with network, credential and raw/test-data access blocked; independent Decimal estimates matched, and all 775 source files remained byte-identical. No prior experiment implementation, protocol or artifact was changed.

[Prepared protocol, exact dry/live/merge commands, sources, cost breakdown and verification](../experiments/exp006r-429-recovery-preparation/README.md). No live recovery or real merged evaluation was executed. Next: resolve quota, review current pricing and obtain new explicit approval for the recovery hash plus numeric cap; only then collect the 52 cases and evaluate offline. Preserve previous results, including unresolved/negative outcomes.


## 2026-09-26 — EXP-007 fixed-threshold preparation

**Status: PREPARATION COMPLETE; OFFICIAL TEST NOT RUN. Preparation API spend: $0.** No paid call, model execution, fit, calibration, new split or official-test access occurred. Test data were not opened even for a checksum. Dataset counts/checksum metadata were read from the already-versioned source specification only.

Saved validation evidence now confirms EXP-006R completed with 52/52 successful recovery responses and a completed 770-response merged set. The original 718 successes and original 775-file inventory remain unchanged. This supersedes the old preparation-only status as a current-state observation; it does not overwrite historical EXP-006/006R artifacts or introduce recovered-quality claims here. The completed responses supplied validation-only token usage for the later cost projection.

At the user's direction, fix the 20-shot specialist and existing 90% EXP-005 landmark. Audit all five EXP-004 primary classifiers/probabilities and EXP-005 records before deriving thresholds from IDs/confidences alone. Twenty-five source files are hash-bound; classifier bytes are checked without deserialization. Original labels, IDs, model revision, packages, LR and embedding settings remain unchanged.

| Seed | Fixed threshold (`>=`) | Validation acceptance | Coverage | Changed IDs |
| --- | ---: | ---: | ---: | ---: |
| 11 | 0.10354116298070118 | 693/770 | 90% | 0 |
| 22 | 0.10243964501544885 | 693/770 | 90% | 0 |
| 33 | 0.1021902311171956 | 693/770 | 90% | 0 |
| 44 | 0.10160314104812371 | 693/770 | 90% | 0 |
| 55 | 0.10612054364389162 | 693/770 | 90% | 0 |

Each threshold is the exact rank-693 confidence, serialized as a round-trip JSON/decimal/hex binary64 value. Rank 694 is strictly below it for every seed: no boundary tie and no accepted-set difference. The deterministic tie policy chooses all-or-none boundary ties to minimize set difference, including all on an equal-distance tie; it never consults correctness. `>=` runs independently per request without ranking a future workload or enforcing a target fraction. Confidence is uncalibrated. This cutoff computation uses no labels, but choosing 20 shots/90% was informed by previous development results; disclose all **770 reused validation labels** plus 1,540 fitting labels per seed, with zero new labels.

Frozen future test protocol SHA-256: **`0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`**. It will use all 3,080 official test requests, all five existing specialists, observed threshold coverage, and unchanged `gpt-6-luna` prompt/schema/settings below threshold. The full Luna-only comparator requires one all-case response set shared by the five seeds. Report specialist-only/Luna-only/routed accuracy and macro-F1 over all cases, fallback number/percentage and rejected-subset accuracy, failures and all seed results/sample SD. Never retune, recalibrate, refit, force 90% on test, or select the best seed after results. New test access and capped spending authorization are required; preparation exposes no test loader/live mode.

API estimates, using only completed validation token usage and current official rates verified 2026-09-26: nominal **$0.3325308**, or **$0.4083580** for the same tokens all at cache-write input rates. The 770 successful validation responses used 758,272 input and 14,611 output tokens (zero recorded caching/reasoning tokens). Known successful-response charges $0.0831327 exclude 105 original failure attempts with unknown usage; this is not verified total experiment spend. The engineering envelope uses the largest validation payload (4,817 compact bytes), `2*bytes+8192` input and 128 output tokens, with no cache discount. For 3,080 first attempts this is **$7.06013**; four attempts each yields **$28.24052**. This assumes test requests fit the validation byte envelope and is not a guaranteed bill or approved cap. Actual test preflight must occur only after separate access approval. Specialist deployment cost and total-system savings remain unmeasured.

Implemented three small modules and synthetic regression tests. **495 tests passed** (124 new). A standard-library checker independently reproduced the thresholds, acceptance-set hashes and cost arithmetic. The real preparation passed with network, credentials, raw/processed dataset access and model loading forbidden; all counters were zero. An initial check rejected a harmless difference between original execution ID order and evaluation metadata order; exact predictions/populations matched, so validation now permits metadata permutation while rejecting missing/duplicate/foreign IDs, changed predictions or hashes. Dedicated regressions pass. No source evidence or threshold changed to resolve that implementation issue.

[Exact protocol, evidence, commands and limitations](../experiments/exp007-fixed-threshold-preparation/README.md). No official-test quality/coverage, calibrated-confidence, production-threshold, total-system savings or latency result is claimed. Next: implement and authorize the one-time frozen test evaluation, then report all outcomes without adapting this protocol to them.


## 2026-09-26 — repository synchronization and verified completed validation

**Completed and independently reproduced: EXP-006 + EXP-006R validation evaluation.** EXP-006R recovered all 52 eligible cases in 52 attempts with zero unresolved outputs. The merged evaluation preserves 718 original successes, substitutes only the original 52 HTTP-429 cases, and resolves all 770 IDs exactly once. The original EXP-006 remains unchanged at 718 ok/52 http_429; its 823 attempts and all preparation/history are preserved. This current completion record supersedes the earlier preparation-only status without rewriting it.

Luna standalone: **614/770 correct, 79.74% accuracy, 79.01% macro-F1**, with zero unresolved predictions. Reuse one general-response set across all 15 specialists and all 90 existing coverage combinations. Below, mean ± sample SD across the five seeds at the previously fixed 90% landmark (77 fallback cases per model):

| Shots/class | Specialist coverage | Routed accuracy (%) | Routed macro-F1 (%) | Fallback accuracy on rejected cases (%) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 693/770 (90%) | 77.14 ± 1.49 | 76.09 ± 1.54 | 63.38 ± 4.72 |
| 10 | 693/770 (90%) | 82.05 ± 0.89 | 81.47 ± 0.96 | 61.30 ± 2.50 |
| 20 | 693/770 (90%) | 85.27 ± 0.27 | 84.92 ± 0.38 | 60.26 ± 3.26 |

All 90 combinations, every seed and all predeclared landmarks are preserved in the compact evidence; this table does not replace them. Negative comparisons remain visible: the 5-shot 90% combination is worse than Luna alone. Luna's standalone accuracy is also below the 20-shot specialist-only mean (83.77%). The 20-shot 90% combination improves observed validation quality to 85.27% accuracy / 84.92% macro-F1, but this is an exploratory development comparison on the same reused 770 cases, not test performance or a production guarantee. The EXP-007 scalar thresholds reproduce its acceptance sets exactly; no threshold, label, model or prompt changed in this synchronization.

Accounting remains separate from quality completion: recovery usage-priced charges are **$0.0055932**; combined known usage-priced charges are **$0.0831327**. **Exact combined total spend remains unknown** because 105 original failed attempts have no returned usage. The saved conditional interval is $0.0831327–$0.31921220 under the frozen reservation assumptions, not an invoice reconciliation. Original failures are retained even though every final prediction is resolved. No actual cash-saving, total-system savings or production-latency claim follows.

The complete saved offline evaluation was reproduced exactly from audited local caches. Published compact predictions plus existing versioned EXP-005 records reproduce all quality metrics; derived per-attempt token/reservation records reproduce accounting without raw response bodies, customer texts, API request IDs, credentials, weights or private caches. [Completed-validation evidence and reproduction](../experiments/exp006-completed-validation/README.md). The same **770 additional validation labels** are reused; five seeds do not create five independent holdouts. No new API calls, model fitting or official-test access occurred during promotion.

EXP-007 preparation source, tests, frozen protocol and compact evidence are included in the same repository checkpoint. Thresholds remain unchanged for seeds 11/22/33/44/55: 0.10354116298070118 / 0.10243964501544885 / 0.1021902311171956 / 0.10160314104812371 / 0.10612054364389162. Each accepts exactly 693/770 validation IDs with no tied boundary. Frozen protocol SHA-256 remains `0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`.

The full suite was rerun for this checkpoint: **495 tests passed**. EXP-007's independent threshold/cost replay also passed. The official BANKING77 test stayed sealed; no paid API call was made. Historical preparation-only entries are retained as snapshots of their original milestones; current README/AGENTS/project-context/plan point to completed validation plus unrun test preparation.

The exact next implementation milestone is a guarded one-time test runner for that frozen protocol. The current `baseline.fixed_routing` CLI supports preparation only and has no test/live execution mode. Do not invent or execute such a command. Existing safe verification: `.venv/bin/python experiments/exp007-fixed-threshold-preparation/independent_check.py`. Only after the runner is implemented/tested and separately authorized may it unseal the test and collect one capped Luna response set for all 3,080 test IDs. This synchronization grants neither test access nor spending permission.


## 2026-09-26 — EXP-007 guarded runner implementation (review pending)

**Status: IMPLEMENTED WITH SYNTHETIC TESTS ONLY; OFFICIAL TEST SEALED; PAID TEST NOT RUN.** Four new modules implement the already frozen final-test protocol without modifying its bytes/hash (`0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`), thresholds, models, zero-shot prompt/schema, metrics or retry policy. No official test file was opened, including for checksum verification, and no real model weights were read during implementation. Existing research results/artifacts remain unchanged.

`baseline.final_protocol` checks five authorization controls before official-test file access and provides the trusted ID/text-only input boundary. `baseline.final_specialists` verifies local model/classifier/package hashes, reuses frozen encoder/LR models without fitting and durably persists probabilities/predictions/scalar gates. `baseline.final_collection` plans every payload and its full retry reservation before dispatch, collects one shared Luna response set, preserves all 3,080 IDs through explicit duplicate aliases, records failures/usage/unknown reservations and enforces the unchanged bounded serial retry policy. `baseline.final_test` provides sealed `dry-run`, guarded `preflight` and future `live` modes, verifies terminal collection evidence, freezes predictions before the scoring-only truth join and reports all seeds/full-denominator metrics.

**Full suite: 678 passed in 57.16s**, including 183 new synthetic tests across five files. The suite and sealed metadata-only dry run completed with real raw/processed data, ignored experiment artifacts, model weights and network access blocked; all forbidden-access counters were zero. Full-population synthetic integration covers 3,080 IDs, shared payload aliases, refused/failed responses, observed 50% fixture coverage (never forced 90%), no label influence on routing, crash/resume without resending successes, exhausted fourth-attempt interruptions and interrupted evaluation publication. These fixture outcomes are not research measurements.

Two crash edge cases were addressed before completion: an interrupted fourth reserved attempt is terminal failure rather than an excuse for a fifth attempt, and an atomic evaluation checkpoint allows missing evaluation/verification files to be materialized without rejoining labels or rerunning inference. Live collection also rejects a missing API key before creating a reservation; tests use a synthetic empty environment, never the real key.

The sealed estimate is unchanged: nominal **$0.3325308**, conditional one-attempt envelope **$7.06013000**, conditional four-attempt envelope **$28.24052000**. The latter assumes all test payloads fit the previously observed 4,817-byte validation maximum. Actual test lengths and required full-run reservations remain unknown until separately authorized preflight. The pricing-date flag attests that the operator verified unchanged official rates on the current UTC date; the runner does not claim to fetch a current quote. No cap or test access is authorized by this implementation. Paid API calls during implementation: **0**.

[Runner documentation, exact future commands and verification](../experiments/exp007-runner-preparation/README.md). Next: review the uncommitted implementation and commit it before any future launch; obtain separate explicit test/live authorization and numeric cap. Do not change code/Git revision during a live run or reset its directory/ledger. No commit/push or real preflight/live execution was performed in this implementation session.


## 2026-09-26 — EXP-007 permission-scope review (not checkpointed)

**Scope separation changed; frozen experimental protocol unchanged.** The initial runner's offline preflight unnecessarily required live/API permission. The protocol's `fallback.preflight` permits separately authorized unsealing to compute reservations before paid calls; standalone preflight now uses that test-inspection scope. `one_time_sequence[0]` continues to govern live evaluation, where both approvals remain mandatory. No protocol bytes, thresholds, models, prompt/schema, retry policy or evaluation definition changed.

`dry-run` requires no approval, stays sealed and cannot call APIs. `preflight` requires the exact approved protocol hash plus explicit test access; it needs no live/API approval, cap, key or pricing acknowledgement. It computes every payload/reservation requirement under frozen rates, and an optional positive cap is reported as sufficient/insufficient without authorizing spending. `live`, direct collection, every API attempt and scoring access still require all existing live controls. An insufficient live cap still fails closed.

**Final full suite: 720 passed in 53.76s**, including 42 new synthetic scope regressions. Those tests prove that an offline preflight cannot reach model inference, collection or scoring, and cannot grant later live permission. Safe `dry-run` CLI execution passed under audit guards; protected-data/model and network access counters were zero. No real preflight or live execution, official test read/hash, real model-weight access, credential inspection or API call occurred. The sealed projection remains conditional at $28.24052000; no actual test payload/cost is measured.

The first full review run also passed all 720 tests, but its audit hook then misclassified pytest's directory-relative temporary cleanup as a repository access. The final rerun used an explicit isolated synthetic-fixture directory, passed without that cleanup error, and left the protection enabled throughout. This was verification-harness behavior, not a test-data access or runner/protocol change.

Current commands and scope rationale are recorded in `experiments/exp007-runner-preparation/README.md`; updated `verification.json` and `dry_run.json` preserve the review evidence. Working-tree changes were inspected for secrets, real response bodies, raw data, weights, caches and accidental files; only source/tests/docs/compact preparation evidence are included. No files were staged, committed or pushed. Next: review/checkpoint the runner; a later preflight needs separately granted test-access permission, and a later live run additionally needs explicit API approval, a positive approved cap and current pricing acknowledgement.


## 2026-09-26 — EXP-007 runner preparation checkpoint

The reviewed runner, six synthetic test modules, documentation and compact preparation evidence are included in this checkpoint. No runner feature, source, test, frozen setting or evaluation definition changed during checkpointing; only current documentation status was advanced. Earlier implementation/review entries and their verification manifest remain historical evidence.

Reused the latest **720 passed / 0 failed** full-suite result after verifying every recorded source/test SHA-256 against `experiments/exp007-runner-preparation/verification.json` and confirming that existing tracked source, tests and dependencies were unchanged. The frozen protocol SHA-256 remains `0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`. The staged scope excludes credentials, raw data/responses, model weights and private caches.

The official test remains sealed: no official-test file was opened or hashed, no real preflight/live evaluation was executed, and no inference API calls were made. Next: separately authorize offline test-access preflight; live evaluation additionally requires explicit spending authorization, a sufficient numeric cap and current pricing acknowledgement. This checkpoint grants none of those permissions.
