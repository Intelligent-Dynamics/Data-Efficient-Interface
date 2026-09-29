# Verified reproducible results

Current status (2026-09-29 UTC): **v1 DONE.** Both EXP-007 and EXP-009 official-test evaluations are complete. The new EXP-009 companion is a single-seed result; EXP-007 retains its five-seed summary separately. Validation and official-test results below are separate populations. Dated earlier preparation, interrupted-run and negative results are preserved as history, not current instructions to run more experiments.

## EXP-009 — official-test retrieved-example companion (2026-09-29)

**Retrieved-example routing reached 88.084416% accuracy and 87.982622% macro-F1 while sending 268/3,080 requests (8.701299%) to Luna.** This improves accuracy by **2.500000 percentage points** and macro-F1 by **2.635497 points** over the matched seed-11 specialist. Specialist coverage is exactly **0.912987012987013** (2,812 accepted). The frozen scalar threshold is applied independently, without ranking or forcing test coverage.

All five rows below use the same 3,080 held-out cases and the **same seed-11, 20-shot training pool** where applicable. The specialist and zero-shot controls are reused from EXP-007's seed-11 results; they are not EXP-007's five-seed averages. Luna-only has one shared response set per prompt condition, not five independently sampled models. This is a single-seed/single-pool companion comparison, with no seed SD or confidence interval claimed.

| System (matched seed-11 comparison) | Accuracy (%) | Macro-F1 (%) | Correct / 3,080 | LLM use under policy |
| --- | ---: | ---: | ---: | ---: |
| Specialist | 85.584416 | 85.347125 | 2,636 | 0% |
| Zero-shot Luna | 81.363636 | 80.585867 | 2,506 | 100% |
| Zero-shot hybrid | 86.525974 | 86.389275 | 2,665 | 8.701299% |
| Retrieved-example Luna | 92.175325 | 92.134813 | 2,839 | 100% |
| Retrieved-example hybrid | 88.084416 | 87.982622 | 2,713 | 8.701299% |

The retrieved prompt uses exactly 20 examples from the already labeled 1,540-example training pool, chosen with the frozen MiniLM retrieval procedure. Label budget: **1,540 task training labels + 770 additional validation labels**, with external encoder pretraining. No extra fitting or validation/test demonstrations were added for EXP-009.

| Paired difference on the same seed-11 controls | Accuracy change (pp) | Macro-F1 change (pp) |
| --- | ---: | ---: |
| Retrieved Luna − zero-shot Luna | +10.811688 | +11.548947 |
| Retrieved Luna − specialist | +6.590909 | +6.787688 |
| Retrieved hybrid − specialist | +2.500000 | +2.635497 |
| Retrieved hybrid − retrieved Luna | −4.090909 | −4.152191 |
| Retrieved hybrid − zero-shot hybrid | +1.558442 | +1.593347 |

Retrieved Luna is substantially stronger than zero-shot Luna **under these frozen prompt conditions on this benchmark**. It is also more accurate than the routed system: accepting specialist answers trades some quality for fewer LLM requests. On the same 268 rejected requests, retrieved fallback accuracy is **0.753731343283582** (202 correct), versus zero-shot fallback **0.5746268656716418** (154 correct). This is complementary routing, not a claim that Luna is universally stronger. The observed 8.7012987012987% describes offline policy replay; the experiment actually queried all 3,080 cases for each standalone Luna condition.

**Accounting:** all 3,080 retrieved outputs are `ok`; 3,083 attempts include three `transport_unknown` attempts that later resolved. Known usage-priced API charges are **$0.576343155**. Missing usage carries a **$0.0084480** reservation, so the frozen-assumption spend interval is **[$0.576343155, $0.584791155]**. Actual invoice spend remains **unknown / not reconciled**. Reservations are not additional known charges. These are experiment charges, not production-system costs or savings; specialist deployment cost and production latency remain unmeasured.

**Verification and provenance:** compact [evidence and offline replay](../experiments/exp009-official-test/README.md) preserve all original metric values, per-class counts, paired differences, accounting and source hashes. Replay recomputes accuracy/macro-F1 from saved per-class sufficient statistics; aligned saved predictions, gates and prediction marginals are separately checked. Maximum independent arithmetic difference is **1.1102230246251565e-16**; the original saved metric floats remain unchanged. This checkpoint does not rejoin test labels or rerun inference. All 3,092 original EXP-009 artifact files remain byte-identical. The original prediction freeze is `c2963a59ca83911a9ccacf9390390fed18fb9bb5282bf38ae1b33b5c44be2a8c`, and protocol remains `b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334`.

**Limits:** one seed and one training pool do not establish seed robustness for the retrieved condition. EXP-007's five seeds share this same test population, and their SD is not a confidence interval; the two study summaries do not have identical uncertainty. Public-benchmark pretraining exposure and near-duplicate contamination cannot be ruled out. The test is now evaluated; no further tuning is justified by these outcomes. Production dollar savings are not established. Calibration, SetFit/fine-tuning, a second dataset (CLINC150), self-hosted vLLM/H100 and production cost measurement are deferred to v2.

## EXP-007 — official-test complementary routing (2026-09-28)

The frozen 20-shot MiniLM/LR specialists use 1,540 fitting labels each plus the same 770 development labels, with external MiniLM pretraining. All five seeds are evaluated on the same 3,080 official BANKING77 cases using fixed validation-derived thresholds. One all-case zero-shot Luna prediction set is reused across seeds; no test ranking or forced coverage. Results are mean ± **sample SD** across training seeds, with SD in percentage points; Luna has one shared result.

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Luna-only | 81.363636% | 80.585867% |
| Specialist-only | 85.435065% ± 0.317784 | 85.190044% ± 0.328684 |
| Hybrid | 86.487013% ± 0.435235 | 86.298575% ± 0.448328 |

The hybrid adds **1.051948 accuracy percentage points** over specialist-only (paired seed SD **0.190150 pp**), while the policy sends only **9.012987% ± 0.329989 pp** to Luna. Mean specialist coverage is **90.987013%**. All five seeds improve; no best-seed selection. Macro-F1 gain is **1.108531 pp**.

| Seed | Specialist accuracy / macro-F1 (%) | Hybrid accuracy / macro-F1 (%) | Specialist accepted | Luna fallback | Fallback accuracy (%) | Accuracy gain (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 85.584416 / 85.347125 | 86.525974 / 86.389275 | 2812 | 268 (8.701299%) | 57.462687 | +0.941558 |
| 22 | 85.097403 / 84.794841 | 86.266234 / 86.066432 | 2810 | 270 (8.766234%) | 53.333333 | +1.168831 |
| 33 | 85.259740 / 85.014634 | 86.071429 / 85.843303 | 2805 | 275 (8.928571%) | 53.818182 | +0.811688 |
| 44 | 85.324675 / 85.137002 | 86.363636 / 86.174694 | 2798 | 282 (9.155844%) | 55.673759 | +1.038961 |
| 55 | 85.909091 / 85.656617 | 87.207792 / 87.019170 | 2787 | 293 (9.512987%) | 55.290102 | +1.298701 |

Mean accuracy on matching rejected subsets is **55.115613%** for Luna and **43.466706%** for the specialist. Luna is weaker overall but makes useful complementary errors; it is not established as universally stronger or frontier. Earlier **validation** accuracy over 770 reused cases was specialist 83.7662%, Luna 79.7403%, hybrid 85.2727% at 90% specialist coverage; those are development findings, not these official-test estimates.

**Accounting:** 3,080 final outputs resolved; 3,081 attempts, including one HTTP 503 followed by a successful retry. Known usage-priced charges **$0.332622650**; the failed attempt lacks usage, so the frozen-envelope spend interval remains **$0.332622650–$0.334866150**, not an exact invoice total. The experiment queried all 3,080 cases for the standalone baseline; ~9% is the policy fallback fraction. Reservations are not additional charges.

Five seeds share the same test population and Luna predictions; seed SD is **not a confidence interval**. No total production cost/savings or production latency is measured. At this EXP-007 checkpoint, EXP-009 was pending; its completed single-seed companion is now reported above. No tuning follows either test. All per-class metrics, seed results, raw-count arithmetic, accounting and provenance are in the [compact official-test evidence](../experiments/exp007-official-test/README.md). Independent replay matched all saved numeric metrics exactly (maximum difference **0.0**); it used saved evaluation labels and never reopened the raw test CSV, reran inference or called an API. The original protocol remains `0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`.


## EXP-002 — validation learning curve (2026-09-24)

EXP-002 uses the unchanged word unigram/bigram TF-IDF + L2 multinomial logistic-regression pipeline at clean Git commit `30645360724d7fb8fe0c5a11d7dd7e02f3ac339b`. Dataset: BANKING77 revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Split protocol: `banking77-val10-v2`, seed 20260924, all 77 classes, the same 770 validation examples. Training seeds: 11, 22, 33, 44, 55; N=5/10/20. No model settings, splits, or seeds were selected using these scores.

| Shots/class | Train/run | Accuracy mean ± SD (%) | Macro-F1 mean ± SD (%) | Macro-F1 min–max (%) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 385 | 51.84 ± 1.69 | 50.41 ± 1.58 | 48.77–52.59 |
| 10 | 770 | 62.49 ± 1.60 | 61.58 ± 1.81 | 60.05–64.14 |
| 20 | 1,540 | 69.95 ± 0.61 | 69.19 ± 0.63 | 68.18–69.80 |

Accuracy and F1 are shown as percentages. SD is sample standard deviation across five training seeds (`ddof=1`). Minimum/maximum F1 includes all five seeds. Exact values and per-seed records: [study evidence](../experiments/exp002-learning-curve/README.md), [summary JSON](../experiments/exp002-learning-curve/summary.json), [per-run CSV](../experiments/exp002-learning-curve/per_run.csv).

![Validation learning curve](../experiments/exp002-learning-curve/learning_curve.png)

Both increments improved accuracy and macro-F1 for every paired seed. Mean macro-F1 gains were **+11.17 percentage points** from 5→10 (seed gains 9.81–11.71) and **+7.61 points** from 10→20 (4.04–9.50). Accuracy gains were +10.65 and +7.45 points. These are substantial, consistent development-set gains; no formal population significance claim is made.

Macro-F1 seed SD is **1.58 → 1.81 → 0.63 percentage points**: variance does not decrease monotonically, but is much lower at 20 shots. Accuracy SD is **1.69 → 1.60 → 0.61 points**. Seed variability is not validation-sampling uncertainty.

The slope is flattening, but the curve does **not yet show a clear plateau**: 10→20 still adds 7.61 macro-F1 points on average. Diminishing returns are especially visible per added example/class (approximately 2.23 F1 points for 5→10 versus 0.76 for 10→20). Three budgets cannot establish an asymptote.

## Reproducibility and scope

All 15 primary runs and 15 independent local refits completed without warnings or failures. Refits used separate clean output directories, reproduced sampled IDs, predicted labels, and complete quality metrics exactly, and had zero maximum absolute difference in classifier coefficients/intercepts. Only the 15 primary runs enter the aggregates. This is local reproducibility, not independent external or cross-platform replication.

The verifier checked every required artifact and recorded hash, dataset/split/configuration agreement, exact N-per-class training membership, train/validation/test ID and normalized-text isolation, and shared validation order/labels. Each saved TF-IDF vocabulary and IDF vector exactly matched a vectorizer rebuilt using only its intended training texts. Saved models reproduced their validation predictions/probabilities. Metrics were recomputed from all saved predictions. The aggregate summary, error-analysis JSON, and per-run CSV were independently regenerated byte-for-byte from the versioned records. No official test rows were parsed during this study; test access was limited to pinned byte-integrity checks. No test evaluation occurred.

The complete suite passed **35 tests**, including checks that aggregation rejects missing/duplicate seeds, changed validation IDs/configurations, fabricated metrics, and ID leakage, and uses sample rather than population SD. Verification details: [verification JSON](../experiments/exp002-learning-curve/verification.json). Plot axes, seed points, and error bars were visually inspected.

Across all primary runs, 5,394 unique training rows and 770 validation rows were used (6,164 distinct fitting/validation labels). Per-run budgets are documented in the plan; all source-training labels accessed for auditing/stratification are disclosed separately. The official test file remains unchanged and unevaluated.

## Measured intent weaknesses and confusions

Across all regimes and seeds (equal weight per model), the lowest mean recalls were:

| Intent | Mean validation recall across all 15 primary models (%) |
| --- | ---: |
| `topping_up_by_card` | 26.00 |
| `supported_cards_and_currencies` | 26.00 |
| `transfer_fee_charged` | 26.00 |
| `card_delivery_estimate` | 28.00 |
| `unable_to_verify_identity` | 28.67 |

At 20 shots, the lowest recalls and all highest-recall ties were:

| 20-shot intent | Mean recall (%) | Mean F1 (%) |
| --- | ---: | ---: |
| `transfer_fee_charged` | 30.00 | 39.36 |
| `unable_to_verify_identity` | 32.00 | 34.17 |
| `supported_cards_and_currencies` | 32.00 | 39.50 |
| `wrong_exchange_rate_for_cash_withdrawal` | 36.00 | 48.74 |
| `topping_up_by_card` | 38.00 | 42.63 |
| `verify_source_of_funds` | 100.00 | 92.64 |
| `apple_pay_or_google_pay` | 100.00 | 99.05 |
| `lost_or_stolen_phone` | 100.00 | 99.05 |

| True intent → predicted intent (20-shot) | Errors across five seeds | Distinct validation requests involved |
| --- | ---: | ---: |
| `get_disposable_virtual_card` → `disposable_card_limits` | 19/50 | 4/10 |
| `getting_virtual_card` → `virtual_card_not_working` | 17/50 | 6/10 |
| `unable_to_verify_identity` → `why_verify_identity` | 16/50 | 7/10 |
| `topping_up_by_card` → `top_up_by_cash_or_cheque` | 15/50 | 5/10 |
| `wrong_exchange_rate_for_cash_withdrawal` → `card_payment_wrong_exchange_rate` | 15/50 | 4/10 |

Full class metrics/confusion matrices are in [error analysis](../experiments/exp002-learning-curve/error_analysis.json). These counts reflect repeated predictions on the same fixed requests; they do not multiply the validation sample size.

## Limitations

These are validation results on only 10 fixed examples per class, reused across training seeds and regimes. SD uses `ddof=1` across the five training seeds; it is neither a confidence interval nor an estimate of uncertainty over new traffic. The legacy validation=20 smoke is excluded. Class-level 100% recall is an observation on a tiny holdout, not a reliability guarantee. Confusion counts repeat the same requests across models; 50 prediction events per class/regime represent only 10 distinct validation requests. Training subsets overlap across seeds, and validation was available in earlier development, so these are development comparisons rather than a fresh confirmatory evaluation. Near-duplicate leakage beyond the implemented normalized-exact audit remains untested. No production-quality, routing, calibration, or cost-savings claim is supported.


## EXP-004 — frozen MiniLM versus TF-IDF validation curve (2026-09-25)

**Completed and independently reproduced locally.** Same 5/10/20-shot budgets, seeds 11/22/33/44/55 and exact EXP-002 v2 train/validation IDs. Model revision/settings and LR hyperparameters are unchanged from EXP-003; the encoder stayed frozen. No tuning or official-test evaluation.

Values below are mean ± sample SD across five primary training seeds. Refit runs are excluded from these aggregates.

| Shots/class | MiniLM accuracy (%) | MiniLM macro-F1 (%) | TF-IDF accuracy (%) | TF-IDF macro-F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 74.13 ± 1.30 | 72.60 ± 1.34 | 51.84 ± 1.69 | 50.41 ± 1.58 |
| 10 | 80.05 ± 0.93 | 79.19 ± 1.03 | 62.49 ± 1.60 | 61.58 ± 1.81 |
| 20 | 83.77 ± 0.66 | 83.35 ± 0.68 | 69.95 ± 0.61 | 69.19 ± 0.63 |

Macro-F1 improvements against each matching TF-IDF run (percentage points):

| Shots/class | Seed 11 | Seed 22 | Seed 33 | Seed 44 | Seed 55 | Mean gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | +23.33 | +22.74 | +20.39 | +24.01 | +20.49 | +22.19 |
| 10 | +16.87 | +17.64 | +17.62 | +20.01 | +15.87 | +17.60 |
| 20 | +13.86 | +13.59 | +13.36 | +13.86 | +16.11 | +14.16 |

MiniLM improves macro-F1 for **all 15 matched training-budget/seed pairs**. Mean advantages over TF-IDF are +22.19, +17.60, and +14.16 percentage points at 5/10/20 shots. Absolute differences for every seed are retained above; no best seed was selected.

The MiniLM curve gains **6.58 macro-F1 points from 5→10** and **4.16 points from 10→20**. Returns diminish, but these three budgets do not establish a plateau. MiniLM macro-F1 seed SD decreases **1.34 → 1.03 → 0.68 points**, and accuracy SD decreases **1.30 → 0.93 → 0.66 points**. MiniLM's seed SD is lower than TF-IDF's at 5/10 shots but slightly higher at 20 shots. Macro-F1 ranges across all five seeds: 5-shot 70.94–74.46%; 10-shot 77.99–80.06%; 20-shot 82.65–84.29%.

![Paired validation learning curves](../experiments/exp004-minilm-learning-curve/learning_curve.png)

All 15 primary runs and all 15 independent local refits completed without warnings or failures. The reproduction stage reloaded the pinned frozen encoder and recomputed all 6,164 required vectors into a separate cache, then fitted new logistic-regression instances for every configuration. Embedding maximum absolute difference was **0.0** (predeclared tolerance 1e-6, relative tolerance zero). Every pair had identical sampled IDs, predicted labels, full metrics, and classifier coefficients/intercepts; maximum parameter difference was **0.0** for all 15. The 5-shot seed-11 run also reproduced EXP-003's predictions and metrics exactly. Only primary runs enter aggregates.

This is independent local re-encoding and refitting using the same implementation and machine, not independent external replication or uncertainty over new traffic. Saved classifier replay is an additional integrity check, not counted as another independent fit. Artifact/cache/source hashes, reference sample hashes, exact per-class training counts, fixed validation IDs/labels, normalized-text/ID isolation, feature alignment, unchanged encoder state, and classifier probability/prediction replay all passed. A separate NumPy calculation agreed with all reported means and sample SDs to maximum absolute difference 1.11e-16. Summary JSON, per-seed CSV and plot were regenerated byte-for-byte from compact primary records. **64 tests passed**; the figure was visually inspected.

Every run uses **770 additional validation labels** (10 per class), besides 385/770/1,540 training labels for 5/10/20 shots. Per-run totals are 1,155/1,540/2,310. Across the study, 5,394 unique training rows plus 770 validation rows give 6,164 unique task labels. These are the same examples used by EXP-002; the new representation and refits added no new unique task labels. All 10,003 official-training labels were mechanically read for audit/stratification and are separately disclosed. The encoder benefits from substantial external pretraining; this is simulated few-shot task adaptation, not total training-data or annotation-cost accounting.

The official BANKING77 test set stayed sealed. Test access was limited to byte-integrity checks; no test rows were parsed, labeled, inspected, encoded, tuned on, or evaluated. Existing stored test IDs/text hashes were used for isolation checks. The dataset and split manifest are unchanged.

SD is **sample SD across five training seeds (`ddof=1`)**, not a confidence interval or uncertainty over new validation samples. All runs share the same previously used 770-case holdout, with only ten examples per class; seeds share overlapping training data. These are paired development comparisons, not confirmatory unseen-test or production results. Public-benchmark exposure in upstream pretraining and semantic near-duplicate leakage cannot be ruled out. No calibration, routing, production-quality, or cost-savings claim is established.

Classifier timings use cached embeddings and **are not end-to-end inference measurements**. Encoding/cache construction is recorded separately; end-to-end speed and costs are unmeasured.

Reproducible evidence: [study README](../experiments/exp004-minilm-learning-curve/README.md), [summary JSON](../experiments/exp004-minilm-learning-curve/summary.json), [all seed values and paired improvements](../experiments/exp004-minilm-learning-curve/per_seed.csv), [independent refit checks](../experiments/exp004-minilm-learning-curve/verification.json), and [artifact/aggregate audit](../experiments/exp004-minilm-learning-curve/analysis_audit.json).


## EXP-006R — preparation status only (2026-09-26)

**Paid recovery NOT RUN; no recovered research results are available.** A separate, tested recovery protocol is prepared for exactly the original EXP-006's 52 unresolved HTTP-429 IDs. All original EXP-006 files are preserved. Preparation used synthetic responses and a zero-inference dry run only; preparation API spend is $0. The quota/credit issue must be resolved and a new explicit recovery approval/cap obtained before live execution.

No recovered accuracy, macro-F1, combined quality, actual recovery charge or production saving is claimed. The original 770 validation IDs and additional validation-label budget remain unchanged, and the official test remains sealed. Any later merged result must be labeled **EXP-006 + EXP-006R recovery** and retain failures. [Preparation protocol and verification](../experiments/exp006r-429-recovery-preparation/README.md).


## EXP-007 — verified validation threshold reproduction (2026-09-26)

**Official test NOT RUN.** The existing 20-shot EXP-004/005 validation artifacts yield the following exact scalar thresholds. Each `confidence >= threshold` gate reproduces its corresponding prior 90% rank acceptance set, including membership, with no changed IDs:

| Seed | Fixed threshold (`>=`) | Validation acceptance | Coverage | Changed IDs |
| --- | ---: | ---: | ---: | ---: |
| 11 | 0.10354116298070118 | 693/770 | 90% | 0 |
| 22 | 0.10243964501544885 | 693/770 | 90% | 0 |
| 33 | 0.1021902311171956 | 693/770 | 90% | 0 |
| 44 | 0.10160314104812371 | 693/770 | 90% | 0 |
| 55 | 0.10612054364389162 | 693/770 | 90% | 0 |

No boundary tie occurs. Thresholds are stored losslessly as binary64 decimal/hex values and independently reproduced from saved confidences. The rule is label-blind at derivation/runtime, but the 20-shot/90% choice follows exploratory validation results. The same 770 additional validation labels were already used in development. No calibration, refitting, new labels, paid calls or official-test access occurred; 495 tests passed.

These are reproducible **validation acceptance-set checks**, not test/production quality or confidence calibration results. Future coverage must be observed, not forced to 90%. [Frozen protocol and evidence](../experiments/exp007-fixed-threshold-preparation/README.md), SHA-256 `0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`. No future API projection is promoted to actual spend or production savings.


## EXP-006 + EXP-006R — completed validation checkpoint (2026-09-26)

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


## Bounded v1 — independently replayed complementary validation errors (2026-09-27)

An independent Counter-based implementation reproduced all 90 saved combinations and every frozen 20-shot 90% acceptance set from versioned EXP-004/005/006+006R evidence, without model inference, raw data, private response caches, or test access.

| Accuracy on the reused validation set | Mean ± sample SD (%) |
| --- | ---: |
| 20-shot specialist alone | 83.7662 ± 0.6622 |
| Zero-shot Luna alone (one shared response set) | 79.7403 |
| 20-shot specialist + Luna, 90% specialist coverage | 85.2727 ± 0.2693 |
| Specialist on each matching 77-case rejected subset | 45.1948 ± 4.8069 |
| Luna on those same rejected cases | 60.2597 ± 3.2597 |

The hybrid gains **1.5065 ± 0.6660 accuracy points** and **1.5711 ± 0.7424 macro-F1 points** over its matching specialist. Seed 11/22/33/44/55 gains are respectively 8/14/18/13/5 correct answers out of 770. Luna rescues 111 specialist-error events and harms 53 specialist-correct events across the five overlapping rejected subsets, net +58. These are repeated model/case events, not independent new requests.

Negative comparisons remain: at 50% fallback, the 20-shot hybrid averages 82.57% accuracy, below specialist-only 83.77%; the 5-shot 10%-fallback hybrid is 77.14%, below Luna-only 79.74%. No claim that Luna is stronger overall or frontier is supported. All curves/seed values are retained in the [independent evidence and chart](../experiments/v1-complementary-validation/README.md).

Each 20-shot specialist uses 1,540 fitting labels plus the same 770 additional validation labels; the encoder has external pretraining. Five seeds do not create independent validation sets, and sample SD is not a confidence interval over new traffic. This is complementary-error evidence on development data, not final-test quality, a production guarantee, or total-system dollar savings.


## EXP-008 — measured local CPU pipeline (2026-09-27)

The representative **20-shot seed-11** specialist was measured as one deployed model on the same 770 validation texts: Apple M1 Pro (10 cores, 32 GiB), macOS 26.5.1, Python 3.13.0, CPU only, PyTorch/native libraries single-threaded. Encoder/classifier weights, packages, normalization and the scalar threshold were unchanged. No fitting or new labels. Protocol `1a1c02fd3d1a72ddfa45cd844e47b8602cc143f8649919fe0b2700509ca6043e` was written before collecting timings.

Each batching mode had one untimed full validation pass, then five measured passes (3,850 request events per mode). The timed path includes raw-text tokenization, MiniLM, normalization, LR probabilities and fixed routing, without cached features, network or disk logging.

| Measurement | Observed value |
| --- | ---: |
| Warm single-request median | 5.906833 ms |
| Warm single-request p95 | 7.824673 ms |
| Batch-32 pooled throughput | 370.1814 requests/s |
| Batch-32 per-pass throughput range | 307.8937–404.1805 requests/s |
| Local encoder loading | 2.424610 s |
| Classifier loading | 0.629625 ms |
| Process RSS after model loading | 513.53125 MiB |
| Lifetime process peak RSS | 582.921875 MiB |

RSS is sampled with `ps` (KiB converted to bytes); the lifetime high-water mark comes from `resource.ru_maxrss` (bytes on macOS). This includes Python, libraries and evidence, not isolated model memory. Loading excludes hash verification and may benefit from warm filesystem caches; it is not a cold-process startup claim. Batch throughput is not interactive latency. All pass/per-request/batch durations and resource snapshots are retained.

Every label and routing decision matched saved validation evidence in all ten timed passes: exactly 693/770 accepted, zero decision mismatches. Maximum absolute probability differences were `2.51655065e-7` at batch 1 and `2.77835884e-7` at batch 32; no confidence value or threshold was rounded or adjusted. An independent standard-library replay reproduced the timing summaries. Access guards recorded zero official-test and network attempts. [Full measurements, methodology and reproduction](../experiments/exp008-cpu-validation/README.md).

A real setup failure occurred before timings: the original final-runner loader rejects the pinned library's inert `{'query': '', 'document': ''}` prompt mapping. The new benchmark-only loader permits only empty prompt strings and no default/applied prompt, while checking exact frozen weights and settings. The original runner is unchanged; it needs a minimal compatibility review before final execution. These measurements are specific to this machine and workload; no cloud price, free-compute assumption, production latency or total-system saving is inferred.


## EXP-009 — measured offline retrieval overhead; no LLM quality result (2026-09-27)

One serial, single-threaded pass over the 770 existing validation query embeddings measured exact retrieval median/p95 **0.750563/0.853890 ms** and prompt construction plus isolation checks **2.051438/2.270002 ms**. Query vectors were reused from the verified frozen cache; these values exclude encoding and are not end-to-end inference or service latency. Every query duration is retained in [the companion evidence](../experiments/exp009-retrieved-luna/overhead.json).

The prepared baseline retrieves exactly 20 examples by normalized-vector cosine similarity from only the 1,540 original seed-11 20-shot training IDs; ties use ascending training ID. All 1,540 training labels count, plus the reused 770 validation labels. The companion preserves original encoder, generation settings, strict intent schema and fixed threshold. **Paid pilot/validation/test evaluation NOT RUN:** no retrieved-Luna quality, fallback gain, actual API charge or total-system saving is claimed. Historical estimates are preparation assumptions documented in the experiment log, not measured spend.
