# Project context

Updated: 2026-09-28 UTC. Organization: Intelligent Dynamics.
Project: **Data-Efficient Specialist Inference**.
Canonical checkout: `/Users/Andrew/Developer/data-efficient-inference`. The existing checkout, including Git history and local artifacts, was moved here from the former Documents/ChatGPT path on 2026-09-24. Origin remains `https://github.com/Intelligent-Dynamics/data-efficient-inference.git`.

## Current completed state

**2026-09-28 UTC: EXP-007 official-test evaluation is complete.** The same frozen five 20-shot MiniLM/LR specialists, validation-derived scalar thresholds and zero-shot Luna prompt were evaluated on all **3,080 BANKING77 test cases**. Mean specialist accuracy/macro-F1 is **85.435065% / 85.190044%**, Luna-only is **81.363636% / 80.585867%**, and mean hybrid is **86.487013% / 86.298575%**. The hybrid adds **1.051948 accuracy points** over specialist-only, using Luna on **9.012987%** of requests on average. Observed specialist coverage is **90.987013%**, never forced to 90%. All five seeds improve. [Compact test evidence and independent replay](../experiments/exp007-official-test/README.md).

All 3,080 Luna outputs are resolved in 3,081 attempts. One HTTP-503 attempt has unknown usage; usage-priced charges are **$0.332622650**, with a bounded interval of **$0.332622650–$0.334866150**. These are frozen-rate charges for an all-case collection, not invoice-reconciled spend or the cost of a hypothetical routed deployment. Specialist deployment cost, total-system savings and production latency remain unmeasured.

Each specialist used 1,540 fitting labels plus 770 repeatedly used validation/development labels, with external MiniLM pretraining. Five seeds share one test population and one Luna set; sample SD measures training-seed variability, not a confidence interval over new requests. Public-benchmark contamination and production distribution differences cannot be ruled out. No threshold, prompt or model changes follow test scores. EXP-009's already-frozen retrieved-example test comparison remains pending and needs separate authorization.

Validation is separate: EXP-006R recovered all 52 unresolved requests and EXP-006 + EXP-006R completed all 770 validation IDs/all 90 combinations. Luna validation accuracy/macro-F1 is **79.74% / 79.01%**, versus **85.27% / 84.92%** for the exploratory 20-shot 90%-coverage hybrid. Those reused-development findings selected the fixed policy; they are not held-out test estimates. The original 718-success/52-HTTP-429 execution and unknown usage from 105 failed attempts remain preserved in the [validation evidence](../experiments/exp006-completed-validation/README.md).

EXP-007 threshold preparation matched 693/770 validation acceptance for every seed with no boundary ties. The subsequent loader/cooldown compatibility fixes and live resume retain their separate records; neither frozen protocol changed. The interrupted 509-response checkpoint is historical, superseded by the completed immutable prediction bundle. This synchronization uses saved evidence only: it does not rerun inference, call APIs, modify protocols or run EXP-009. The exact next conditional companion preflight is in [CURRENT_PLAN.md](CURRENT_PLAN.md); it reads/encodes test inputs but makes no API calls.

## Thesis and research question

Intelligent Dynamics explores data-efficient AI: specialized systems that adapt to narrow tasks with limited labeled data and reduce unnecessary reliance on expensive general-purpose models.

**Can confidence-based routing combine a few-shot specialist with a complementary LLM to improve classification quality while reducing LLM calls, and what local inference overhead does this introduce?**

The hypothesis is that a specialist and an LLM can make complementary errors. Overall model ranking does not determine performance on the specialist’s rejected subset. Luna performs worse than the 20-shot specialist overall on validation and official test, yet its complementary predictions improve the frozen hybrid on both populations. The official-test comparison now measures this gain; it does not establish a universal model ranking or deployment guarantee. Confident errors, fallback deterioration and local compute overhead remain important negative findings.

Operating principle: **Measure → understand → improve → re-measure.**

The bounded v1 report states measured specialist/routed/comparator accuracy and macro-F1, observed specialist coverage and LLM call fraction, training and development label budgets, and local CPU overhead. The original cost-saving thesis remains historical motivation; total-system dollar savings require a defensible deployment-cost basis that this study does not yet have. No reduction in API calls is automatically a reduction in total cost.

## Historical bounded v1 update (2026-09-27)

This checkpoint preceded the completed official test above. Its then-pending execution status and proposed pilot remain historical.

EXP-008 now measures the existing seed-11 model at 5.91 ms median / 7.82 ms p95 warm CPU request latency and 370.18 requests/s batch-32 throughput on this Apple M1 Pro; process memory and load times are separate. EXP-009 is implemented and prepared but has no paid response or quality result. Its cached-vector retrieval/prompt median overhead is 0.751/2.051 ms. Inherited prices are historical; $0.26 is only a proposed unapproved pilot cap. The original runner's inert-empty-prompt compatibility assertion is now fixed and verified offline on all five existing validation specialists, with zero probability, label or routing differences. Only that assertion changed; original and companion protocols and previous evidence remain unchanged. Separate evidence is in `experiments/exp007-loader-compatibility/`.

The then-current scope added CPU-only seed-11 timing and one retrieved-example Luna comparison (seed 11 only) before any final-test execution. Exactly 20 demonstrations are retrieved from the 1,540 seed-11 training IDs using the same normalized frozen MiniLM; all 1,540 available task labels count, in addition to the reused 770 validation labels. No k/prompt/threshold/model search is allowed. The 20-query pilot is for formatting, usage and cost only. Original EXP-007 bytes and all prior experimental evidence remain unchanged; the companion requires its own approval. Historical sections below describe their original milestones, including then-sealed test status.

## Initial repository inspection

At the initial inspection on 2026-09-24 the repository contained only Git metadata, no tracked files, and no commits on `master`. There was no implementation, dependency configuration, test suite, dataset, or prior result. The first session established documentation only.

## Implementation state (2026-09-24)

The local baseline now has pinned dataset downloads/dependencies, normalized-duplicate auditing, fixed validation and nested training samples, TF-IDF + logistic regression, a dummy comparator, JSON provenance/metrics/predictions, and automated tests. EXP-002 has completed the full 5/10/20-shot learning curve across five seeds, with all 15 configurations independently refitted locally and matched exactly. Reproducible validation aggregates and their limitations are in `RESULTS.md`; the earlier validation=20 smoke remains separate historical evidence. There is no routing, calibration, cost model, API, or frontend implementation.

The initial validation=20 design left too few examples in the smallest class (35 usable source rows). At the user's direction, protocol `banking77-val10-v2` now reserves a fixed **10 examples per class** from official training data for validation/calibration, leaving at least 25 training candidates per class. Initial regimes are **5, 10, and 20 shots**, all with the same 770 held-out validation examples and sampling seeds 11/22/33/44/55. Fifty-shot is dropped: even without a holdout the smallest class cannot supply 50 unique examples. All 77 classes, duplicate/ID isolation, deterministic nested sampling, and the sealed official test split are preserved. All training-source labels read for stratification/auditing are disclosed separately from sampled fitting/validation labels; this is a simulated few-shot benchmark, not proof of total annotation requirements.

The existing N=5/seed=11 smoke run used the old validation=20 protocol. Keep its artifacts and metrics as historical evidence; do not compare or aggregate its score directly with new-protocol results. The revised protocol now has 15 primary validation runs and 15 independent local refits. Mean macro-F1 rises from 50.41% to 61.58% to 69.19% at 5/10/20 shots; these are development-set measurements, not test or production results. No routing/cost claim is established.

Ten validation examples per class (770 overall) are a reasonable development budget for coarse comparisons among a small, prespecified set of simple baselines. They do not guarantee sensitivity to small differences: each additional correct case changes a class's recall by 10 percentage points, and macro-F1 can be noisy. Reusing this fixed holdout for many tuning decisions risks overfitting; report paired development comparisons and training-seed variability, not definitive quality claims. For later calibration, this is a shared held-out development pool, not independent calibration and evaluation sets. Do not fit calibration or select thresholds on these labels and then call performance on the same labels unbiased. Reliable per-class calibration, high-confidence tail estimates, and narrow non-inferiority claims will need a separately designed, label-budgeted calibration/evaluation procedure (additional development labels or suitable cross-fitting). The official test split remains sealed during that work.

## Frozen representation milestone (2026-09-25)

The user chose a frozen sentence-embedding comparison next, superseding the proposed 25-shot lexical extension. EXP-003 implements `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with only logistic regression trained. Its initial classifier settings are identical to TF-IDF. CPU encoding produces 384-dimensional normalized vectors; encoder weights remain frozen and unchanged. Dataset, cleaning, fixed validation, training IDs, and all classical results are preserved.

At completion of EXP-003, exactly one 5-shot seed-11 smoke had run using the existing EXP-002 IDs. Its comparison was recorded in `EXPERIMENTS.md` without promotion to verified research results. Artifact replay and 47 tests passed; independent embedding training reproduction and a multi-seed study were still pending. The proposed single-run reproduction was subsequently superseded by EXP-004 below.

The label budget is 385 task-training labels **plus 770 validation labels**, reused from the TF-IDF comparator. External encoder pretraining is an additional source of knowledge: this is not representation learning from 385 labels alone, and overlap with public benchmarks cannot be excluded. The pinned [publisher model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md) describes its pretraining. Keep the single-seed result exploratory, disclose this holdout's repeated use, and preserve the sealed test. No paid inference, fine-tuning, routing, calibration, frontend or cost model was added.

## Full frozen-representation learning curve (2026-09-25)

At the user's direction, EXP-004 completed the full 5/10/20-shot × five-seed frozen-MiniLM curve and independently recomputed all embeddings/refitted every classifier. This supersedes the proposed one-run reproduction while preserving EXP-003's smoke artifact. Same exact v2 IDs, model revision/settings and LR hyperparameters; no tuning. All 30 fits succeeded without warnings, and all independently reconstructed embeddings, classifier parameters, predictions and metrics matched exactly. The existing lexical and smoke pipelines remain unchanged; 64 tests pass.

Mean macro-F1 is 72.60%, 79.19%, and 83.35% at 5/10/20 shots (sample SD 1.34, 1.03, 0.68 points). Frozen MiniLM exceeds the matching TF-IDF run for every seed, with mean paired advantages of 22.19, 17.60, and 14.16 points. These verified, reproducible **validation** comparisons are in `RESULTS.md`; they do not establish test, production, calibration, or cost performance. The full matrix reuses 5,394 unique training and 770 validation labels from EXP-002, while retaining the external-pretraining and source-label-audit disclosures.

Caches store frozen representations, not classifier training on the whole union. Each LR fit sees only its specified few-shot subset. Encoding time and cached classifier time are recorded separately; end-to-end inference speed is unmeasured. Independent cache reconstruction and classifier refits provide local reproducibility, not external replication. The official test remains sealed. The proposed next milestone is paired error analysis using saved predictions before more model complexity or calibration design.

## Dataset comparison

These are publisher-reported dataset properties, checked on 2026-09-24, not locally verified experimental results. Difficulty and commercial relevance below are our qualitative assessments.

| Dataset | Commercial relevance | Classes | Published size and splits | Few-shot suitability | Licensing | Expected difficulty |
| --- | --- | --- | --- | --- | --- | --- |
| BANKING77 | Strong fit for banking customer-support intent classification | 77 intents in one domain | 13,083 total; 10,003 train, 3,080 test; no official validation split | Good for controlled per-class subsampling and learning curves; closely related intents test specialization | Publisher repository: CC BY 4.0; retain attribution, license, and modification notices | Fine-grained distinctions can be hard with few lexical examples; no explicit out-of-scope evaluation |
| CLINC150, `data_full.json` | Broad assistant/service intents; less focused on one business workflow | 150 supported intents plus out-of-scope queries | 22,500 in-scope: 15,000 train / 3,000 validation / 4,500 test. Additional 1,200 out-of-scope: 100 / 100 / 1,000. Total 23,700 | Balanced in-scope classes support subsampling; official validation and out-of-scope examples support later abstention studies | Original GitHub LICENSE: CC BY 3.0. UCI lists CC BY 4.0; record the license shipped with the chosen source rather than treating them as interchangeable | More classes and broad domain variation; out-of-scope detection adds difficulty beyond closed-set classification |

Sources: [BANKING77 publisher and split counts](https://github.com/PolyAI-LDN/task-specific-datasets), [BANKING77 license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE), [CLINC150 publisher and split definitions](https://github.com/clinc/oos-eval), [CLINC150 original license](https://github.com/clinc/oos-eval/blob/master/LICENSE), [UCI counts and license metadata](https://archive.ics.uci.edu/dataset/570/clinc150).

**Recommendation: start with all 77 BANKING77 intents.** It most directly tests narrow business specialization, has a manageable public benchmark, and avoids choosing an easy subset after seeing results. Keep CLINC150 as a later robustness experiment, particularly for out-of-scope queries. Retain dataset attribution to Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020), and, if CLINC150 is used, Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction* (2019).

Neither benchmark alone establishes production suitability. Public benchmarks may have appeared in a general-purpose model's pretraining; that contamination cannot be ruled out here. Production traffic may have different class frequencies, language, ambiguity, and unsupported intents. Report benchmark conclusions with these limits.

## First baseline: what it means

**TF-IDF** means term frequency–inverse document frequency. It turns text into a sparse numeric vector: words and short phrases receive weights based on their occurrence in the request and how uncommon they are across training requests. A phrase such as “card arrival” can carry more useful information than a ubiquitous word. Vocabulary and rarity weights are learned exclusively from the sampled training set. See the [scikit-learn feature extraction guide](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting).

**Logistic regression** is a classifier despite its name. It learns a weight for each feature and class; multiclass softmax converts the resulting scores into class probabilities. L2 regularization discourages overly large weights, helping control overfitting with few examples. See the [scikit-learn linear-model guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

Together they provide a lightweight CPU baseline, interpretable word associations, and a clear test of how far lexical patterns go before investing in more complex models. They can miss paraphrases, context, and subtle intent differences. Predicted probabilities are not guaranteed to be calibrated or to identify unsupported requests; routing requires separate evidence.

## Evaluation and claim standards

The exact initial split and baseline recipe live in `CURRENT_PLAN.md`. Primary quality metric: macro-F1 across all 77 labels; secondary: accuracy, per-class precision/recall/F1, and confusion counts. Report every planned seed and mean/sample standard deviation; do not report only the best run. Seed variation and test-sampling uncertainty answer different questions.

EXP-007 evaluates specialist-only, Luna-only and the combined policy on the same frozen test population against dataset labels. Luna is a comparator, not ground truth. Future comparisons must retain this paired design. Freeze its model version, prompt, decoding, label mapping, retry policy, and invalid-output handling. Prompt examples must come from an explicitly counted development budget. Tune routing only on development data; evaluate accepted-subset quality, coverage, fallback rate, and full-system quality separately.

Before routing experiments, predeclare the acceptable quality loss and useful coverage target. For a higher-is-better metric Q, quality loss in percentage points is `100 * (Q_comparator - Q_hybrid)` when Q is on [0, 1]. A non-inferiority claim needs a prespecified margin and uncertainty analysis, not just a favorable point estimate. Use paired comparisons on the same requests. Do not assume Luna handles fallback cases perfectly.

For M benchmark requests and accepted set A:

- Coverage = `|A| / M`.
- All-Luna cost = sum of each request's measured usage priced under a frozen pricing schedule.
- Hybrid cost = specialist inference on every request + routing overhead + Luna cost for each fallback request.
- Benchmarked cost reduction = `100 * (1 - hybrid_cost / all_luna_cost)`; the denominator must be positive. Negative savings remain results.

Use request-level token usage because fallback requests may be longer or more expensive. Record pricing date, currency, cache/batch discounts, retries, and hardware/rate/utilization assumptions. Record local timing even when a defensible monetary rate is unavailable; label any monetization as an estimate.

**Actual spend** is what the experiment consumed or was billed, including baseline calls, development calls, failed calls, and retries. **Benchmarked inference cost** is the cost of a defined policy on the fixed workload; replaying cached comparator outputs estimates a counterfactual and does not create actual cash savings. **Projected deployment savings** additionally assume volume and traffic mix and must be labeled projections. Training, labeling, and development costs are reported separately from per-request inference costs and included in any later break-even analysis.

## Confidence diagnostic (2026-09-25)

EXP-005 analyzes all 15 saved primary EXP-004 validation probability files without fitting or executing a model. Maximum class probability is explicitly uncalibrated. A fixed label-blind ranking and tie rule yields reproducible risk/coverage curves and per-intent acceptance counts at approximately 25/50/75/90/100% coverage. The higher-confidence subsets are more accurate on these reused cases, but can reject entire intents; this is exploratory development evidence, not a calibrated confidence or production-threshold claim. The same 770 validation labels are reused and no new labels are added. All earlier artifacts/results and the sealed test remain unchanged.

The next proposed work is analysis of persistent rejections/confident errors in saved predictions, not another model or routing implementation. Future calibration still needs an independently evaluated, explicitly label-budgeted protocol. See the EXP-005 entry in `EXPERIMENTS.md` and `CURRENT_PLAN.md` for exact observations and boundaries.

## General-model evaluation preparation (2026-09-25)

Historical checkpoint retained below; the completed state at the top of this document supersedes its preparation-only status.

EXP-006 prepares an OpenAI `gpt-6-luna` zero-shot baseline on exactly the existing 770 validation texts, with one response set shared by all 15 specialist configurations. The user authorized only runner implementation, synthetic tests, protocol freezing and cost estimation; **the paid experiment has NOT RUN** and actual API spend is zero. The fixed 77-intent prompt sends no ground truth, specialist output/confidence or demonstrations. Official documentation exposes an alias without a dated snapshot, limiting immutable model reproducibility.

The future evaluation will measure general-model standalone and full-workload specialist/fallback metrics using EXP-005's fixed prefixes plus all fallback. Unresolved responses remain errors; fallback accuracy is measured on rejected requests, never assumed. Explicit spending approval and a numeric cap are required. Actual collection charges, hypothetical API-only routed charges and unmeasured specialist deployment cost remain separate; no total-system savings or production-latency claim follows. All prior results, splits, dependencies and the sealed test are preserved. See the EXP-006 preparation write-up and current plan for exact settings, estimates, failure controls and approval command.

## Separate EXP-006R recovery preparation (2026-09-26)

Historical checkpoint retained below; the completed state at the top of this document supersedes its preparation-only status.

The saved EXP-006 execution now has 718 successful and 52 unresolved HTTP-429 validation requests, each unresolved ID having exhausted the original two-attempt allowance. The earlier preparation entry is historical. Preserve all original evidence byte-for-byte; any new attempts belong to a separate EXP-006R protocol, cache and spending authorization. The request model/prompt/schema/settings, validation IDs and evaluation definitions remain unchanged. Original error bodies identify exhausted credits, which require a quota/billing remedy before recovery. A new runner halts on recurring quota errors and applies bounded serial backoff for transient rate limits.

**EXP-006R is prepared only; no paid recovery or real recovered evaluation has run.** The offline merge uses 718 unchanged original successes and only the 52 recovery outcomes, retains failures, and labels results EXP-006 + EXP-006R recovery. The same 770 validation labels are reused; no new labels or test access. No recovered quality, production threshold or total-system savings is asserted. Protocol hash, future command and conditional cost estimates are in the recovery preparation write-up and current plan.

## Fixed-threshold preparation (EXP-007, 2026-09-26)

The saved EXP-006 + EXP-006R validation collection is now complete with 770 successful combined responses; historical preparation statuses above describe their dates. Preserve both stages' original artifacts and unknown-usage failures. The user's next choice is a scalar routing rule for the five existing 20-shot specialists, matching their prior 90% validation rank boundary. Each threshold is derived only from saved validation confidences/IDs, and all five reproduce exactly 693/770 accepted IDs with no boundary ties. No recalibration or retraining occurs.

Freeze `confidence >= per-seed threshold` and the unchanged Luna fallback before any test access. Runtime cannot use dataset-wide ranks or force 90% coverage; future test coverage is a measured outcome. The 20-shot/90% design was chosen after exploratory validation results, so disclose repeated use of the same 770 additional development labels. This is a deployable decision rule, not a calibrated probability, production guarantee or measured test result.

The authoritative EXP-007 protocol specifies one eventual evaluation of all 3,080 official test IDs, all five saved specialists, and one all-case Luna response set reused for standalone/routed comparisons. It requires separate explicit test access and paid spending authorization. Preparation has read no test bytes and made no paid calls. Current plan/evidence record the exact thresholds, protocol hash, conditional API estimate and 495-test verification. Specialist deployment cost, production latency and total-system savings remain unmeasured.

## Scope boundaries and persistent records

Keep the completed local experiment reproducible. The offline routing evaluator exists; no frontend, production serving stack, database, orchestration platform or GPU fine-tuning is in scope. Preserve protocols and evidence, and keep any separately authorized EXP-009 execution within its frozen design.

- `AGENTS.md`: operational rules for future sessions.
- `CURRENT_PLAN.md`: current milestone, protocol, decisions, tasks, and unresolved choices.
- `EXPERIMENTS.md`: chronological experiment history, including failures and deviations.
- `RESULTS.md`: verified reproducible project measurements only.

Update these records when major decisions change. Keep prior experiment records intact; supersede with a dated explanation.
