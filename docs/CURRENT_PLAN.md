# Current plan

Updated: 2026-09-25. Active protocol: **`banking77-val10-v2`**.

## Repository and status

Canonical checkout: `/Users/Andrew/Developer/data-efficient-inference`. The existing repository was moved here intact from `/Users/Andrew/Documents/ChatGPT/Data-Efficient Specialist Inference`; no new repository was initialized. Before protocol edits, `main` was clean at `717b7e6`, one commit ahead of `origin/main`. Origin remains `https://github.com/Intelligent-Dynamics/data-efficient-inference.git` in the Intelligent Dynamics organization.

**EXP-006 preparation complete; PAID EXPERIMENT NOT RUN.** Implemented and synthetically tested an offline dry run, an explicitly authorized/capped resumable Responses runner, and standalone/general-plus-specialist evaluation using the fixed EXP-005 acceptance sets. No API inference, new model training, calibration, test access or real fallback evaluation occurred. Actual API spend this session: **$0**. All earlier artifacts/results and dependency/model pins remain unchanged.

Candidate `gpt-6-luna` and Standard prices were checked in official documentation on 2026-09-25; no dated snapshot is published there. Prompt/schema/settings are frozen in `experiments/exp006-general-model-preparation/`. One set of 770 responses will serve all 15 specialist configurations; it does not exist yet. See the study README and verification JSON for complete implementation/test evidence.

## Decisions and rationale

- Retain all 77 BANKING77 intents and the original sealed official test split.
- Reserve a fixed **10 examples per class** from cleaned official training data for the validation/calibration development pool. Use it for validation only at this stage; no calibrator is fitted.
- Use nested training regimes **5, 10, 20 examples per class** and seeds **11, 22, 33, 44, 55**. Every regime and seed uses the same validation IDs.
- Drop 50-shot. The smallest class has only 35 usable source-training examples, so 50 unique training examples cannot be supported even with no holdout. Do not exclude classes, oversample, borrow validation/test rows, or claim repeated examples are new labels. Reconsider only with a documented source of additional valid training data.
- Keep the fixed TF-IDF + logistic-regression baseline and most-frequent-class dummy. No hyperparameter search follows from the earlier smoke score.
- Preserve the old validation=20 smoke run and protocol. New-protocol scores form a separate study; do not pool them with the old run.

## Dataset and split protocol

1. Pin publisher revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Source URLs, SHA-256 checksums, expected counts, and attribution are in `data/banking77-source.json`; retain the upstream license and ordered label mapping.
2. Keep the 3,080 official test rows intact. Before a frozen final evaluation, allow only mechanical schema/count checks, normalized-text hashes for contamination auditing, and byte-integrity checks. No test labels, predictions, metrics, qualitative inspection, feature fitting, model selection, or threshold selection.
3. Use zero-based original CSV data-row IDs, prefixed by official split. Audit text keys use Unicode NFKC, lowercase, trimming/collapsed whitespace, then SHA-256. Model inputs retain their original text. Remove training-side rows matching test keys. Within source training, retain only the lowest-ID row in same-label duplicate groups and quarantine all conflicting-label duplicate groups. Log exclusions; do not claim semantic near-duplicate protection.
4. For each class, sort SHA-256 of compact JSON `["validation", 20260924, row_id]`, breaking hash ties by row ID. Reserve the first **10** as validation. The remaining rows form the candidate training pool. Require at least **30** usable source rows per class so every planned regime, including 20-shot, is feasible.
5. For each training seed, order each remaining class pool by SHA-256 of compact JSON `["training", seed, row_id]`, breaking ties by row ID. Take the first 5, 10, or 20 without replacement. This is stable across input ordering and random-number-library versions and produces nested samples within each seed. Fail on unknown regimes or insufficient classes.
6. Check exact label counts, unique row IDs/text keys, and train/validation/test isolation. Fit TF-IDF only on the sampled training texts; unused pool/validation/test texts cannot enter vocabulary or IDF fitting. Do not refit on validation while claiming an N-shot budget.
7. Persist schema-2 manifest with protocol ID `banking77-val10-v2`, validation size, split seed, and regimes at `data/processed/banking77-val10-v2/manifest.json`. The CLI uses this path by default. Preparation is deterministic/idempotent and refuses changed manifests. The training loader rejects legacy or incompatible protocol metadata explicitly.

## Label budget and measured feasibility

The unchanged duplicate audit removes 7 train/test text overlaps and 4 repeated training rows from 10,003 source-training rows, leaving **9,992**. No conflicting-label groups were found. V2 reserves **770** validation rows and leaves **9,222** pool rows. The minimum remaining class size is **25**; all 77 classes support 20-shot. The official test checksum is unchanged. These are dataset/split checks, not model-performance results.

| Training regime | Training labels | Validation labels | Unique fitting/validation labels per run |
| --- | ---: | ---: | ---: |
| 5-shot | 385 | 770 | 1,155 |
| 10-shot | 770 | 770 | 1,540 |
| 20-shot | 1,540 | 770 | 2,310 |

All 10,003 public training labels are mechanically read for stratification and duplicate-conflict auditing; disclose that separately. Calibration/prompt/test-evaluation labels used so far: none. Record unique examples across the later multi-seed study rather than summing overlapping per-run budgets. Public labels simulate few-shot fitting, not measured annotation cost.

## Is this validation budget sufficient?

For the current stage, **yes for coarse development comparisons among a small, prespecified set of simple models**, with cautious interpretation. The common 770-case validation set makes paired comparisons possible and avoids changing evaluation examples with N. It is not enough to establish small quality differences or precise class-level estimates: per-class recall changes in 10-percentage-point increments, and macro-F1 can vary substantially. Multiple training seeds measure training-sample sensitivity, not uncertainty from drawing a different validation set. Limit tuning and preserve the sealed test for a later frozen evaluation.

This is one held-out development pool, not separate validation and calibration evidence. Later calibration/threshold fitting and evaluation must not reuse the same labels as if independent. Ten examples per class are especially weak for per-class calibration and high-confidence/rare-error estimates. Before calibration work, specify either additional independent development labels or an appropriate cross-fitting procedure, account for all labels, and keep final evaluation independent. No calibration claim or non-inferiority margin is established now.

## Unchanged baseline and reporting

- Word TF-IDF: lowercase, unigrams/bigrams, default word tokenization, min_df=1, max_df=1.0, no stop-word removal, smoothed IDF, L2 normalization, sublinear_tf=True.
- Logistic regression: multinomial loss, L2, C=1.0, lbfgs, max_iter=2000, tol=1e-4, intercept, no class weighting. Fixed settings; convergence warnings make the run incomplete.
- Dummy: most-frequent class with recorded tie behavior.
- Primary metric: macro-F1 over all 77 labels. Also accuracy, per-class precision/recall/F1/support, and confusion counts (zero division mapped to zero).
- Record predictions, class probabilities in the fixed label order, split/sample manifests, source/configuration, protocol ID, label access, source snapshot/Git provenance, environment, warnings/failures, and model artifact.
- Record single-thread fit time, model bytes, and vectorizer-plus-classifier prediction timings: batch size 1 and full validation; one warmup and five timed passes. Report hardware, units, median/throughput. These are local measurements, not cost savings.
- Retain every planned seed; later report mean/sample standard deviation by N. Promote only independently reproduced measurements to `RESULTS.md`. Keep failures and negative results.

## Validation performed for the v2 protocol fix (historical)

`.venv/bin/python -m pytest -q`: **29 passed**, no warnings. Added coverage for 10-case holdout feasibility, supported regimes, frozen validation across sizes/seeds, idempotent preparation, test rows excluded from the training loader, legacy/mutated protocol rejection, and refusal to overwrite old manifests. Existing metric, convergence-failure, ID/text isolation, and train-only TF-IDF checks still pass.

Real-data preparation and sampling checks verified every N/seed combination, all 77 class budgets, nested/deterministic sampling, fixed validation IDs, and isolation. Evidence is in `experiments/protocol-val10-v2-check.json`. The prior manifest and smoke artifact hashes remain unchanged. No model fit or evaluation was involved in these real-data checks; unit-test fits used synthetic data only.

## Latest learning-curve findings

| Shots/class | Train/run | Accuracy mean ± SD (%) | Macro-F1 mean ± SD (%) | Macro-F1 min–max (%) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 385 | 51.84 ± 1.69 | 50.41 ± 1.58 | 48.77–52.59 |
| 10 | 770 | 62.49 ± 1.60 | 61.58 ± 1.81 | 60.05–64.14 |
| 20 | 1,540 | 69.95 ± 0.61 | 69.19 ± 0.63 | 68.18–69.80 |

Both increments improved accuracy and macro-F1 for every paired seed. Mean macro-F1 gains were **+11.17 percentage points** from 5→10 (seed gains 9.81–11.71) and **+7.61 points** from 10→20 (4.04–9.50). Accuracy gains were +10.65 and +7.45 points. These are substantial, consistent development-set gains; no formal population significance claim is made.

Macro-F1 seed SD is **1.58 → 1.81 → 0.63 percentage points**: variance does not decrease monotonically, but is much lower at 20 shots. Accuracy SD is **1.69 → 1.60 → 0.61 points**. Seed variability is not validation-sampling uncertainty.

The slope is flattening, but the curve does **not yet show a clear plateau**: 10→20 still adds 7.61 macro-F1 points on average. Diminishing returns are especially visible per added example/class (approximately 2.23 F1 points for 5→10 versus 0.76 for 10→20). Three budgets cannot establish an asymptote.

The study preserved the same 770 validation IDs and exact per-class budgets for every run. Thirty complete artifact sets exist locally, with compact evidence for all primary/refit runs in Git. Verified aggregates and error analysis are reproducible from the versioned JSON without raw data. The full verifier additionally checks all hashes, intended sample membership, TF-IDF vocabulary/IDF, saved predictions, and classifier refits. All primary/refit metrics matched exactly; all coefficient/intercept differences were zero. Full suite: 35 passed.

At 20 shots, `transfer_fee_charged` (30% mean recall), `unable_to_verify_identity` (32%), and `supported_cards_and_currencies` (32%) remain weak. Closely related virtual-card and identity intents frequently confuse the model. Full class metrics and error counts are retained; do not tune on selected validation examples without logging the resulting exploratory status.

## Completed milestone — EXP-003 frozen MiniLM smoke (2026-09-25)

The user's new direction supersedes the proposed 25-shot lexical extension. Do not extend the split protocol. Implement frozen `sentence-transformers/all-MiniLM-L6-v2` plus the existing logistic regression, then run **only 5-shot, seed 11**. Preserve EXP-001/EXP-002 and `RESULTS.md`. This is a pipeline smoke check, not final research evidence.

- Immutable model revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, verified from the publisher on 2026-09-25. Apache-2.0 model; retain model-card provenance and downloaded file hashes.
- Reuse `experiments/exp002-learning-curve/runs/exp002-v2-n5-s11.json` sample IDs verbatim: 385 training and 770 validation. Verify against the existing v2 manifest; do not call preparation or sample another split. The comparator is this matching v2 run, never the legacy EXP-001 smoke.
- Encoding: frozen parameters, evaluation mode, inference mode, CPU float32, one compute thread, deterministic PyTorch operations, batch size 32, mean pooling, L2 normalization, maximum sequence length 256, 384 dimensions, no prompts, no remote code. Verify an unchanged encoder state hash before/after the run.
- Classifier: reuse `baseline.experiment.LOGISTIC` unchanged (L2, C=1, lbfgs, max_iter=2000, tol=1e-4, intercept, no class weighting). No tuning, scaler fitting, or fine-tuning. The training seed identifies the existing sample; lbfgs does not use it as a stochastic optimizer seed.
- Pin sentence-transformers 5.1.1, PyTorch 2.8.0, transformers 4.56.2; retain the existing scientific-library pins and lock all transitive dependencies. Record the complete installed package list and hardware.
- Save source/lockfile provenance, sample IDs, predictions/probabilities, metrics, model-file hashes, encoder freeze checks, classifier and row-indexed embeddings. Downloads live in ignored `.cache/huggingface/hub`; feature arrays and classifier remain in ignored `artifacts/`. Commit compact JSON evidence only.
- Measure snapshot download/cache resolution and model loading separately from train/validation encoding and classifier fitting/prediction. Single cold passes are resource observations, not warmed latency benchmarks or cost estimates.
- Label budget: 385 fitting + **770 additional validation labels** = 1,155 unique task examples, already used in the matching TF-IDF run. No new unique task labels; all 10,003 source-training labels are still read for audit/stratification. The encoder uses external pretraining, so this is not training a language model from only 385 labels. Calibration/prompt/test labels: zero.
- Validate row/text/label alignment, encoder freeze, exact reference-ID reuse, isolation, artifact consistency and unchanged LR. Tests use synthetic encoders/data, not extra BANKING77 experiments.

Executed exactly once, after tests passed:

```sh
HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m baseline.embeddings --output artifacts/exp003-minilm-v2-n5-s11
```

Outcome: 75.84% accuracy and 74.46% macro-F1 versus 52.21% and 51.13% for the matching TF-IDF run. This is a smoke comparison only. Same sample/validation hashes; unchanged encoder state, zero trainable encoder parameters; LR settings unchanged; no warnings. Source and artifact hashes, feature alignment, saved classifier replay and metric recomputation passed. Full suite: 47 tests. No implementation blocker was discovered. Single-seed uncertainty, reused validation labels and potential upstream pretraining overlap remain unresolved limitations.

Implemented `baseline/embeddings.py` with a fixed 5-shot seed-11 command and `--verify` replay mode. Downloaded weights and row-indexed embeddings are ignored. Tests and compact JSON/protocol evidence are versioned; the TF-IDF pipeline, split protocol, existing experiments and `RESULTS.md` remain unchanged.

## Completed milestone — EXP-004 full frozen-MiniLM learning curve

The user's 2026-09-25 request supersedes the single-reproduction next step. Run all **5/10/20 shots × seeds 11/22/33/44/55**, then independently reproduce every score. No hyperparameter tuning or official-test evaluation. Keep all EXP-001/002/003 artifacts unchanged.

- Read each ordered train/validation ID list from its matching versioned EXP-002 record. Verify its hash against the existing v2 protocol-check report, the dataset/source/manifest, per-class counts, row labels and ID/text isolation. Never call preparation or sample a new split.
- Reuse EXP-003's exact MiniLM revision, package pins, encoding settings, and unchanged LR settings above. Frozen encoder, CPU float32, one compute thread. No new dependencies.
- Primary feature cache: validate and import EXP-003's 385 training and 770 validation embeddings, then encode only the remaining unique training rows needed across the matrix. Cache provenance includes model files, revision/settings/runtime, IDs/text hashes/labels, array hash and unchanged encoder state. No unused training-pool or test texts are encoded.
- Reproduction: in a separate stage, reload the pinned encoder, independently re-encode all required rows into a fresh cache using the same three ordered groups, and refit 15 new classifiers. Compare embeddings at predeclared absolute tolerance 1e-6 (relative tolerance 0); require identical predicted labels and complete quality metrics; report maximum embedding and classifier-parameter differences. Independent caches/refits are local reproduction, not external replication. Reproduction runs are excluded from averages.
- Every classifier run uses only its exact recorded N-shot embeddings/labels; the shared immutable cache is a storage optimization for the frozen representation, not fitting on the union. Persist separate primary/refit metadata, samples, predictions, probabilities, classifier, source snapshot, runtime and warnings/failures.
- Record encoding time once per cache group and classifier fit/predict time separately per run. Cache hits have no new encoding measurement (`null`), and end-to-end inference time is unmeasured. Never report classifier-only cached timings as end-to-end speed. The EXP-003 metadata's wording about cold passes does not establish a cold-model benchmark: validation encoding followed training encoding.
- Aggregate five primary seeds for each N: mean accuracy/macro-F1, sample SD (`ddof=1`), min/max and all seed values. Compute macro-F1 improvement in percentage points against each matching TF-IDF seed. Produce a comparison plot with seed points and mean ± sample SD; no cherry-picking.
- Per-run labels: 385/770/1,540 training **plus 770 validation**. Reuse adds no new unique task labels beyond EXP-002; record the union rather than summing overlapping runs. Disclose all 10,003 source-training labels mechanically read for auditing/stratification and the encoder's external pretraining. Test/calibration/prompt labels remain zero.
- Test cache invalidation/alignment, exact sample reuse, train-only LR fitting, frozen checks, failure recording, full-grid aggregation and sample SD, paired gains and rejection of corrupted records. Verify all artifacts and recompute aggregates from compact records before promoting findings to `RESULTS.md`.

Executed study command after tests:

```sh
HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m baseline.minilm_curve --output artifacts/exp004-minilm-learning-curve --reuse-smoke artifacts/exp003-minilm-v2-n5-s11
```

Commit/push the completed milestone after verification and documentation. No fine-tuning, routing, paid APIs, frontend, or cost claims.

## Current findings and verification

| Shots/class | MiniLM accuracy (%) | MiniLM macro-F1 (%) | TF-IDF accuracy (%) | TF-IDF macro-F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 74.13 ± 1.30 | 72.60 ± 1.34 | 51.84 ± 1.69 | 50.41 ± 1.58 |
| 10 | 80.05 ± 0.93 | 79.19 ± 1.03 | 62.49 ± 1.60 | 61.58 ± 1.81 |
| 20 | 83.77 ± 0.66 | 83.35 ± 0.68 | 69.95 ± 0.61 | 69.19 ± 0.63 |

Mean paired macro-F1 advantages over TF-IDF: +22.19 / +17.60 / +14.16 points at 5/10/20 shots; all 15 individual paired improvements were positive. MiniLM's F1 seed SD decreases from 1.34 to 1.03 to 0.68 points. Mean 5→10 and 10→20 F1 gains are 6.58 and 4.16 points; the curve is flattening but no plateau is established. All exact seed values, ranges, paired gains and sample SDs are recorded, not only the best seed.

Implemented `baseline/minilm_curve.py`: reads existing reference IDs, verifies/imports compatible EXP-003 features, encodes only the required union, independently reconstructs a second cache, fits/verifies 30 separate classifiers, and aggregates/plots only 15 primary runs. No earlier pipeline or dependency change was needed.

The primary cache reused 1,155 EXP-003 vectors and encoded 5,009 missing training rows; the independent cache recomputed all 6,164 rows. Both caches retained the exact frozen encoder state. Maximum embedding and classifier-parameter differences were 0.0; all scores/predictions matched, including EXP-003's seed-11 smoke. Cache hits and per-run cached classifier timings are explicitly distinct from encoding and unmeasured end-to-end inference. Both caches and all run sources/artifacts were audited; summary/CSV/plot regenerated byte-for-byte. Test suite: **64 passed**. No failures, warnings or blockers were discovered.

Label budget: 385/770/1,540 training per run plus **770 validation labels**. Across the full matrix: 5,394 unique training + 770 validation = 6,164, all already present in EXP-002. Additional audit/stratification access to all 10,003 training-source labels and external encoder pretraining remain disclosed. Seed SD does not measure holdout/traffic uncertainty; reused validation and possible upstream benchmark exposure limit claims. The official test was checked only by byte hash.

## Completed milestone — EXP-005 confidence/selective-prediction diagnostic

The user's request supersedes the proposed paired 20-shot error analysis. Analyze all 15 existing EXP-004 primary runs (5/10/20 shots × seeds 11/22/33/44/55). No model or calibrator fitting, new embedding computation, dataset change or official-test access.

- Audit archived probabilities against EXP-004's versioned record/artifact hashes and original sample IDs, class order, predictions, labels and full metrics before extracting scores. Confirm cache/source provenance. Regenerate probabilities only from the saved classifier and matching cache if necessary, with explicit logging; never refit. No regeneration is expected while intact saved probabilities are available.
- Confidence is the maximum predicted class probability, explicitly **UNCALIBRATED**. Rank descending without consulting correctness. Exact ties use ascending SHA-256 of the UTF-8 row ID, then row ID; do not round probabilities or use class labels to break ties.
- Evaluate every prefix from 0 through 770 accepted requests. Landmarks use `ceil(target * 770)` at 25/50/75/90/100%: **193/385/578/693/770** accepted. Actual coverages are approximately 25.065/50/75.065/90/100%. The label-blind tie rule may split tied scores; this is a ranking diagnostic, not a fixed probability-threshold policy.
- Record accepted, correct and error counts, actual coverage, accepted accuracy and selective risk. At zero accepted, accuracy/risk are undefined (`null`), never zero. At full coverage, accuracy must exactly match the corresponding EXP-004 run.
- Record per-true-intent acceptance/error counts at every landmark, including zero-acceptance intents. Labels are used only after ranking, to audit outcomes and class composition. Aggregate each training budget over five seeds using mean/sample SD (`ddof=1`), retain every individual run, and show all prefix curves with seed variability.
- Commit compact records of ordered IDs, uncalibrated scores, truth/predictions, source hashes and landmarks; these reconstruct the full curves exactly. Keep full class-probability arrays, classifiers and embedding caches in their existing ignored locations. No raw/private text is needed in the diagnostic evidence.
- This reuses the same 770 validation labels (10 per class), adds no labels, and does not make five independent validation datasets. Preserve all prior artifacts and results. No calibration guarantee, production threshold, fallback-answer assumption, combined-system accuracy, inference-speed or dollar-savings claim.

Executed command after tests:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m baseline.selective --artifacts artifacts/exp004-minilm-learning-curve --output artifacts/exp005-selective-diagnostic
```

All saved probabilities were intact; none were regenerated. Full-coverage scores match EXP-004 exactly for all 15 models. A separate checker agrees with every confidence rank, prefix error count and per-class landmark count; aggregate differences are at most 1.78e-15. All records, summaries, CSVs, full aggregate curves and the plot regenerate byte-for-byte. The 99-test suite covers the new arithmetic/alignment invariants and existing pipelines. No diagnostic failures or implementation blockers were found.

At 50% coverage (385/770), accepted accuracy is **87.90 ± 0.91%, 93.25 ± 1.31%, 96.68 ± 0.34%** at 5/10/20 shots, versus full-coverage **74.13 ± 1.30%, 80.05 ± 0.93%, 83.77 ± 0.66%**. For 20 shots, 25.065/50/75.065/90/100% coverage incurs **1.4/12.8/42.8/82.8/125.0 mean errors** out of 193/385/578/693/770 accepted. All landmark means/sample SDs and each seed's integer counts are retained in the experiment log/evidence.

Confidence identifies a more accurate subset on these validation cases, with material intent omissions: at 50%, 20-shot models reject every example of 5–8 intents, including `cash_withdrawal_not_recognised` and `topping_up_by_card` in every seed. At 75.065%, every intent is represented for 20 shots. Risks at individual prefixes need not be monotonic. No generalization, calibration, independent-holdout, threshold, LLM fallback or cost claim follows.

## Superseded proposal — persistent rejection and confident errors

Not started in EXP-005; superseded by the user's EXP-006 preparation request. Reuse the saved 15 MiniLM diagnostics and matching TF-IDF predictions to characterize all 77 intents and request-level recurrence of rejection/errors across seeds at the existing 25/50/75/90/100% prefixes. Preserve the same cases and label budget; do not adjust prefixes based on correctness, fit another model, introduce new labels, or access the official test. Report whether omissions persist across budgets/seeds and whether TF-IDF corrects the same errors; overlapping seeds remain repeated observations of the same requests.

Before any later calibration or threshold-selection work, specify independent evaluation or suitable cross-fitting and all additional label access. No calibration protocol is selected yet. Stop after committing and pushing EXP-005; no fine-tuning, routing, paid APIs or frontend.

## EXP-006 prepared protocol and exact next milestone

The user authorized implementation, synthetic testing and cost estimation **only**. The next milestone is the paid 770-request validation collection/evaluation, pending explicit user approval of spending. Do not execute it automatically after preparation.

- Frozen candidate: OpenAI `gpt-6-luna` alias, no documented dated snapshot; record every returned model ID and halt on mismatch. Responses API, reasoning `none`, output cap 128, Standard/default service tier, no tools or conversation history, strict `intent` enum, store/stream false. No silent model substitution or prompt tuning. Provider sampling defaults are disclosed; responses need not be deterministic.
- Fixed 77-label prompt/schema/protocol hash: `d7f824ee761ca90d4dc3a848dd19e3928475c799d6713e03e336dc2a80c9ca68`. Each request sends one original validation text without truth, row ID, specialist output/confidence or demonstrations.
- One prediction per existing validation ID, reused across 15 specialists and six accepted prefixes (0,193,385,578,693,770). Preserve the same 770 development labels; zero new labels. No new split or official-test file access.
- Plan 770 initial requests and at most one identical retry for each transient transport/HTTP failure: 1,540 attempts maximum. Refusals, incomplete/invalid answers and valid wrong answers are terminal. Save raw responses/usage/timing/model IDs locally; persist reservations before calls and validate the ledger on resume/evaluation. Fatal drift/errors halt with all unresolved cases retained.
- Standalone and combined accuracy/macro-F1 use all 770 requests. Count unresolved outputs incorrect; report rejected-subset fallback accuracy separately, with `null` when no requests fall back. Use all seeds and sample SD; never assume fallback improves correctness.
- Dry-run estimate: **$0.124057 nominal**, **$1.7313625 conservative for one attempt each**, **$3.462725 including all allowed retries**. The proposed **$4.00 cap is NOT approved**. Byte-based input estimates are heuristics, with a large token envelope; reserve full output and all-input cache-write pricing, no cache-read discounts. Limits are conditional on the documented price/token envelope, not guaranteed invoices. Halt on observed breaches.
- Actual collection charges (including failures/unknown usage) remain separate from hypothetical API-only routed charges. Replayed cache behavior is an explicit assumption. Specialist deployment cost, total-system savings and production latency remain unmeasured/null.
- Required live environment: `OPENAI_API_KEY`, optional `OPENAI_PROJECT_ID`; configure locally, never paste or commit secrets. No credentials are needed/read by dry-run or evaluation. The live command requires `--authorize-live`, numeric `--max-spend-usd`, exact `--approve-protocol-sha256`, and `--acknowledge-pricing-date 2026-09-25`. Pricing verification expires after seven days; reverify before later approval. The reviewed command is in the EXP-006 README.

Implemented `baseline/general_protocol.py` (frozen input/protocol/cost preparation), `baseline/general.py` (guarded execution/cache/accounting), and `baseline/general_metrics.py` (pure full-workload comparison). No new package dependencies. Synthetic review found and fixed malformed-response, cache-loss, duplicate-cache-key, drift-pricing, and read-only ledger-accounting issues; regressions are tested. Live model/account compatibility and actual token usage are still unverified. The preparation dry run succeeded; it did not emit model predictions or evaluation scores. **243 tests passed**, including 144 new synthetic tests. The final dry run passed with network and credential access blocked; request counts, frozen configuration and independently recomputed costs match.

Before authorizing paid execution: review the frozen prompt/settings, current official model availability/prices, alias limitation, conservative cost assumptions and numeric cap. After approval, collect the single response set, preserve unresolved events and actual usage, independently reproduce every standalone/combined score from saved responses, then report positive and negative tradeoffs. Do not use resulting validation performance to select a production threshold or rewrite this prompt. The current session stops after committing/pushing preparation.

## Protocol history

The original validation=20 protocol and measured infeasibility remain recorded in `EXPERIMENTS.md` and `experiments/exp001-smoke-n5-s11/protocol.md`; its manifest stays at `data/processed/banking77/manifest.json`. The user's 2026-09-24 direction changes validation to 10 and drops 50-shot based on class availability, not on performance tuning. Source data, cleaning, deterministic ordering, all 77 classes, and the official test split are unchanged.
