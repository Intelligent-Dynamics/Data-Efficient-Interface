# Current plan

Updated: 2026-09-24.

## Status and decisions

M0 — repository inspection and research design: complete. M1 implementation is in progress; this session covers tests and **one N=5, seed=11 validation smoke experiment**, not the full matrix or a final result. No paid model calls are in scope.

Repository setup decision (2026-09-24): use the private GitHub repository `Intelligent-Dynamics/data-efficient-inference`, with `main` as the default branch and `origin` as the remote. Organization identity and repository-creation permission were verified through the authenticated GitHub CLI. This setup does not advance the ML implementation milestone.

Decisions made on 2026-09-24:

- Recommend BANKING77, all 77 intents, for its narrow banking-support workload. Dataset comparison and sources are in `PROJECT_CONTEXT.md`.
- Start with a most-frequent-class sanity baseline and TF-IDF + L2 logistic regression.
- Originally planned training budgets: N = 5, 10, 20, 50 per class, with seeds 11, 22, 33, 44, 55. The data audit below establishes that only N=5 and N=10 are feasible under the current split; larger regimes must fail explicitly.
- Use a separate, fixed validation budget of 20 labels per class. Disclose it in every few-shot claim.
- Defer stronger-model selection, paid calls, calibration, routing, UI, and infrastructure.

## Exact next implementation milestone

**M1a — implement the reproducible local BANKING77 few-shot lexical baseline, run tests and one N=5/seed=11 validation smoke experiment. Keep the official test split sealed.** The user's implementation-session instruction narrows the original M1 full-matrix acceptance criteria; multi-seed experiments and independent reproduction are deferred to M1b.

Build only a small Python CLI experiment with pinned dependencies, data preparation, split manifests, a scikit-learn pipeline, evaluation, and artifact output. No service or frontend.

### Data and split protocol

1. Download the publisher's BANKING77 files from a pinned immutable Git revision. Save source URLs, revision, SHA-256 checksums, license/attribution, and ordered label mapping. Validate actual schema, 77 labels, and published counts; log discrepancies rather than assuming counts are correct.
2. Preserve the official test split. Permit only mechanical schema/count and contamination checks before final evaluation; do not display test texts, inspect test errors, or use test-derived vocabulary/statistics in model development.
3. Audit duplicate groups using Unicode NFKC normalization, lowercase, trimming, and collapsed whitespace, for audit keys only. Preserve original model input texts. Remove training-side rows matching a test key, without inspecting test labels; keep the test set intact. Within official training data, collapse same-label duplicate keys to the lowest original row ID and quarantine all rows for keys with conflicting labels. Record all exclusions and hashes. Do not use semantic inspection of test data to design cleaning rules. This audit does not guarantee absence of near duplicates; record that limitation.
4. From the cleaned official training data, reserve exactly 20 examples per class for validation with split seed 20260924. Use stable original row IDs and a recorded deterministic sampling algorithm. Save the manifest once; reuse across models and sampling seeds.
5. For each training seed, deterministically order each class's remaining pool and take nested prefixes for feasible budgets. The implementation sorts SHA-256 of compact JSON `[namespace, seed, row_id]`, with row ID breaking any hash ties. Namespaces are `validation` and `training`; source row IDs use zero-based CSV data-row positions. This makes ordering independent of input order and random-number-library versions. Sampling is without replacement. Require at least 25 usable examples per class to reserve 20 and allow N=5; check each requested N separately and reject insufficient classes. This replaces the original all-classes-at-least-70 prerequisite after the documented audit failure below. Never oversample, silently reduce N, or drop a class.
6. For feasible N=5/10 runs, training uses 385/770 labels and validation adds 1,540, totaling 1,925/2,310 model-development labels per run. N=20/50 counts are hypothetical, not runnable under this protocol. All 10,003 public training labels are mechanically read for stratification and duplicate-conflict auditing; record that access separately. This simulates few-shot fitting and does not establish total annotation cost. Record unique model-development labels across the later multi-seed study as well. Test labels are not used during this milestone. No unused training-pool texts may enter vocabulary fitting or other representation learning.
7. Do not refit on validation data when claiming the stated N-shot training budget. Any later extra calibration or prompt-example labels must be counted separately. Public labels simulate limited-data access, not actual annotation work.

### Baseline recipe

- Sanity model: `DummyClassifier(strategy="most_frequent")`; record deterministic tie behavior, and measure rather than invent its scores.
- One pipeline: word TF-IDF with lowercase enabled, word unigrams and bigrams, default word tokenization, `min_df=1`, `max_df=1.0`, no stop-word removal, smoothed IDF, L2 normalization, and `sublinear_tf=True`.
- Multiclass logistic regression using multinomial loss, L2 regularization, `C=1.0`, `solver="lbfgs"`, `max_iter=2000`, `tol=1e-4`, intercept enabled, and no class weighting. Resolve version-specific parameter syntax when pinning dependencies.
- Keep these settings fixed for M1. No hyperparameter search, synthetic data, pretrained embeddings, or calibration. Treat convergence warnings as a failed/incomplete run requiring a documented correction, not a valid result.
- Fit vectorizer and classifier only on each sampled training set. Predict all validation requests. Save predicted labels and class probabilities in the fixed label order, along with true labels and row IDs.
- Measure macro-F1 over all 77 labels, accuracy, per-class precision/recall/F1 (zero division mapped to zero), and confusion counts. Report all seeds and aggregate mean/sample standard deviation by N. Do not select a winning seed.
- Measure fit time, vectorizer-plus-classifier prediction time, and serialized model size. Record hardware, software, thread count, batch size, warmup, and repetition count. Use batch size 1 and full-validation batch, one warmup plus five timed repetitions; report median duration and throughput with units. These are local measurements, not monetary savings or production latency guarantees.

### Artifacts and acceptance criteria

1. Provide documented CLI commands for preparation and the full run matrix, with no credentials required. Pin runtime/package versions; record code revision and dirty-tree patch/hash if uncommitted.
2. Save configuration, source checksums, exclusions, label map, split/sample row IDs, predictions, metrics, timings, warnings, and model artifacts under unique run IDs. Keep downloaded data and bulky generated files out of Git; retain enough manifests and instructions to regenerate them.
3. Add meaningful checks for disjoint splits/duplicate keys, exact class budgets and nesting, deterministic sampling, and train-only TF-IDF fitting using a token exclusive to validation. Verify metric recomputation from saved predictions. No tests of incidental implementation details.
4. In M1a run only N=5, seed=11. In M1b run the feasible N=5/10 matrix over all five seeds and retain failures; do not run N=20/50 without a new documented protocol. Independently rerun the N=10, seed=11 reference from a clean output directory; require identical split hashes/predicted labels and metric agreement within 1e-10. Timing need not be identical.
5. Log runs in `EXPERIMENTS.md`. Promote only artifact-backed, reproduction-checked measurements to `RESULTS.md`, with a working command and provenance. Unverified runs remain in the log. Mark results as validation results.
6. Update this plan with findings and the next bounded milestone. Do not add routing merely because baseline probabilities are available.

### Test evaluation policy after M1

Freeze preprocessing, candidate configuration, metrics, and the evaluation plan before opening the official test split. A later evaluation milestone can execute the full prespecified N/seed matrix once and report all results. Do not optimize settings or choose seeds based on test results. Log every test access. Once test results inform further design, disclose reuse and obtain a fresh holdout for new confirmatory claims. Validation error analysis is allowed during M1; test error analysis belongs after a frozen evaluation and cannot feed back into that same confirmatory claim.

## Open questions for later milestones

- How should infeasible 20/50-shot regimes be revised? Reducing validation alone still cannot support 50 unique training examples in a class with only 35 source examples. Any class subset, additional source data, or alternative regime needs an explicitly documented protocol; do not oversample or reuse test data.

- What minimum useful specialist coverage and maximum quality loss define success? Set the non-inferiority margin before evaluating a hybrid on test; no Z value is asserted yet.
- Which stronger model, prompt, model snapshot, and experiment spending budget should be used? Verify then-current pricing when selected.
- Will a frozen lexical model support reliable abstention, or are embeddings/calibration needed? Decide from development evidence and account for extra labels.
- What local compute cost basis and traffic volume make a monetary comparison defensible?
- When should CLINC150 out-of-scope testing or a fresh business-domain holdout be added to address benchmark limitations?

None of these questions blocks the local M1 baseline. Routing and savings claims remain out of scope until the necessary comparisons exist.

## 2026-09-24 data audit and protocol revision

Pinned publisher revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`; exact source-file checksums are in `data/banking77-source.json`. Mechanical checks found 10,003 training rows and 3,080 test rows. Cleaning removed 7 training rows matching normalized test text and 4 repeated training rows; no conflicting-label duplicate groups were found. This leaves 9,992 rows: 1,540 validation and 8,452 candidate training rows. Near duplicates have not been audited.

The original preparation attempt correctly stopped at the minimum-size prerequisite. `contactless_not_working` has 35 usable source rows (15 after validation); `virtual_card_not_working` has 41 (21 remaining), `card_acceptance` 57 (37), and `card_swallowed` 61 (41). Thus N=20 fails for one class and N=50 for four. Preserve all 77 classes, validation=20, and the official test set. Preparation now reports feasible/infeasible regimes, and sampling raises an error for any infeasible request. No models were trained before this revision.
