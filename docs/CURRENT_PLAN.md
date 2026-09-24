# Current plan

Updated: 2026-09-24. Active protocol: **`banking77-val10-v2`**.

## Repository and status

Canonical checkout: `/Users/Andrew/Developer/data-efficient-inference`. The existing repository was moved here intact from `/Users/Andrew/Documents/ChatGPT/Data-Efficient Specialist Inference`; no new repository was initialized. Before protocol edits, `main` was clean at `717b7e6`, one commit ahead of `origin/main`. Origin remains `https://github.com/Intelligent-Dynamics/data-efficient-inference.git` in the Intelligent Dynamics organization.

M1a implementation and one legacy N=5/seed=11 validation smoke run are complete. **The v2 protocol fix is implemented and tested: 29 tests pass, and all 15 combinations of N=5/10/20 and seeds 11/22/33/44/55 pass real-data sampling/isolation checks.** No BANKING77 model was trained or evaluated during this protocol-fix session. The next model experiment remains pending. No final project result, test evaluation, API, routing, calibration, embeddings, SetFit, cost model, frontend, or GPU work is in scope.

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

## Validation performed for this change

`.venv/bin/python -m pytest -q`: **29 passed**, no warnings. Added coverage for 10-case holdout feasibility, supported regimes, frozen validation across sizes/seeds, idempotent preparation, test rows excluded from the training loader, legacy/mutated protocol rejection, and refusal to overwrite old manifests. Existing metric, convergence-failure, ID/text isolation, and train-only TF-IDF checks still pass.

Real-data preparation and sampling checks verified every N/seed combination, all 77 class budgets, nested/deterministic sampling, fixed validation IDs, and isolation. Evidence is in `experiments/protocol-val10-v2-check.json`. The prior manifest and smoke artifact hashes remain unchanged. No model fit or evaluation was involved in these real-data checks; unit-test fits used synthetic data only.

## Exact next experiment (pending; not run this session)

Run **N=10, seed=11 under v2**, using unchanged model settings and the v2 manifest:

```sh
cd /Users/Andrew/Developer/data-efficient-inference
uv sync --locked --cache-dir .cache/uv
.venv/bin/python -m baseline prepare
.venv/bin/python -m baseline run --shots 10 --seed 11 --output artifacts/exp001-v2-n10-s11
```

Then independently repeat into `artifacts/exp001-v2-n10-s11-reproduction`; compare split/sample hashes and predicted labels exactly and metrics within `1e-10`. Afterward complete the 5/10/20 × five-seed matrix under v2, including a **fresh v2 5-shot run**. The legacy smoke cannot substitute for that run because its validation set and candidate training pool differed. Current v2 validation is a deterministic subset of the legacy validation set, not a fresh confirmatory holdout.

Freeze preprocessing, configurations, metrics, and evaluation policy before any later official-test evaluation. Log all test access. If test results later guide design, disclose reuse and obtain a fresh holdout for new confirmatory claims. Stronger-model choice, calibration design, useful coverage/quality-loss targets, and cost assumptions remain later decisions.

## Protocol history

The original validation=20 protocol and measured infeasibility remain recorded in `EXPERIMENTS.md` and `experiments/exp001-smoke-n5-s11/protocol.md`; its manifest stays at `data/processed/banking77/manifest.json`. The user's 2026-09-24 direction changes validation to 10 and drops 50-shot based on class availability, not on performance tuning. Source data, cleaning, deterministic ordering, all 77 classes, and the official test split are unchanged.
