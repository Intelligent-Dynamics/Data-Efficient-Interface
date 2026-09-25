# EXP-004 — frozen MiniLM validation learning curve

Completed 2026-09-25T18:09:29.047274+00:00: **15 primary runs + 15 independently re-encoded/refitted reproductions**, compared with the existing EXP-002 TF-IDF curve. No failed or discarded experiment. Prior EXP-001/002/003 artifacts are preserved.

## Paired results

Values are mean ± sample SD over five primary seeds.

| Shots/class | MiniLM accuracy (%) | MiniLM macro-F1 (%) | TF-IDF accuracy (%) | TF-IDF macro-F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| 5 | 74.13 ± 1.30 | 72.60 ± 1.34 | 51.84 ± 1.69 | 50.41 ± 1.58 |
| 10 | 80.05 ± 0.93 | 79.19 ± 1.03 | 62.49 ± 1.60 | 61.58 ± 1.81 |
| 20 | 83.77 ± 0.66 | 83.35 ± 0.68 | 69.95 ± 0.61 | 69.19 ± 0.63 |

Macro-F1 improvement over TF-IDF, in percentage points, using the same budget and seed:

| Shots/class | Seed 11 | Seed 22 | Seed 33 | Seed 44 | Seed 55 | Mean gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | +23.33 | +22.74 | +20.39 | +24.01 | +20.49 | +22.19 |
| 10 | +16.87 | +17.64 | +17.62 | +20.01 | +15.87 | +17.60 |
| 20 | +13.86 | +13.59 | +13.36 | +13.86 | +16.11 | +14.16 |

MiniLM improves macro-F1 for **all 15 matched training-budget/seed pairs**. Mean advantages over TF-IDF are +22.19, +17.60, and +14.16 percentage points at 5/10/20 shots. Absolute differences for every seed are retained above; no best seed was selected.

The MiniLM curve gains **6.58 macro-F1 points from 5→10** and **4.16 points from 10→20**. Returns diminish, but these three budgets do not establish a plateau. MiniLM macro-F1 seed SD decreases **1.34 → 1.03 → 0.68 points**, and accuracy SD decreases **1.30 → 0.93 → 0.66 points**. MiniLM's seed SD is lower than TF-IDF's at 5/10 shots but slightly higher at 20 shots. Macro-F1 ranges across all five seeds: 5-shot 70.94–74.46%; 10-shot 77.99–80.06%; 20-shot 82.65–84.29%.

![Frozen MiniLM and TF-IDF learning curves](learning_curve.png)

## Frozen protocol

BANKING77 publisher revision `57ec275d8078af65b7731c2a98be812d844a6d6b`; `banking77-val10-v2`, split seed 20260924, all 77 classes. Each ordered training/validation ID list was read from its matching EXP-002 artifact and checked against the preexisting v2 protocol report and manifest. No preparation, resampling, holdout changes or hyperparameter tuning occurred.

Encoder: `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with the exact EXP-003 model-file hashes. CPU float32, max length 256, batch 32, mean pooling, L2 normalization, 384 dimensions, no prompts, eager attention, eval/inference mode, deterministic operations, seed 11 for encoding, all weights frozen. Encoder state remained identical to EXP-003 before/after both cache stages. Runtime pins are unchanged: sentence-transformers 5.1.1, torch 2.8.0, transformers 4.56.2, scikit-learn 1.7.2, NumPy 2.3.3, SciPy 1.16.2; all installed versions are recorded.

Only logistic regression was trained: unchanged L2, C=1, lbfgs, max_iter=2000, tol=1e-4, intercept, no class weighting, multinomial loss. Each fit receives only its exact N-shot rows/labels; encoding the shared union with a frozen model does not fit on the union. No encoder updates, learned scaler or calibration.

Source base commit: `8dcffc549ce170c875ec15adaf3b78b886e06836` plus the hashed implementation snapshot in each artifact. The new runner is `baseline/minilm_curve.py`; the EXP-003 smoke module, TF-IDF code, package lock and existing split code are unchanged. `protocol.md` preserves the pre-execution plan. Full source snapshots and the dirty patch remain locally with the run/cache artifacts; the milestone commit contains the run source unchanged, with post-run documentation updates.

## Verification

All 15 primary runs and all 15 independent local refits completed without warnings or failures. The reproduction stage reloaded the pinned frozen encoder and recomputed all 6,164 required vectors into a separate cache, then fitted new logistic-regression instances for every configuration. Embedding maximum absolute difference was **0.0** (predeclared tolerance 1e-6, relative tolerance zero). Every pair had identical sampled IDs, predicted labels, full metrics, and classifier coefficients/intercepts; maximum parameter difference was **0.0** for all 15. The 5-shot seed-11 run also reproduced EXP-003's predictions and metrics exactly. Only primary runs enter aggregates.

This is independent local re-encoding and refitting using the same implementation and machine, not independent external replication or uncertainty over new traffic. Saved classifier replay is an additional integrity check, not counted as another independent fit. Artifact/cache/source hashes, reference sample hashes, exact per-class training counts, fixed validation IDs/labels, normalized-text/ID isolation, feature alignment, unchanged encoder state, and classifier probability/prediction replay all passed. A separate NumPy calculation agreed with all reported means and sample SDs to maximum absolute difference 1.11e-16. Summary JSON, per-seed CSV and plot were regenerated byte-for-byte from compact primary records. **64 tests passed**; the figure was visually inspected.

Evidence: `verification.json` has every primary/refit comparison; `analysis_audit.json` records persisted artifact checks, compact-record hashes and independent summary checks. `runs/` and `reproductions/` contain separate compact records with configurations, timings, samples, predictions, per-class precision/recall/F1, confusion matrices and exact metrics. `primary-cache.json` and `reproduction-cache.json` retain model hashes, runtime, group order/IDs, row counts, feature hashes, freeze checks and encoding timings. `study.json` is the complete attempt ledger. `summary.json` and `per_seed.csv` retain all seed values and gains.

## Resource observations

The primary cache reused EXP-003's 1,155 validated vectors and encoded 5,009 additional training texts in **7.492437s**. The independent cache encoded 385 / 770 / 5,009 rows in **0.618963s / 1.141228s / 7.422240s**. Encoder model loading/cache resolution is recorded separately. These are single component wall-time observations on Apple M1 Pro, 32 GiB RAM, macOS 26.5.1, Python 3.13.0, CPU float32, one compute thread.

Each run records classifier fitting and prediction time on cached embeddings. **These are classifier-only measurements, not end-to-end inference speed.** Cache hits record no new encoding time (`null`); end-to-end inference time is unmeasured (`null`). Encoding, cache I/O, hashing, serialization and probability generation are excluded from classifier timers. No warmed repeated latency comparison or monetary model was run. Do not amortize these cache observations into a production savings claim.

## Label budget and scope

Every run uses **770 additional validation labels** (10 per class), besides 385/770/1,540 training labels for 5/10/20 shots. Per-run totals are 1,155/1,540/2,310. Across the study, 5,394 unique training rows plus 770 validation rows give 6,164 unique task labels. These are the same examples used by EXP-002; the new representation and refits added no new unique task labels. All 10,003 official-training labels were mechanically read for audit/stratification and are separately disclosed. The encoder benefits from substantial external pretraining; this is simulated few-shot task adaptation, not total training-data or annotation-cost accounting.

The official BANKING77 test set stayed sealed. Test access was limited to byte-integrity checks; no test rows were parsed, labeled, inspected, encoded, tuned on, or evaluated. Existing stored test IDs/text hashes were used for isolation checks. The dataset and split manifest are unchanged.

SD is **sample SD across five training seeds (`ddof=1`)**, not a confidence interval or uncertainty over new validation samples. All runs share the same previously used 770-case holdout, with only ten examples per class; seeds share overlapping training data. These are paired development comparisons, not confirmatory unseen-test or production results. Public-benchmark exposure in upstream pretraining and semantic near-duplicate leakage cannot be ruled out. No calibration, routing, production-quality, or cost-savings claim is established.

## Commands

From `/Users/Andrew/Developer/data-efficient-inference`, with the existing raw dataset and v2 manifest:

```sh
uv sync --locked --cache-dir .cache/uv
.venv/bin/python -m pytest -q
# Exact original study command; existing output directories are refused:
HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m baseline.minilm_curve --output artifacts/exp004-minilm-learning-curve --reuse-smoke artifacts/exp003-minilm-v2-n5-s11
# Recompute summary/CSV/plot from the versioned primary evidence; no model fits or encoding:
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m baseline.minilm_curve --records-dir experiments/exp004-minilm-learning-curve/runs --output artifacts/exp004-summary-new
```

Choose a fresh output directory for any later reproduction. Omit `--reuse-smoke` when EXP-003's local binary artifacts are unavailable: all three groups will be encoded independently for the primary cache too, using the same pinned model/settings and existing IDs. Fresh model/dependency downloads need network access; offline runs are possible with the existing model cache and `HF_HUB_OFFLINE=1`. No raw/test parsing or fitting is needed for compact-record aggregation.

Full artifacts remain ignored under `artifacts/exp004-minilm-learning-curve/`: separate `runs/` and `reproductions/`, classifiers/probabilities, source snapshots, and both row-indexed feature caches. Downloaded weights remain ignored in `.cache/huggingface/hub`. Only compact JSON/CSV evidence, the plot, code, tests and documentation are versioned. Keep failed artifacts if future runs fail; never overwrite this completed study.
