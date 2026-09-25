# EXP-003 — frozen MiniLM validation smoke

**One smoke experiment, not a final research result or independent training reproduction.** Run ID: `exp003-minilm-v2-n5-s11`. Completed 2026-09-25T17:50:15.549859+00:00. All 77 intents, 5 shots/class, seed 11, protocol `banking77-val10-v2`.

| 5-shot, seed 11; same v2 IDs | Accuracy (%) | Macro-F1 (%) |
| --- | ---: | ---: |
| TF-IDF + logistic regression (`exp002-v2-n5-s11`) | 52.21 | 51.13 |
| Frozen MiniLM + unchanged logistic regression | 75.84 | 74.46 |
| Difference (percentage points) | +23.64 | +23.33 |

Exact values, per-class metrics and confusion counts are in `metrics.json`; the paired comparison is in `comparison.json`. Predictions and ordered sample IDs are retained. The comparator is the versioned [EXP-002 v2 5-shot seed-11 run](../exp002-learning-curve/runs/exp002-v2-n5-s11.json), not the obsolete validation=20 smoke. No comparator refit occurred.

## Frozen configuration and provenance

- Model: [sentence-transformers/all-MiniLM-L6-v2 at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41). Publisher metadata identifies Apache-2.0. The model card, weights, tokenizer and configuration hashes are in `metadata.json`.
- Packages: sentence-transformers 5.1.1, torch 2.8.0, transformers 4.56.2; scikit-learn 1.7.2, NumPy 2.3.3 and SciPy 1.16.2 unchanged. All transitive versions are in `uv.lock` and the installed package inventory in metadata.
- CPU float32, 384 dimensions, masked mean pooling, L2 normalization, batch 32, maximum length 256 tokens, no prompts, eager attention, safetensors, no remote code. PyTorch seed 11 and deterministic algorithms; one compute thread. Encoder evaluation/inference mode; all parameters frozen.
- Logistic regression imported unchanged from the TF-IDF implementation: L2, C=1, lbfgs, max_iter=2000, tol=1e-4, intercept, no class weighting, multinomial loss. No scaling, tuning, or classifier-setting changes. Converged in 9 iterations without warnings.
- Machine: Apple M1 Pro, arm64, 10 logical CPUs, 32 GiB RAM, macOS 26.5.1, Python 3.13.0. CPU execution only.
- Code: base commit `059cd9d65f6dde72955f214cbfe8bb75d489e4b7` plus the hashed implementation snapshot. `protocol.md` is the pre-run plan. Full source snapshot and working-tree patch are retained locally under the ignored run directory; the milestone commit contains the run source/lockfile unchanged, with post-run documentation updates.
- Dataset revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`. Split-manifest SHA-256: `f07ac5a03a3ae444db042dd064db920daa3f6462b1e92c9df36ad6118196030a`. Sample SHA-256: `018f481053a04611d7171083b1235e8108a26a1a4ac8d4a3eb3cba210e481af8`. Exact sample bytes equal the reference. No new split or sample was generated; the existing loader mechanically verifies the original split.

## Label budget and limitations

385 training labels **plus 770 additional validation labels** (10/class), totaling 1,155 unique task examples. They are the same labels used by the matching TF-IDF run: this experiment adds no new unique task labels. All 10,003 official-training labels are mechanically read for stratification/audit and disclosed separately. No calibration, prompt, or test-evaluation labels were used.

The encoder benefits from external pretraining. Five-shot describes task-specific classifier fitting, not training the representation from scratch or measuring total annotation cost. Public-benchmark overlap in upstream pretraining cannot be ruled out. This holdout has only 10 cases/class and has already been used in development. One training seed gives no seed-variance estimate and the paired difference is not a confirmatory significance or generalization claim.

The official test set remained sealed: byte checksum verification only, no test-row parsing, predictions, labels, tuning, or qualitative inspection. The existing split audit supplies test IDs/text hashes for isolation checks.

## Resource observations

| Component | Wall seconds |
| --- | ---: |
| Snapshot/cache resolution (weights already cached) | 0.335369 |
| Model loading (after library imports) | 0.100827 |
| Encoding 385 training texts | 0.674077 |
| Encoding 770 validation texts | 1.102255 |
| Classifier fit on training embeddings | 0.018407 |
| Classifier predict on validation embeddings | 0.000910 |

Single measured passes; full precision is in metadata. No warmup/repeated latency study was performed. Validation encoding follows training encoding, so it is not a cold-model benchmark. Model files were downloaded before the run; initial download duration and first library imports are excluded. Hashing, serialization, and probability generation are excluded from the component timers. Do not compare these timings directly with EXP-002's warmed repeated timings. These are not monetary cost measurements or evidence of production savings.

## Verification and reproduction commands

47 tests pass. Tests use synthetic data/encoders; only one real BANKING77 fit occurred. `verification.json` confirms artifact/source hashes, unchanged encoder state, sample membership/budgets/isolation, feature row/text/label alignment, unchanged LR, prediction/probability replay, and exact metric/comparison recomputation. The encoder has 22,713,216 parameters, zero trainable parameters and no gradients. Before/after state SHA-256 is `33ce382d629c3737ea2dab072e1e1e6286abfd23741655c3bb719c3f336d3456`. Artifact replay does not constitute an independent training reproduction.

From the canonical repository root, with the existing raw data and v2 manifest:

```sh
uv sync --locked --cache-dir .cache/uv
.venv/bin/python -m pytest -q
# Exact command used for the only experiment in this milestone:
HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m baseline.embeddings --output artifacts/exp003-minilm-v2-n5-s11
# Verification only: no encoding, fitting, or test evaluation:
.venv/bin/python -m baseline.embeddings --verify --output artifacts/exp003-minilm-v2-n5-s11
```

An existing output directory is refused. To reproduce later, choose a fresh output directory; no independent reproduction or second experiment was run in this milestone. Downloaded model files live in ignored `.cache/huggingface/hub`; row-indexed `train_embeddings.npz` / `validation_embeddings.npz`, full probabilities, `classifier.joblib`, split audit, and source snapshot live in ignored `artifacts/exp003-minilm-v2-n5-s11`. Compact evidence here is sufficient to recompute classification metrics from the saved predictions. Raw data, weights, embedding caches, and binary model files are not committed.
