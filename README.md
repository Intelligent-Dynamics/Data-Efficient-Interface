# Data-Efficient Specialist Inference

An **Intelligent Dynamics** research project exploring specialized AI systems that learn narrow business tasks from limited labeled data.

## Research question

Can a small specialist model trained with limited labeled data handle a significant portion of a narrow business classification workload while uncertain cases fall back to a stronger general-purpose model, reducing benchmarked inference cost without materially reducing model quality?

## Current status

The classical TF-IDF and frozen MiniLM 5/10/20-shot validation curves are measured across five seeds and independently reproduced locally. See [verified validation results](docs/RESULTS.md), [the paired MiniLM/TF-IDF study](experiments/exp004-minilm-learning-curve/README.md), and [the experiment log](docs/EXPERIMENTS.md). Every configuration uses the same existing BANKING77 v2 sample IDs; only logistic regression is trained for the frozen embedding model. Official-test performance, end-to-end embedding inference speed, and cost savings remain unmeasured. Fine-tuning, serving/routing infrastructure, calibration and frontend work are deferred. EXP-006R recovery is prepared but has not run; it needs new spending authorization.

## Run locally

Use Python 3.13.0 and [uv](https://docs.astral.sh/uv/). The canonical checkout is `/Users/Andrew/Developer/data-efficient-inference`; run commands from that repository root. `uv.lock` pins all resolved dependencies; the first sync and uncached dataset preparation need network access.

```sh
cd /Users/Andrew/Developer/data-efficient-inference
uv sync --locked --cache-dir .cache/uv
.venv/bin/python -m pytest -q
.venv/bin/python -m baseline prepare
.venv/bin/python -m baseline run --shots 5 --seed 11 --output artifacts/exp001-v2-n5-s11
```

Choose a new output directory for each run; existing runs are never overwritten. Preparation is idempotent, checks file hashes, and refuses to overwrite a changed split manifest. Once files are cached, preparation and training work offline.

The source is fixed to publisher revision `57ec275d8078af65b7731c2a98be812d844a6d6b`, with SHA-256 checksums in [the source manifest](data/banking77-source.json). Preparation audits exact/normalized duplicates, leaves official test rows untouched, and stores only test IDs and text hashes in the split manifest. Training loads only training-source texts and verifies test-file byte integrity. There is no test-evaluation CLI.

Protocol `banking77-val10-v2` reserves **10 examples per class** (770 total), fixed with seed `20260924` and shared across every regime and sampling seed. Training seeds `11`, `22`, `33`, `44`, `55` define nested **5-, 10-, and 20-shot** subsets. All three regimes are feasible across all 77 classes: the smallest class has 35 usable source rows and retains 25. **50-shot is dropped**, since supporting it would require excluded classes, repeated examples, or new data. No official test examples enter development.

The default manifest is `data/processed/banking77-val10-v2/manifest.json`. The legacy validation=20 manifest and smoke artifacts are preserved; current code rejects that old manifest. Earlier smoke scores must not be pooled with new-protocol results. The 770-example holdout supports coarse early model comparisons, but not precise per-class estimates or trustworthy high-confidence calibration claims. See [the current plan](docs/CURRENT_PLAN.md) for the limitations and future calibration requirements.

Each run saves `metadata.json`, `metrics.json`, `samples.json`, `split_manifest.json`, `predictions.json`, `probabilities.json`, `model.joblib`, and a source snapshot. Metrics include accuracy, macro-F1 over all 77 classes, per-class precision/recall/F1/support, and confusion counts. Metadata records the label budget, source/configuration, Git revision plus uncommitted-source hashes, dependencies, warning/failure status, single-thread fit/prediction timings, and model size. TF-IDF fits only the sampled training texts. Label access for stratification/auditing is disclosed separately from the few-shot fitting budget.

Downloaded files, environments, and bulky artifacts are ignored by Git. Compact smoke-run evidence is kept under `experiments/`; full local artifacts can be regenerated with the recorded source and commands. Load serialized models only from trusted runs.

Dataset attribution: Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*. [Publisher](https://github.com/PolyAI-LDN/task-specific-datasets), [CC BY 4.0 license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE). The downloader retains the upstream license. Split filtering and subsampling are project modifications.

## Learning-curve analysis

EXP-002 includes 15 primary runs plus 15 independent verification refits. All seeds are retained; summary SD is sample SD across training seeds, not a confidence interval. Recompute its summary, error analysis, CSV, and PNG/SVG figures from the checked-in records without downloading data or fitting models:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m baseline.learning_curve --records-dir experiments/exp002-learning-curve/runs --output artifacts/exp002-summary-recomputed
```

Use a fresh output directory. [The study README](experiments/exp002-learning-curve/README.md) contains full-matrix fitting commands, artifact verification, interpretation, and limitations. Large model/probability artifacts remain local; their hashes and compact records are versioned.

## Frozen embedding smoke baseline

EXP-003 adds frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings plus the same logistic regression. The original 5-shot, seed-11 smoke is preserved; see [the smoke comparison and evidence](experiments/exp003-minilm-v2-n5-s11/README.md). It reuses the exact EXP-002 v2 training/validation IDs and keeps the official test sealed. This is not a final research result.

The encoder revision and dependency versions are pinned. Weights download to ignored `.cache/huggingface/hub`; cached features and classifiers stay in ignored `artifacts/`. The command requires the existing v2 manifest and versioned comparator and cannot prepare or select a new split. It trains only logistic regression and checks unchanged encoder state. The label budget is 385 fitting plus 770 validation labels, and the representation benefits from external pretraining.

```sh
# Verification of the existing local smoke; no new fitting or encoding:
.venv/bin/python -m baseline.embeddings --verify --output artifacts/exp003-minilm-v2-n5-s11
```

The study README records the execution command and future reproduction instructions. EXP-004 subsequently completed all 15 configurations and independent refits. All original TF-IDF commands, results and smoke artifacts are preserved.

## Full frozen MiniLM comparison

[EXP-004 evidence](experiments/exp004-minilm-learning-curve/README.md) includes all 15 primary runs, 15 independently re-encoded/refitted reproductions, exact paired seed gains, cache provenance and the comparison plot. Encoder revision/settings and LR hyperparameters are unchanged from EXP-003; no tuning or test evaluation occurred. See the study README for full execution and cache-reuse commands.

Recompute summary/CSV/plot from versioned records without fitting or encoding:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m baseline.minilm_curve --records-dir experiments/exp004-minilm-learning-curve/runs --output artifacts/exp004-summary-new
```

Use a fresh output directory. Sample SD reflects five training seeds, not uncertainty over new validation samples. The additional 770 validation labels and pretrained representation are disclosed. Cached classifier-only timings are explicitly not end-to-end inference speed.

## Confidence and selective-prediction diagnostic

[EXP-005 evidence](experiments/exp005-selective-diagnostic/README.md) reports all 15 existing MiniLM runs at approximately 25/50/75/90/100% validation coverage, full risk/coverage curves, sample SDs and per-intent acceptance/error counts. Confidence is uncalibrated; the same 770 cases are reused. High aggregate accepted accuracy can hide entirely rejected intents. No model was fitted/executed and no production threshold or fallback claim is established.

Reproduce tables and curves using compact records alone, with a fresh output path:

```sh
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m baseline.selective --records-dir experiments/exp005-selective-diagnostic/runs --output artifacts/exp005-summary-new
```

Original probabilities/caches/models stay ignored. The study README records provenance checks and a separate independent checker; the official test remains sealed.

## General-model evaluation and separate recovery preparation

The saved EXP-006 run finished with **718 successful predictions and 52 unresolved HTTP-429 requests**. Its original evidence and two-attempt limit are preserved. [EXP-006R preparation](experiments/exp006r-429-recovery-preparation/README.md) implements a separately authorized recovery for only those 52 IDs, plus an offline merge/evaluation command. **Paid recovery has NOT RUN.** The original errors report exhausted credits; resolve quota before a later live run.

No network, credentials or raw dataset access is needed for the recovery dry run:

```sh
.venv/bin/python -m baseline.recovery dry-run --output artifacts/exp006r-dry-run-new
```

The recovery write-up contains its protocol hash, cost estimate, retry policy, tests and exact future live/merge commands. New explicit authorization and a numeric cap are required; neither the earlier EXP-006 approval nor a documented command is permission. Keep keys and response caches out of Git. Preserve the [original EXP-006 protocol](experiments/exp006-general-model-preparation/README.md). No recovered quality result or total-system savings is claimed.

## Project documents

- [Project context](docs/PROJECT_CONTEXT.md): thesis, dataset comparison, and evidence standards.
- [Current plan](docs/CURRENT_PLAN.md): exact next milestone and evaluation protocol.
- [Experiment log](docs/EXPERIMENTS.md): planned and executed experiments, including negative findings.
- [Verified results](docs/RESULTS.md): reproducible measurements only.
- [Agent instructions](AGENTS.md): operational guidance for Codex sessions.

## Research principles

Never fabricate metrics or savings. Prevent test-set leakage, count all development labels, start with simple baselines, and distinguish actual spend from benchmark estimates and projected savings. Preserve negative results and update project context when decisions change.

**Measure → understand → improve → re-measure.**
