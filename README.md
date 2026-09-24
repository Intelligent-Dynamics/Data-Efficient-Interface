# Data-Efficient Specialist Inference

An **Intelligent Dynamics** research project exploring specialized AI systems that learn narrow business tasks from limited labeled data.

## Research question

Can a small specialist model trained with limited labeled data handle a significant portion of a narrow business classification workload while uncertain cases fall back to a stronger general-purpose model, reducing benchmarked inference cost without materially reducing model quality?

## Current status

The minimal CPU baseline is implemented. Experimental observations and their verification status are recorded in [the experiment log](docs/EXPERIMENTS.md); there are no final project results or cost-savings claims.

The pipeline uses BANKING77, word unigram/bigram TF-IDF, and L2 logistic regression, with a most-frequent-class sanity comparator. The official test set is excluded from model development. Routing, calibration, general-purpose models, and a frontend are deferred.

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

## Project documents

- [Project context](docs/PROJECT_CONTEXT.md): thesis, dataset comparison, and evidence standards.
- [Current plan](docs/CURRENT_PLAN.md): exact next milestone and evaluation protocol.
- [Experiment log](docs/EXPERIMENTS.md): planned and executed experiments, including negative findings.
- [Verified results](docs/RESULTS.md): reproducible measurements only.
- [Agent instructions](AGENTS.md): operational guidance for Codex sessions.

## Research principles

Never fabricate metrics or savings. Prevent test-set leakage, count all development labels, start with simple baselines, and distinguish actual spend from benchmark estimates and projected savings. Preserve negative results and update project context when decisions change.

**Measure → understand → improve → re-measure.**
