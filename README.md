# Data-Efficient Specialist Inference

An **Intelligent Dynamics** research project exploring specialized AI systems that learn narrow business tasks from limited labeled data.

## Research question

Can a small specialist model trained with limited labeled data handle a significant portion of a narrow business classification workload while uncertain cases fall back to a stronger general-purpose model, reducing benchmarked inference cost without materially reducing model quality?

## Current status

Research planning and repository setup only. No ML implementation or experiments have been completed, and no quality or cost-savings results are claimed.

The next implementation milestone is a local BANKING77 few-shot text-classification baseline using TF-IDF and logistic regression. The plan specifies training budgets, validation methodology, leakage checks, and reproducibility requirements. Routing, general-purpose model integration, and a frontend are deferred.

## Project documents

- [Project context](docs/PROJECT_CONTEXT.md): thesis, dataset comparison, and evidence standards.
- [Current plan](docs/CURRENT_PLAN.md): exact next milestone and evaluation protocol.
- [Experiment log](docs/EXPERIMENTS.md): planned and executed experiments, including negative findings.
- [Verified results](docs/RESULTS.md): reproducible measurements only.
- [Agent instructions](AGENTS.md): operational guidance for Codex sessions.

## Research principles

Never fabricate metrics or savings. Prevent test-set leakage, count all development labels, start with simple baselines, and distinguish actual spend from benchmark estimates and projected savings. Preserve negative results and update project context when decisions change.

**Measure → understand → improve → re-measure.**
