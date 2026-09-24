# Project context

Updated: 2026-09-24. Organization: Intelligent Dynamics.
Project: **Data-Efficient Specialist Inference**.
Canonical checkout: `/Users/Andrew/Developer/data-efficient-inference`. The existing checkout, including Git history and local artifacts, was moved here from the former Documents/ChatGPT path on 2026-09-24. Origin remains `https://github.com/Intelligent-Dynamics/data-efficient-inference.git`.

## Thesis and research question

Intelligent Dynamics explores data-efficient AI: specialized systems that adapt to narrow tasks with limited labeled data and reduce unnecessary reliance on expensive general-purpose models.

**Can a small specialist model trained with limited labeled data handle a significant portion of a narrow business classification workload while uncertain cases fall back to a stronger general-purpose model, reducing benchmarked inference cost without materially reducing model quality?**

The hypothesis is that recurring intent patterns may be learnable by a cheap specialist, while difficult cases warrant a stronger model. This is unproven. A specialist may be inaccurate, confidently wrong, or offer too little useful coverage to offset its overhead. Negative findings are valuable.

Operating principle: **Measure → understand → improve → re-measure.**

The eventual evidence-backed statement would be: “Using N labeled examples per class, our specialist handled X% of requests and reduced benchmarked inference cost by Y% while remaining within Z percentage points of the stronger-model baseline.” N, X, Y, and Z must come from recorded experiments, with the metric, label budget, workload, uncertainty, and cost assumptions disclosed. No values are available yet.

## Initial repository inspection

At the initial inspection on 2026-09-24 the repository contained only Git metadata, no tracked files, and no commits on `master`. There was no implementation, dependency configuration, test suite, dataset, or prior result. The first session established documentation only.

## Implementation state (2026-09-24)

The local baseline now has pinned dataset downloads/dependencies, normalized-duplicate auditing, fixed validation and nested training samples, TF-IDF + logistic regression, a dummy comparator, JSON provenance/metrics/predictions, and automated tests. EXP-002 has completed the full 5/10/20-shot learning curve across five seeds, with all 15 configurations independently refitted locally and matched exactly. Reproducible validation aggregates and their limitations are in `RESULTS.md`; the earlier validation=20 smoke remains separate historical evidence. There is no routing, calibration, cost model, API, or frontend implementation.

The initial validation=20 design left too few examples in the smallest class (35 usable source rows). At the user's direction, protocol `banking77-val10-v2` now reserves a fixed **10 examples per class** from official training data for validation/calibration, leaving at least 25 training candidates per class. Initial regimes are **5, 10, and 20 shots**, all with the same 770 held-out validation examples and sampling seeds 11/22/33/44/55. Fifty-shot is dropped: even without a holdout the smallest class cannot supply 50 unique examples. All 77 classes, duplicate/ID isolation, deterministic nested sampling, and the sealed official test split are preserved. All training-source labels read for stratification/auditing are disclosed separately from sampled fitting/validation labels; this is a simulated few-shot benchmark, not proof of total annotation requirements.

The existing N=5/seed=11 smoke run used the old validation=20 protocol. Keep its artifacts and metrics as historical evidence; do not compare or aggregate its score directly with new-protocol results. The revised protocol now has 15 primary validation runs and 15 independent local refits. Mean macro-F1 rises from 50.41% to 61.58% to 69.19% at 5/10/20 shots; these are development-set measurements, not test or production results. No routing/cost claim is established.

Ten validation examples per class (770 overall) are a reasonable development budget for coarse comparisons among a small, prespecified set of simple baselines. They do not guarantee sensitivity to small differences: each additional correct case changes a class's recall by 10 percentage points, and macro-F1 can be noisy. Reusing this fixed holdout for many tuning decisions risks overfitting; report paired development comparisons and training-seed variability, not definitive quality claims. For later calibration, this is a shared held-out development pool, not independent calibration and evaluation sets. Do not fit calibration or select thresholds on these labels and then call performance on the same labels unbiased. Reliable per-class calibration, high-confidence tail estimates, and narrow non-inferiority claims will need a separately designed, label-budgeted calibration/evaluation procedure (additional development labels or suitable cross-fitting). The official test split remains sealed during that work.

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

Later, evaluate the specialist alone, stronger model alone, and combined policy on the same frozen test requests against dataset labels. The stronger model is a comparator, not ground truth. Freeze its model version, prompt, decoding, label mapping, retry policy, and invalid-output handling. Prompt examples must come from an explicitly counted development budget. Tune routing only on development data; evaluate accepted-subset quality, coverage, fallback rate, and full-system quality separately.

Before routing experiments, predeclare the acceptable quality loss and useful coverage target. For a higher-is-better metric Q, quality loss in percentage points is `100 * (Q_strong - Q_hybrid)` when Q is on [0, 1]. A non-inferiority claim needs a prespecified margin and uncertainty analysis, not just a favorable point estimate. Use paired comparisons on the same requests. Do not assume the stronger model handles fallback cases perfectly.

For M benchmark requests and accepted set A:

- Coverage = `|A| / M`.
- All-strong cost = sum of each request's measured usage priced under a frozen pricing schedule.
- Hybrid cost = specialist inference on every request + routing overhead + strong-model cost for each fallback request.
- Benchmarked cost reduction = `100 * (1 - hybrid_cost / all_strong_cost)`; the denominator must be positive. Negative savings remain results.

Use request-level token usage because fallback requests may be longer or more expensive. Record pricing date, currency, cache/batch discounts, retries, and hardware/rate/utilization assumptions. Record local timing even when a defensible monetary rate is unavailable; label any monetization as an estimate.

**Actual spend** is what the experiment consumed or was billed, including baseline calls, development calls, failed calls, and retries. **Benchmarked inference cost** is the cost of a defined policy on the fixed workload; replaying cached stronger-model outputs estimates a counterfactual and does not create actual cash savings. **Projected deployment savings** additionally assume volume and traffic mix and must be labeled projections. Training, labeling, and development costs are reported separately from per-request inference costs and included in any later break-even analysis.

## Scope boundaries and persistent records

Begin with a local reproducible experiment. No frontend, serving stack, database, orchestration platform, GPU fine-tuning, or routing implementation is needed now.

- `AGENTS.md`: operational rules for future sessions.
- `CURRENT_PLAN.md`: current milestone, protocol, decisions, tasks, and unresolved choices.
- `EXPERIMENTS.md`: chronological experiment history, including failures and deviations.
- `RESULTS.md`: verified reproducible project measurements only.

Update these records when major decisions change. Keep prior experiment records intact; supersede with a dated explanation.
