# EXP-009 completed official-test companion

**Status: complete.** This is one frozen seed-11 specialist/training pool on the 3,080 official BANKING77 test cases. Twenty examples per class use 1,540 training labels; the threshold and development work additionally use 770 validation labels. The frozen scalar threshold is `0.10354116298070118`. Retrieval uses 20 examples from that same training pool, not extra labeled demonstrations.

| Policy (matched seed-11 comparison) | Correct / 3,080 | Accuracy | Macro-F1 | LLM fraction in the policy |
| --- | ---: | ---: | ---: | ---: |
| Specialist | 2,636 | 85.584416% | 85.347125% | 0% |
| Zero-shot Luna | 2,506 | 81.363636% | 80.585867% | 100% |
| Zero-shot hybrid | 2,665 | 86.525974% | 86.389275% | 8.701299% |
| Retrieved-example Luna | 2,839 | 92.175325% | 92.134813% | 100% |
| Retrieved-example hybrid | 2,713 | 88.084416% | 87.982622% | 8.701299% |

The retrieved hybrid improves specialist accuracy by **2.500000 percentage points**, accepting 2,812 specialist predictions (coverage `0.912987012987013`) and sending 268 requests to Luna. On these same 268 rejected cases, retrieved Luna gets 202 correct (`0.753731343283582`), versus 154 (`0.5746268656716418`) for zero-shot Luna. Retrieved all-Luna accuracy is 4.090909 points above the retrieved hybrid, demonstrating a quality-versus-LLM-usage tradeoff. The study itself collected Luna on all 3,080 cases; the policy fraction is not the study API request fraction or a production cost-saving estimate.

These are **single-seed companion results**, with no seed-variance estimate. EXP-007's five-seed specialist and zero-shot-hybrid means remain separate in `../exp007-official-test/`; its seed SD is not a confidence interval, and its five seeds share one test population. The three controls here exactly reproduce that bundle's seed-11/same-population results. No thresholds, prompts, retrieval settings, or models were tuned from test outcomes.

## Evidence and verification

- `summary.json`: exact saved summary floats, all five arms, all five paired differences, label budgets, gate and fallback counts. Percentages in this README are rounded for display only.
- `per_class.json`: all saved TP/FP/FN/support and precision/recall/F1 statistics for 77 intents and five arms; no sample IDs or text.
- `accounting.json`: independently replayed response, ledger, usage, compatibility, and artifact-hash audit; no response bodies or provider IDs.
- `provenance.json`: original source/artifact and matched EXP-007 control bindings, plus scope of read-only alignment checks.
- `verification.json`: compact file hashes and verification/test status. It does not hash itself.
- `reproduce.py`: standard-library replay using compact sufficient statistics only.

The read-only export checked every saved evaluation/execution/specialist ID exactly once and aligned classes and scalar gates; all five prediction marginal counts match per-class TP+FP. Frozen prediction SHA-256 is `c2963a59ca83911a9ccacf9390390fed18fb9bb5282bf38ae1b33b5c44be2a8c`. Protocol SHA-256 remains `b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334`. All **3,092 original artifact files** and the original 826 response files bound by the resume receipt were unchanged.

Accuracy and macro-F1 replay from saved per-class sufficient statistics. The largest independent arithmetic difference from stored values is **1.1102230246251565e-16**, below 1e-12; stored source floats remain unchanged. This verification does not reconstruct confusion cells or newly join row-level truth labels. Fallback correct counts are checked against the saved ratios and both hybrid correct totals. No raw test CSV was reopened and no inference, training, API, preflight, or new experiment was run during export.

```sh
python3 -B experiments/exp009-official-test/reproduce.py
```

## Accounting and limitations

All 3,080 final outputs are `ok`, from 3,083 attempts. Three `transport_unknown` attempts have unknown usage and were each followed by a successful retry. Usage-priced charges are **$0.576343155**, with **$0.0084480** reserved for unknown usage: frozen-assumption interval **[$0.576343155, $0.584791155]**. Actual invoice spend remains unknown and unreconciled. Original approved cap was $36.00; full-retry reservation was $35.32756300. These are accounting bounds, not additional charges.

Single seed/single training pool and one public benchmark limit generalization. Test access was one-time evaluation under frozen protocols; results do not authorize tuning or reruns. LLM request reduction does not equal dollar savings: specialist deployment, retrieval, and production serving costs remain unmeasured. Calibration, SetFit/fine-tuned specialists, CLINC150/second-dataset work, self-hosted vLLM/H100 work, and a production cost study are deferred to v2.
