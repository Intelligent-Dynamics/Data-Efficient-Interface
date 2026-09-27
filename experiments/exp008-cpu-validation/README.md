# EXP-008 — measured frozen seed-11 CPU pipeline

Completed **2026-09-27** using the existing 20-shot, seed-11 specialist and all 770 existing validation texts. This is one representative deployed model, not an ensemble. No fitting, new labels, cached embeddings in the timed path, network/API calls, or official-test access occurred. The official test had previously been mechanically opened by authorized preflight; it was not accessed, encoded, predicted or scored in this study.

The procedure was frozen in `protocol.json` at **19:52:23 UTC**, before the measured run began at **19:54:38 UTC**. Protocol SHA-256: `1a1c02fd3d1a72ddfa45cd844e47b8602cc143f8649919fe0b2700509ca6043e`. The original EXP-007 protocol remains unchanged at `0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`.

## Measurements

Machine: **Apple M1 Pro, 10 CPU cores, 32 GiB RAM**, macOS 26.5.1 arm64, Python 3.13.0. CPU float32 MiniLM, one Torch intra-op/inter-op thread and one native compute thread; tokenizer parallelism disabled. Complete package/native-runtime versions are recorded in `results.json`. Hardware power/thermal conditions and unrelated OS activity were not controlled.

| Measurement | Observed value |
| --- | ---: |
| Single-request median end-to-end local latency | **5.906833 ms** |
| Single-request p95 end-to-end local latency | **7.824673 ms** |
| Batch-size-32 pooled throughput | **370.1814 requests/second** |
| Encoder loading, including lazy encoder-library import | **2.424610 s** |
| Saved LR classifier loading | **0.000630 s** |
| Process RSS before model loading | **265.796875 MiB** |
| Process RSS after model loading | **513.531250 MiB** |
| Process lifetime peak RSS by final timed checkpoint | **582.921875 MiB** |
| Process RSS at final timed checkpoint | **500.625000 MiB** |

The single-request timer includes raw-text tokenization, frozen encoder inference, normalization, LR probabilities, and the unchanged scalar routing gate. Raw texts are loaded before timing. Per-batch logging, evidence checks, file hashing and network are outside the timer. Model-loading times may benefit from a warm filesystem; they are not guaranteed cold-start measurements. Memory uses `ps` RSS in KiB converted to bytes and `resource.ru_maxrss` (bytes on macOS), includes the whole Python process/libraries/evidence, and does not isolate model allocation or continuously sample transient memory.

Each batching mode had one untimed full-validation warmup and five timed full passes. Mode order was batch size 1 followed by 32; all original validation rows were kept in their saved order. The last batch contains two requests. Every pass and all 3,850 request timings / 125 batch timings are retained. Pooled request latency uses NumPy's linear percentile, not a percentile of pass-level medians. Batched throughput divides 3,850 requests by the sum of measured local-inference batch durations and **is not interactive request latency**.

| Pass | Single-request pass duration (s) | Batch-size-32 pass duration (s) | Batch-size-32 throughput (requests/s) |
| --- | ---: | ---: | ---: |
| 1 | 4.944553 | 1.954066 | 394.0502 |
| 2 | 4.768036 | 1.947906 | 395.2962 |
| 3 | 4.857303 | 1.905090 | 404.1805 |
| 4 | 4.894688 | 2.500863 | 307.8937 |
| 5 | 4.596362 | 2.092382 | 368.0017 |

The slower fourth batched pass is preserved. These are machine/workload-specific measurements, not a concurrent serving benchmark, cloud cost estimate, total-system savings claim or evidence that local compute is free.

## Output and isolation verification

Every pass had **zero predicted-label mismatches and zero routing-decision mismatches** against the saved seed-11 validation probabilities. The unchanged threshold `0.10354116298070118` accepted **693/770** cases independently. No ranking, forced coverage or tolerance was applied.

Batching/padding produced small numerical differences: maximum absolute probability/confidence differences were **2.5165506478685984e-7** (single request) and **2.7783588352203736e-7** (batch 32). They did not change any prediction or gate. Within each batching mode, all five full prediction/confidence/gate hashes are identical. First-pass per-row predictions/confidences/gates are retained for each mode; no raw customer text is included.

Encoder state and classifier parameter hashes matched before/after, and all pinned model/package files were checked. The process audit hook recorded **zero attempted official-test accesses and zero network accesses**. The loader resolves only the saved validation IDs from the pinned **training** CSV, avoiding the older development loader that hashes the official test. Inference receives ID/text only. Existing label budget remains 1,540 training labels plus 770 development-validation labels, with no additional labels.

## Compatibility finding and preserved failed attempt

The first launch stopped **before any timed inference** because the existing `baseline.final_specialists._load_local_encoder` asserted that the encoder prompt dictionary must be empty. Pinned SentenceTransformer 5.1.1 initializes `{'query': '', 'document': ''}` even when constructed with `prompts={}`. Those empty names apply no prompt when `default_prompt_name=None` and `encode(prompt=None, prompt_name=None)`.

The benchmark's small companion loader keeps all frozen constructor/encoding settings and checks that every named prompt is empty and the default is absent, then validates the frozen encoder state. It does not clear prompt mappings or alter weights. This is a dependency compatibility check, not a model/prompt change. The original EXP-007 runner is untouched; its overly strict assertion remains a **blocker for future original-runner execution**, requiring a separate minimal compatibility review before any authorized test inference. No benchmark measurement was selected or discarded based on speed/quality; the original failed launch yielded no timed pass.

Focused synthetic tests passed **12/12** after fixing an initial fixture-directory setup error. The full project suite is recorded by the enclosing v1 checkpoint. `reproduce.py` independently recomputes timing summaries and verifies compact output ordering/gates without loading data or models:

```sh
.venv/bin/python experiments/exp008-cpu-validation/reproduce.py
```

The executed measurement command was:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m baseline.cpu_benchmark --run
```

The command refuses to overwrite existing results. Source hashes, base Git commit, predeclared protocol, timestamps and compact numerical evidence are preserved. Future repetition requires a separately named evidence location/protocol revision rather than erasing this run.
