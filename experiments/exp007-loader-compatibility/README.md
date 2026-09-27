# EXP-007 original encoder-loader compatibility fix

This source-only compatibility fix resolves the original loader blocker found during EXP-008. It changes no experimental protocol, model, prompt, threshold, encoding setting or evaluation definition. No final-test access, API call, model fitting or CPU timing rerun occurred.

## Assertion

`baseline.final_specialists._load_local_encoder` previously required a falsey `encoder.prompts`. It now requires a `collections.abc.Mapping` whose keys are strings and whose values are all empty strings, together with `default_prompt_name is None`. An empty mapping satisfies the same rule. Nonempty strings (including whitespace), malformed mappings/values and every non-None default are rejected. The mapping is never cleared or mutated. Constructor arguments and freezing remain unchanged.

32 new synthetic cases exercise accepted empty mappings, invalid mappings/values, selected defaults, constructor settings, unchanged metadata and frozen state.

## Actual original-loader verification

[validation_check.json](validation_check.json) records one fresh **untimed** encoding of the existing 770 validation texts through the patched **original** loader and all five saved 20-shot classifiers. The CPU benchmark companion loader is not used. Its existing offline guard and training-only validation-input reader are reused.

- Actual pinned encoder metadata: `{"query": "", "document": ""}`, default `None`.
- Encoder state matches the protocol before and after; zero trainable parameters.
- Pinned encoder files, all five classifier files, dependency versions, classifier settings/class order and classifier state pass before/after checks.
- Seeds 11/22/33/44/55: maximum absolute probability and confidence difference **0.0**, no predicted-label mismatch, no routing mismatch; each accepts **693/770**.
- Network and official-test guard counters: **0**. Existing validation IDs/text provenance is checked; no test row or test-file checksum is read.

The zero differences describe this untimed full-list batch-32 call. They do not replace EXP-008's historical timing-path numerical differences, which remain recorded unchanged.

Reproduce offline in the canonical checkout with the pinned local environment/artifacts, writing a new file rather than overwriting this evidence:

```sh
.venv/bin/python experiments/exp007-loader-compatibility/verify_validation.py > /private/tmp/exp007-loader-validation-recheck.json
```

## Integrity and provenance

Base commit: `b5a2191d97f00a25344f65b35c3b5d86b5696237`.

- Previous loader source SHA-256: `036af93ee40b74ea4563f7e5b445cd7ba74ff0d0ffdcbdaf848b4aed38f441b1`.
- Patched loader source SHA-256: `c443f4a4136f4737e9b4188dcd3007e6a4c461627dd630deebbc67406fe41961`.
- EXP-007 protocol unchanged: `0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00`.
- EXP-009 protocol unchanged: `b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334`.

Neither frozen protocol binds this loader's source hash; existing preparation bindings still pass. Historical verification records retain their original source hashes and were not regenerated. Runtime resume source checks remain intact: a live run bound to other source bytes must still reject drift. No protection was disabled or rebound. This check does not inspect any real final-test execution artifact.

Full-suite result and separate source/evidence hashes are recorded in [verification.json](verification.json). This compatibility record grants no pilot, test-access or paid-execution authorization.
