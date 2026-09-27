"""One untimed offline validation of the patched ORIGINAL loader; no test access."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                  TOKENIZERS_PARALLELISM='false')
from baseline.cpu_benchmark import offline_guard, validation_inputs
GUARDS = offline_guard(ROOT)

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from baseline import final_specialists as original
from baseline.data import read_json, sha256, json_bytes
from baseline.embeddings import ENCODING, assert_frozen, state_digest
from baseline.final_protocol import load_frozen, verify_preparation
from baseline.general import now
from baseline.retrieved_run import load_protocol as load_companion_protocol
from baseline.selective import require


def verify():
    protocol, _, _ = load_frozen()
    companion, _, _ = load_companion_protocol()
    verify_preparation(ROOT, protocol)  # Existing source/package bindings; no test-file I/O.
    checked = original.verify_specialist_files(protocol)
    rows, labels, _, train_hash = validation_inputs(protocol)
    ids = [r['id'] for r in rows]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(11)
    torch.use_deterministic_algorithms(True)
    reports = {}
    with threadpool_limits(limits=1):
        encoder = original._load_local_encoder(checked['snapshot'])
        assert_frozen(encoder)
        prompts = dict(encoder.prompts)
        state_before = state_digest(encoder)
        require(state_before == protocol['specialists']['embedding']['encoder_state_sha256'],
                'Encoder differs from pinned state')
        with torch.inference_mode():
            features = encoder.encode([r['text'] for r in rows], batch_size=ENCODING['batch_size'],
                precision='float32', device='cpu', normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False, prompt=None, prompt_name=None)
        require(features.shape == (770, ENCODING['dimensions']) and features.dtype == np.float32 and
                np.isfinite(features).all() and np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5),
                'Feature format or normalization changed')
        for seed in original.SEEDS:
            item = protocol['thresholds'][str(seed)]
            require(original._file_sha256(ROOT / item['classifier_path']) == item['classifier_sha256'],
                    'Classifier bytes changed before deserialization')
            classifier = original._load_classifier(ROOT / item['classifier_path'])
            columns = original._validate_classifier(classifier, labels,
                            protocol['specialists']['embedding']['logistic_regression'])
            classifier_before = original._classifier_state(classifier)
            probabilities = classifier.predict_proba(features)[:, columns]
            actual = original._predictions(ids, labels, probabilities, item['value'])
            require(classifier.predict(features).tolist() == [r['predicted_label'] for r in actual],
                    'Class order mismatch')
            relative = f'artifacts/exp004-minilm-learning-curve/runs/exp004-minilm-v2-n20-s{seed}/probabilities.json'
            require(original._file_sha256(ROOT / relative) == protocol['input_files_sha256'][relative],
                    'Archived probabilities changed')
            saved = read_json(ROOT / relative)
            require(saved['validation_ids'] == ids and saved['labels'] == labels, 'Archived alignment mismatch')
            reference = np.asarray(saved['probabilities'], dtype=np.float64)
            expected = original._predictions(ids, labels, reference, item['value'])
            primary_path = f'experiments/exp004-minilm-learning-curve/runs/exp004-minilm-v2-n20-s{seed}.json'
            require(original._file_sha256(ROOT / primary_path) == protocol['input_files_sha256'][primary_path],
                    'Archived primary record changed')
            primary = read_json(ROOT / primary_path)
            require([(r['id'], r['predicted_label']) for r in expected] ==
                    [(r['id'], r['predicted_label']) for r in primary['predictions']], 'Reference prediction drift')
            classifier_after = original._classifier_state(classifier)
            require(classifier_before == classifier_after, 'Classifier state mutated')
            reports[str(seed)] = {
                'classifier_sha256': item['classifier_sha256'],
                'classifier_state_sha256_before': classifier_before,
                'classifier_state_sha256_after': classifier_after,
                'threshold': item['value'], 'reference_probability_sha256': protocol['input_files_sha256'][relative],
                'maximum_absolute_probability_difference': float(np.max(np.abs(probabilities - reference))),
                'maximum_absolute_confidence_difference': max(abs(a['confidence']-b['confidence']) for a,b in zip(actual,expected)),
                'prediction_mismatch_ids': [a['id'] for a,b in zip(actual,expected) if a['predicted_label'] != b['predicted_label']],
                'routing_mismatches': [{'id': a['id'], 'actual_confidence': a['confidence'],
                    'saved_confidence': b['confidence'], 'actual_gate': a['use_specialist'], 'saved_gate': b['use_specialist']}
                    for a,b in zip(actual,expected) if a['use_specialist'] != b['use_specialist']],
                'accepted_count': sum(r['use_specialist'] for r in actual),
                'reference_accepted_count': sum(r['use_specialist'] for r in expected),
                'actual_predictions_sha256': sha256(json_bytes(actual)),
            }
        assert_frozen(encoder)
        state_after = state_digest(encoder)
        require(state_before == state_after and dict(encoder.prompts) == prompts and
                encoder.default_prompt_name is None, 'Frozen encoder or prompt metadata changed')
    require(original.verify_specialist_files(protocol) == checked, 'Pinned files/packages changed during verification')
    require(original._file_sha256(ROOT / 'data/raw/banking77/train.csv') == train_hash, 'Training CSV changed')
    require(not any(GUARDS.values()), 'Forbidden access attempted')
    source_paths = ['baseline/final_specialists.py', 'tests/test_final_specialists.py',
                    'baseline/cpu_benchmark.py', 'baseline/embeddings.py', 'pyproject.toml', 'uv.lock',
                    'experiments/exp007-loader-compatibility/verify_validation.py']
    return {'status': 'patched original loader loaded and frozen-state checks passed', 'verified_utc': now(),
        'base_commit': subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'previous_loader_source_sha256': sha256(subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:baseline/final_specialists.py'])),
        'source_files_sha256': {p: original._file_sha256(ROOT/p) for p in source_paths},
        'protocol_sha256': sha256(json_bytes(protocol)), 'companion_protocol_sha256': sha256(json_bytes(companion)),
        'validation_count': len(rows), 'validation_ids_sha256': sha256(json_bytes(ids)),
        'validation_texts_sha256': sha256(json_bytes(rows)), 'training_csv_sha256': train_hash,
        'observed_prompts': prompts, 'default_prompt_name': encoder.default_prompt_name,
        'encoder_state_sha256_before': state_before, 'encoder_state_sha256_after': state_after,
        'trainable_parameters': sum(p.numel() for p in encoder.parameters() if p.requires_grad),
        'pinned_files_and_packages': checked, 'encoding': ENCODING,
        'threads': {'torch': torch.get_num_threads(), 'interop': torch.get_num_interop_threads(), 'native': 1},
        'seeds': reports, 'guard_counters': dict(GUARDS), 'api_calls': 0, 'official_test_access': False,
        'model_fits': 0, 'timing_study_rerun': False,
        'note': 'Single fresh untimed validation encoding using patched original loader; five unchanged classifiers. Differences retained without threshold adjustment. Historical verification records remain unchanged.'}


if __name__ == '__main__':
    # Caller redirects stdout to a new evidence file; this script never overwrites evidence.
    import json
    print(json.dumps(verify(), indent=2, sort_keys=True, allow_nan=False))
