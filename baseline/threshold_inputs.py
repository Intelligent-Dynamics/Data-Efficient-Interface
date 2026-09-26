"""Read-only EXP-007 inputs from the five archived 20-shot primary specialists.

Only audited EXP-004/005 records, saved probability JSON, classifier bytes, cache
metadata and dataset *metadata* are read. No raw split, processed manifest, model
execution/deserialization, feature array, API response, or network is needed.
"""
from collections import Counter
import json
from pathlib import Path
import re

from .data import ROOT, json_bytes, sha256
from .experiment import LOGISTIC
from .selective import MODEL, probability_rows, require, restore

SEEDS = (11, 22, 33, 44, 55)
EXP004 = Path('experiments/exp004-minilm-learning-curve')
EXP005 = Path('experiments/exp005-selective-diagnostic')
ARTIFACTS = Path('artifacts/exp004-minilm-learning-curve')
MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
DATASET_REVISION = '57ec275d8078af65b7731c2a98be812d844a6d6b'
ENCODING = {
    'device': 'cpu', 'backend': 'torch', 'precision': 'float32', 'dimensions': 384,
    'batch_size': 32, 'max_seq_length': 256, 'normalize_embeddings': True,
    'pooling': 'attention-mask mean pooling', 'prompt': None,
    'attention_implementation': 'eager', 'compute_threads': 1,
    'trust_remote_code': False, 'use_safetensors': True,
}
FROZEN = {'deterministic_algorithms': True, 'evaluation_mode': True,
          'gradients_present': False, 'inference_mode': True,
          'torch_threads': 1, 'trainable_parameters': 0}


def load_inputs(root=ROOT):
    """Return five audited run records and a hash map binding every file read."""
    root = Path(root).resolve()
    hashes = {}

    def read(relative, expected=None, *, binary=False):
        relative = Path(relative)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Input path must remain inside the repository')
        path = root / relative
        # Metadata cannot redirect an allowed read into a raw/test file or another checkout.
        require(not any(p.is_symlink() for p in (path, *path.parents) if p != root.parent),
                'Symlink input paths are prohibited')
        contents = path.read_bytes()
        digest = sha256(contents)
        require(expected is None or digest == expected, f'Input checksum mismatch: {relative}')
        hashes[relative.as_posix()] = digest
        return contents if binary else json.loads(contents)

    dataset_path = Path('data/banking77-source.json')
    dataset = read(dataset_path)
    require(dataset['dataset'] == 'BANKING77' and dataset['revision'] == DATASET_REVISION and
            dataset['expected_counts'] == {'train': 10003, 'test': 3080, 'classes': 77},
            'Pinned dataset metadata changed')
    audit = read(EXP004 / 'analysis_audit.json')
    require(audit['status'] == 'passed', 'EXP-004 audit is not passed')
    diagnostic_audit = read(EXP005 / 'metadata.json')
    require(diagnostic_audit['study_id'] == 'EXP-005' and diagnostic_audit['status'] == 'completed' and
            diagnostic_audit['run_count'] == 15 and diagnostic_audit['official_test_access'] is False and
            diagnostic_audit['probabilities_regenerated'] is False and
            all(diagnostic_audit[key] == 0 for key in ('model_fits', 'calibration_fits', 'encoder_executions')),
            'EXP-005 provenance/status changed')
    cache_path = ARTIFACTS / 'primary-cache/metadata.json'
    cache = read(cache_path)
    versioned_cache = read(EXP004 / 'primary-cache.json')
    require(cache == versioned_cache, 'Archived/versioned cache metadata differ')
    require(cache['status'] == 'completed' and not cache['warnings'] and
            cache['state_sha256_before'] == cache['state_sha256_after'] and
            re.fullmatch('[0-9a-f]{64}', cache['state_sha256_before']) and
            cache['frozen_encoder_check'] == FROZEN, 'Cache encoder freeze evidence changed')
    signature = cache['signature']
    require(signature['model_id'] == MODEL_ID and signature['revision'] == MODEL_REVISION and
            signature['encoding'] == ENCODING, 'Frozen MiniLM cache settings changed')
    require(all(cache['environment']['packages'][k] == v for k, v in signature['packages'].items()),
            'Cache runtime package evidence differs')
    fixed_ids = fixed_labels = fixed_truth = fixed_environment = fixed_split = None
    runs = []
    for seed in SEEDS:
        name = f'exp004-minilm-v2-n20-s{seed}'
        relative = f'runs/{name}.json'
        record_path = EXP004 / relative
        record = read(record_path, audit['compact_records_sha256'][relative])
        metadata, samples = record['metadata'], record['samples']
        require(metadata['run_id'] == name and metadata['role'] == 'primary' and
                metadata['status'] == 'completed' and not metadata['warnings'] and
                metadata['shots'] == samples['shots'] == 20 and metadata['seed'] == samples['seed'] == seed,
                'Only the five complete 20-shot primary runs are eligible')
        require(metadata['protocol_id'] == 'banking77-val10-v2' and
                metadata['evaluation_split'] == record['metrics']['evaluation_split'] == 'validation',
                'Validation-only v2 protocol required')
        require(metadata['samples_sha256'] == sha256(json_bytes(samples)), 'Sample manifest checksum changed')
        labels, ids, training_ids = samples['labels'], samples['validation_ids'], samples['train_ids']
        require(len(labels) == len(set(labels)) == 77 and all(isinstance(x, str) and x for x in labels),
                'Exactly 77 unique class names required')
        require(len(ids) == len(set(ids)) == 770 and len(training_ids) == len(set(training_ids)) == 1540 and
                set(ids).isdisjoint(training_ids) and all(re.fullmatch(r'train:\d{5}', x) for x in ids + training_ids),
                'Training/validation IDs, size, or isolation changed')
        truth = {row['id']: row['true_label'] for row in record['predictions']}
        require(Counter(truth.values()) == {label: 10 for label in labels}, 'Expected ten validation rows per intent')
        if fixed_ids is None:
            fixed_ids, fixed_labels, fixed_truth = ids, labels, truth
            fixed_environment, fixed_split = metadata['environment'], metadata['split_manifest_sha256']
        require(ids == fixed_ids and labels == fixed_labels and truth == fixed_truth,
                'Five specialists must reuse identical validation IDs, labels, and truths')
        require(metadata['environment'] == fixed_environment and
                metadata['environment']['packages'] == cache['environment']['packages'] and
                metadata['split_manifest_sha256'] == fixed_split,
                'Specialist runtime or split provenance changed')
        require(metadata['code']['source_files_sha256'][dataset_path.as_posix()] == hashes[dataset_path.as_posix()],
                'Dataset metadata hash differs from EXP-004 source snapshot')
        source = metadata['source']
        for field in ('dataset', 'repository', 'revision', 'license', 'attribution', 'expected_counts'):
            require(source[field] == dataset[field], f'Dataset metadata {field} differs')
        require(set(source['files']) == set(dataset['files']) and all(
            all(source['files'][name][key] == value for key, value in values.items())
            for name, values in dataset['files'].items()), 'Dataset file metadata differs')
        require(metadata['configuration'] == {'embedding': ENCODING, 'logistic_regression': LOGISTIC,
                'classifier_changes_from_tfidf': [], 'multiclass_loss': 'multinomial'},
                'Frozen embedding or logistic regression settings changed')
        require(metadata['frozen_encoder_check'] == FROZEN and
                metadata['embedding_source'] == cache['embedding_source'] and
                metadata['embedding_source']['model_id'] == MODEL_ID and
                metadata['embedding_source']['revision'] == MODEL_REVISION,
                'Frozen encoder source/check changed')
        require(metadata['embedding_cache']['path'] == (ARTIFACTS / 'primary-cache').as_posix() and
                metadata['embedding_cache']['metadata_sha256'] == hashes[cache_path.as_posix()] and
                metadata['embedding_cache']['features_sha256'] == cache['features_sha256'],
                'Embedding cache provenance changed')
        budget = metadata['label_budgets']
        require(budget['training'] == 1540 and budget['validation'] == 770 and
                all(budget[k] == 0 for k in ('calibration', 'prompt_examples', 'test_evaluation')),
                'Archived label budget changed')
        artifact_path = ARTIFACTS / 'runs' / name
        probability_path = artifact_path / 'probabilities.json'
        probabilities = read(probability_path, metadata['artifacts_sha256']['probabilities.json'])
        classifier_path = artifact_path / 'classifier.joblib'
        read(classifier_path, metadata['artifacts_sha256']['classifier.joblib'], binary=True)
        rows = probability_rows(record, probabilities)
        selective_path = EXP005 / relative
        selective = restore(read(selective_path, diagnostic_audit['analysis_artifact_sha256'][relative]))
        require(selective['run_id'] == name and selective['shots'] == 20 and selective['seed'] == seed and
                selective['labels'] == labels and selective['n_validation'] == 770 and
                selective['full_accuracy'] == record['metrics'][MODEL]['accuracy'], 'EXP-005 run identity changed')
        require({row['id']: row for row in selective['ranked_rows']} == {row['id']: row for row in rows},
                'EXP-005 confidence/prediction/label alignment differs from saved probabilities')
        expected = {
            'exp004_record_sha256': hashes[record_path.as_posix()],
            'probabilities_sha256': hashes[probability_path.as_posix()],
            'classifier_sha256': hashes[classifier_path.as_posix()],
            'samples_sha256': metadata['samples_sha256'],
            'split_manifest_sha256': metadata['split_manifest_sha256'],
            'cache_metadata_sha256': hashes[cache_path.as_posix()],
            'cache_features_sha256': cache['features_sha256'],
            'dataset_revision': DATASET_REVISION, 'encoder_revision': MODEL_REVISION,
            'exp004_code': metadata['code']['git_head'], 'probabilities_regenerated': False,
        }
        require(all(selective['provenance'][k] == value for k, value in expected.items()),
                'EXP-005 source provenance differs')
        runs.append({'seed': seed, 'run_id': name, 'record': record, 'selective': selective,
                     'classifier_path': classifier_path.as_posix(),
                     'classifier_sha256': hashes[classifier_path.as_posix()],
                     'probabilities_sha256': hashes[probability_path.as_posix()]})
    return {'runs': runs, 'source_files_sha256': dict(sorted(hashes.items())), 'dataset': dataset,
            'labels': fixed_labels, 'validation_ids': fixed_ids,
            'embedding': {'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'settings': ENCODING,
                          'logistic_regression': LOGISTIC, 'multiclass_loss': 'multinomial',
                          'model_files_sha256': cache['embedding_source']['files_sha256'],
                          'recorded_packages': cache['environment']['packages'],
                          'cache_metadata_sha256': hashes[cache_path.as_posix()],
                          'cache_features_sha256': cache['features_sha256'],
                          'encoder_state_sha256': cache['state_sha256_before']}}
