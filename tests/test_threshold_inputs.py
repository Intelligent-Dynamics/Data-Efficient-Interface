"""Synthetic archived-record fixtures for EXP-007's read-only input boundary."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from baseline.data import json_bytes, sha256
from baseline.experiment import LOGISTIC, classification_metrics
from baseline.selective import MODEL, compact, selective_curve
from baseline.threshold_inputs import (ARTIFACTS, DATASET_REVISION, ENCODING, EXP004,
                                       EXP005, FROZEN, MODEL_ID, MODEL_REVISION, SEEDS,
                                       load_inputs)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))
    return sha256(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def update_audit(root, seed):
    name = f'runs/exp004-minilm-v2-n20-s{seed}.json'
    audit = read(root / EXP004 / 'analysis_audit.json')
    audit['compact_records_sha256'][name] = sha256((root / EXP004 / name).read_bytes())
    save(root / EXP004 / 'analysis_audit.json', audit)


def update_diagnostic_audit(root, seed):
    name = f'runs/exp004-minilm-v2-n20-s{seed}.json'
    audit = read(root / EXP005 / 'metadata.json')
    audit['analysis_artifact_sha256'][name] = sha256((root / EXP005 / name).read_bytes())
    save(root / EXP005 / 'metadata.json', audit)


@pytest.fixture(scope='module')
def archived_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp('SYNTHETIC-threshold-inputs')
    labels = [f'SYNTHETIC_intent_{index:02}' for index in range(77)]
    ids = [f'train:{index:05}' for index in range(770)]
    predictions = [{'id': rid, 'true_label': labels[index // 10], 'predicted_label': labels[index // 10]}
                   for index, rid in enumerate(ids)]
    dataset = {'dataset': 'BANKING77', 'revision': DATASET_REVISION, 'repository': 'SYNTHETIC',
               'license': 'SYNTHETIC', 'attribution': 'SYNTHETIC',
               'expected_counts': {'train': 10003, 'test': 3080, 'classes': 77},
               'files': {'train.csv': {'path': 'banking_data/train.csv', 'sha256': 'd' * 64},
                         'test.csv': {'path': 'banking_data/test.csv', 'sha256': 'e' * 64}}}
    dataset_hash = save(root / 'data/banking77-source.json', dataset)
    source = deepcopy(dataset)
    for value in source['files'].values():
        value['url'] = 'https://example.invalid/SYNTHETIC'
    packages = {'numpy': 'SYNTHETIC', 'scikit-learn': 'SYNTHETIC', 'torch': 'SYNTHETIC'}
    environment = {'packages': packages, 'python': 'SYNTHETIC', 'hardware': 'SYNTHETIC'}
    model_source = {'model_id': MODEL_ID, 'revision': MODEL_REVISION,
                    'files_sha256': {'model.safetensors': 'a' * 64}}
    cache = {'status': 'completed', 'warnings': [], 'state_sha256_before': 'b' * 64,
             'state_sha256_after': 'b' * 64, 'frozen_encoder_check': FROZEN,
             'signature': {'model_id': MODEL_ID, 'revision': MODEL_REVISION,
                           'encoding': ENCODING, 'packages': packages},
             'environment': environment, 'embedding_source': model_source,
             'features_sha256': 'c' * 64}
    cache_hash = save(root / ARTIFACTS / 'primary-cache/metadata.json', cache)
    save(root / EXP004 / 'primary-cache.json', cache)
    audit = {'status': 'passed', 'compact_records_sha256': {}}
    diagnostic_audit = {'study_id': 'EXP-005', 'status': 'completed', 'run_count': 15,
                        'official_test_access': False, 'probabilities_regenerated': False,
                        'model_fits': 0, 'calibration_fits': 0, 'encoder_executions': 0,
                        'analysis_artifact_sha256': {}}
    for seed in SEEDS:
        name = f'exp004-minilm-v2-n20-s{seed}'
        relative = f'runs/{name}.json'
        artifact = root / ARTIFACTS / 'runs' / name
        samples = {'labels': labels, 'validation_ids': ids, 'train_ids': [f'train:{index:05}' for index in range(1000, 2540)],
                   'shots': 20, 'seed': seed}
        matrix = []
        for index in range(770):
            values = [0.5 / 76] * 77
            values[index // 10] = .5
            matrix.append(values)
        probabilities = {'labels': labels, 'validation_ids': ids, 'probabilities': matrix}
        probability_hash = save(artifact / 'probabilities.json', probabilities)
        classifier = b'SYNTHETIC CLASSIFIER BYTES, NOT AN EXECUTABLE PICKLE'
        (artifact / 'classifier.joblib').write_bytes(classifier)
        classifier_hash = sha256(classifier)
        metadata = {
            'run_id': name, 'role': 'primary', 'status': 'completed', 'warnings': [],
            'shots': 20, 'seed': seed, 'protocol_id': 'banking77-val10-v2', 'evaluation_split': 'validation',
            'samples_sha256': sha256(json_bytes(samples)), 'environment': environment,
            'split_manifest_sha256': 'f' * 64, 'code': {'git_head': 'SYNTHETIC',
                'source_files_sha256': {'data/banking77-source.json': dataset_hash}},
            'source': source, 'configuration': {'embedding': ENCODING, 'logistic_regression': LOGISTIC,
                'classifier_changes_from_tfidf': [], 'multiclass_loss': 'multinomial'},
            'frozen_encoder_check': FROZEN, 'embedding_source': model_source,
            'embedding_cache': {'path': (ARTIFACTS / 'primary-cache').as_posix(),
                'metadata_sha256': cache_hash, 'features_sha256': cache['features_sha256']},
            'label_budgets': {'training': 1540, 'validation': 770, 'calibration': 0,
                'prompt_examples': 0, 'test_evaluation': 0},
            'artifacts_sha256': {'probabilities.json': probability_hash, 'classifier.joblib': classifier_hash},
        }
        metrics = classification_metrics([p['true_label'] for p in predictions],
                                         [p['predicted_label'] for p in predictions], labels)
        record = {'metadata': metadata, 'samples': samples, 'predictions': predictions,
                  'metrics': {'evaluation_split': 'validation', MODEL: metrics}}
        record_hash = save(root / EXP004 / relative, record)
        audit['compact_records_sha256'][relative] = record_hash
        diagnostic = {'run_id': name, 'seed': seed, 'shots': 20,
                      **compact(selective_curve([{**row, 'confidence': .5} for row in predictions], labels, 1.0)),
                      'provenance': {'exp004_record_sha256': record_hash, 'probabilities_sha256': probability_hash,
                         'classifier_sha256': classifier_hash, 'samples_sha256': metadata['samples_sha256'],
                         'split_manifest_sha256': metadata['split_manifest_sha256'],
                         'cache_metadata_sha256': cache_hash, 'cache_features_sha256': cache['features_sha256'],
                         'dataset_revision': DATASET_REVISION, 'encoder_revision': MODEL_REVISION,
                         'exp004_code': 'SYNTHETIC', 'probabilities_regenerated': False}}
        diagnostic_audit['analysis_artifact_sha256'][relative] = save(root / EXP005 / relative, diagnostic)
    save(root / EXP004 / 'analysis_audit.json', audit)
    save(root / EXP005 / 'metadata.json', diagnostic_audit)
    return root


@pytest.fixture
def repository(archived_fixture, tmp_path):
    root = tmp_path / 'synthetic-repository'
    shutil.copytree(archived_fixture, root)
    return root


def test_reads_exact_five_runs_preserves_every_file_and_returns_provenance(archived_fixture):
    before = {p.relative_to(archived_fixture).as_posix(): p.read_bytes() for p in archived_fixture.rglob('*') if p.is_file()}
    result = load_inputs(archived_fixture)
    assert [run['seed'] for run in result['runs']] == list(SEEDS)
    assert len(result['labels']) == 77 and len(result['validation_ids']) == 770
    assert len(result['source_files_sha256']) == 25
    assert result['embedding']['settings'] == ENCODING
    assert result['embedding']['revision'] == MODEL_REVISION
    for run in result['runs']:
        assert run['record']['samples']['validation_ids'] == result['validation_ids']
        assert run['selective']['n_validation'] == 770
        assert run['classifier_sha256'] == sha256((archived_fixture / run['classifier_path']).read_bytes())
    after = {p.relative_to(archived_fixture).as_posix(): p.read_bytes() for p in archived_fixture.rglob('*') if p.is_file()}
    assert before == after
    assert result['source_files_sha256'] == {name: sha256(contents) for name, contents in sorted(before.items())}


def test_no_raw_test_processed_features_models_network_or_deserialization(archived_fixture, monkeypatch):
    import joblib
    import pickle
    import socket
    import baseline.data as data
    allowed = {p.resolve() for p in archived_fixture.rglob('*') if p.is_file()}
    old_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path.resolve() in allowed, f'Forbidden read: {path}'
        return old_open(path, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('Forbidden raw dataset/model/network operation')
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(data, 'load_development', forbidden)
    monkeypatch.setattr(data, 'csv_rows', forbidden)
    monkeypatch.setattr(joblib, 'load', forbidden)
    monkeypatch.setattr(pickle, 'loads', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    result = load_inputs(archived_fixture)
    assert all('data/raw/' not in p and 'data/processed/' not in p and not p.endswith('.npz')
               for p in result['source_files_sha256'])


@pytest.mark.parametrize('relative', [
    EXP004 / 'runs/exp004-minilm-v2-n20-s11.json',
    EXP005 / 'runs/exp004-minilm-v2-n20-s11.json',
    ARTIFACTS / 'runs/exp004-minilm-v2-n20-s11/probabilities.json',
    ARTIFACTS / 'runs/exp004-minilm-v2-n20-s11/classifier.joblib',
])
def test_missing_or_byte_modified_bound_evidence_is_rejected(repository, relative):
    path = repository / relative
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='checksum'):
        load_inputs(repository)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        load_inputs(repository)


@pytest.mark.parametrize('change', ['wrong_role', 'wrong_shots', 'test_evaluation', 'train_overlap',
    'test_id', 'duplicate_id', 'validation_order', 'truth', 'settings', 'frozen', 'runtime',
    'split', 'source_metadata', 'source_hash', 'cache_hash', 'label_budget'])
def test_semantically_invalid_record_rejected_even_after_updating_record_hash(repository, change):
    seed = 22
    path = repository / EXP004 / f'runs/exp004-minilm-v2-n20-s{seed}.json'
    record = read(path)
    m, s = record['metadata'], record['samples']
    if change == 'wrong_role': m['role'] = 'reproduction'
    elif change == 'wrong_shots': m['shots'] = s['shots'] = 10
    elif change == 'test_evaluation': m['evaluation_split'] = 'test'
    elif change == 'train_overlap': s['train_ids'][0] = s['validation_ids'][0]
    elif change == 'test_id': s['validation_ids'][0] = 'test:00000'
    elif change == 'duplicate_id': s['validation_ids'][0] = s['validation_ids'][1]
    elif change == 'validation_order': s['validation_ids'].reverse()
    elif change == 'truth':
        record['predictions'][0]['true_label'], record['predictions'][10]['true_label'] = (
            record['predictions'][10]['true_label'], record['predictions'][0]['true_label'])
    elif change == 'settings': m['configuration']['logistic_regression']['C'] = 99
    elif change == 'frozen': m['frozen_encoder_check']['trainable_parameters'] = 1
    elif change == 'runtime': m['environment']['packages']['numpy'] = 'changed'
    elif change == 'split': m['split_manifest_sha256'] = '0' * 64
    elif change == 'source_metadata': m['source']['files']['test.csv']['sha256'] = '0' * 64
    elif change == 'source_hash': m['code']['source_files_sha256']['data/banking77-source.json'] = '0' * 64
    elif change == 'cache_hash': m['embedding_cache']['features_sha256'] = '0' * 64
    else: m['label_budgets']['test_evaluation'] = 1
    m['samples_sha256'] = sha256(json_bytes(s))
    save(path, record)
    update_audit(repository, seed)
    with pytest.raises(ValueError):
        load_inputs(repository)


@pytest.mark.parametrize('change', ['rows', 'labels', 'argmax', 'nan'])
def test_probabilities_must_align_with_the_original_ids_labels_and_predictions(repository, change):
    artifact = repository / ARTIFACTS / 'runs/exp004-minilm-v2-n20-s11'
    p = artifact / 'probabilities.json'
    probability = read(p)
    if change == 'rows': probability['validation_ids'].reverse()
    elif change == 'labels': probability['labels'].reverse()
    elif change == 'argmax':
        probability['probabilities'][0][0], probability['probabilities'][0][1] = probability['probabilities'][0][1], probability['probabilities'][0][0]
    else: probability['probabilities'][0][0] = float('nan')
    # NaN deliberately exercises the input rejection boundary; it is not valid evidence.
    p.write_text(json.dumps(probability))
    r = repository / EXP004 / 'runs/exp004-minilm-v2-n20-s11.json'
    record = read(r)
    record['metadata']['artifacts_sha256']['probabilities.json'] = sha256(p.read_bytes())
    save(r, record)
    update_audit(repository, 11)
    with pytest.raises(ValueError):
        load_inputs(repository)


@pytest.mark.parametrize('change', ['confidence', 'provenance', 'source_model'])
def test_exp005_confidence_and_source_links_cannot_drift(repository, change):
    p = repository / EXP005 / 'runs/exp004-minilm-v2-n20-s11.json'
    record = read(p)
    if change == 'confidence':
        rows = record['ranked_rows']
        for row in rows: row['confidence'] = .6
        record.update(compact(selective_curve(rows, record['labels'], 1.0)))
    elif change == 'provenance': record['provenance']['probabilities_sha256'] = '0' * 64
    else: record['provenance']['encoder_revision'] = '0' * 40
    save(p, record)
    update_diagnostic_audit(repository, 11)
    with pytest.raises(ValueError):
        load_inputs(repository)


def test_symlink_cannot_redirect_an_allowed_artifact_into_forbidden_data(repository):
    path = repository / ARTIFACTS / 'runs/exp004-minilm-v2-n20-s11/probabilities.json'
    hidden = repository / 'data/raw/test.csv'
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(hidden)
    with pytest.raises(ValueError, match='Symlink'):
        load_inputs(repository)
