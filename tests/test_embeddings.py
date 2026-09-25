"""Offline synthetic checks: no downloaded encoder, real-data fits, or test evaluation."""
from copy import deepcopy
import warnings

import joblib
import numpy as np
import pytest
import torch
from sklearn.exceptions import ConvergenceWarning

from baseline import data, embeddings as emb
from baseline.data import json_bytes, read_json, sha256, text_key, write_json
from baseline.experiment import LOGISTIC, classification_metrics


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = []

    @staticmethod
    def vector(text):
        value = np.zeros(384, dtype=np.float32)
        value[ord(text[0]) - ord('a')] = 1
        value[10] = len(text) / 100
        return value / np.linalg.norm(value)

    def encode(self, texts, **kwargs):
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        assert not self.training and not self.weight.requires_grad
        self.calls.append(list(texts))
        return np.stack([self.vector(t) for t in texts])


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    labels = ['a', 'b', 'c']
    def row(label, i):
        text = f'{label} item {i}'
        return {'id': f'train:{label}-{i}', 'label': label, 'text': text, 'key': text_key(text)}
    train = [row(label, i) for label in labels for i in range(5)]
    validation = [row(label, i) for label in labels for i in range(10, 20)]
    unused = [row(label, 100) for label in labels]
    pool = list(reversed(train + unused))  # Input pool order must not become feature order.
    samples = {'labels': labels, 'seed': 11, 'shots': 5,
               'train_ids': [r['id'] for r in train], 'validation_ids': [r['id'] for r in validation]}
    manifest = {'protocol_id': data.PROTOCOL_ID, 'labels': labels,
                'source': {'dataset': 'synthetic'}, 'sealed_test': [],
                'audit': {'original_train': len(pool) + len(validation)}}
    manifest_path = tmp_path / 'manifest.json'
    write_json(manifest_path, manifest)
    truth = [r['label'] for r in validation]
    sample_sha = sha256(json_bytes(samples))
    record = {
        'metadata': {'status': 'completed', 'evaluation_split': 'validation',
                     'run_id': 'synthetic-reference', 'shots': 5, 'seed': 11,
                     'protocol_id': data.PROTOCOL_ID, 'source': manifest['source'],
                     'configuration': {'logistic_regression': LOGISTIC},
                     'split_manifest_sha256': sha256(manifest_path.read_bytes()),
                     'samples_sha256': sample_sha,
                     'label_budgets': {'training': 15, 'validation': 30, 'test_evaluation': 0}},
        'samples': samples,
        'metrics': {'tfidf_logistic_regression': classification_metrics(truth, truth, labels)},
        'predictions': [{'id': r['id'], 'true_label': r['label'], 'predicted_label': r['label']}
                        for r in validation],
    }
    reference = tmp_path / 'reference.json'
    write_json(reference, record)
    encoder = FakeEncoder()
    monkeypatch.setattr(emb, 'REFERENCE', reference)
    monkeypatch.setattr(emb, 'SAMPLE_SHA256', sample_sha)
    monkeypatch.setattr(emb, 'load_development', lambda *args: (manifest, pool, validation))
    monkeypatch.setattr(emb, 'load_encoder', lambda *args: (encoder, {'synthetic': True}))
    monkeypatch.setattr(emb, 'code_provenance', lambda *args: {'synthetic': True, 'source_files_sha256': {}})
    monkeypatch.setattr(emb, 'environment', lambda: {'synthetic': True})
    def no_sampling(*args, **kwargs):
        raise AssertionError('Must reuse sample IDs, not call few_shot or prepare')
    monkeypatch.setattr(data, 'few_shot', no_sampling)
    monkeypatch.setattr(data, 'prepare', no_sampling)
    return tmp_path, manifest_path, reference, record, train, validation, encoder


def test_reuses_exact_reference_ids_despite_reordered_pool(fixture):
    root, manifest, reference, record, train, validation, _ = fixture
    _, _, actual_train, actual_val = emb.reuse_reference(root, manifest, reference)
    assert actual_train == train and actual_val == validation
    assert [r['id'] for r in actual_train] == record['samples']['train_ids']


@pytest.mark.parametrize('change', ['sample', 'legacy', 'split', 'label', 'metric', 'classifier'])
def test_rejects_changed_reference(fixture, change):
    root, manifest, reference, record, *_ = fixture
    record = deepcopy(record)
    if change == 'sample':
        record['samples']['train_ids'].reverse()
    elif change == 'legacy':
        record['metadata']['protocol_id'] = 'banking77-val20-v1'
    elif change == 'split':
        record['metadata']['split_manifest_sha256'] = 'changed'
    elif change == 'label':
        record['predictions'][0]['true_label'] = 'b'
    elif change == 'metric':
        record['metrics']['tfidf_logistic_regression']['macro_f1'] = 0
    else:
        record['metadata']['configuration']['logistic_regression']['C'] = 2
    write_json(reference, record)
    with pytest.raises(ValueError):
        emb.reuse_reference(root, manifest, reference)


def test_encoding_preserves_row_text_label_alignment_and_freezes(fixture, tmp_path):
    *_, train, validation, encoder = fixture
    emb.freeze_encoder(encoder)
    before = emb.state_digest(encoder)
    rows = [validation[-1], train[0], validation[2]]
    features, seconds = emb.encode_rows(encoder, rows)
    x, y = features.aligned(rows)
    np.testing.assert_array_equal(x, np.stack([encoder.vector(r['text']) for r in rows]))
    assert y == [r['label'] for r in rows]
    assert encoder.calls == [[r['text'] for r in rows]]
    assert seconds >= 0 and emb.state_digest(encoder) == before
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    features.save(tmp_path / 'features.npz')
    with np.load(tmp_path / 'features.npz', allow_pickle=False) as saved:
        assert saved['ids'].tolist() == features.ids
        np.testing.assert_array_equal(saved['values'], x)
    for field in ('id', 'label', 'text'):
        corrupted = deepcopy(rows)
        corrupted[0][field] += 'changed'
        with pytest.raises(ValueError, match='aligned'):
            features.aligned(corrupted)
    with pytest.raises(ValueError, match='aligned'):
        features.aligned(list(reversed(rows)))


def test_rejects_unfrozen_or_training_encoder(fixture):
    *_, train, validation, encoder = fixture
    with pytest.raises(ValueError, match='evaluation mode'):
        emb.encode_rows(encoder, train)
    encoder.eval()
    with pytest.raises(ValueError, match='trainable'):
        emb.encode_rows(encoder, train)
    emb.freeze_encoder(encoder)
    encoder.weight.grad = torch.ones_like(encoder.weight)
    with pytest.raises(ValueError, match='gradients'):
        emb.encode_rows(encoder, validation)


def test_synthetic_smoke_artifacts_and_classifier_inputs(fixture, monkeypatch):
    root, manifest, _, record, train, validation, encoder = fixture
    output = root / 'smoke'
    # Capture inputs without changing the serialized classifier's type.
    original_fit = emb.LogisticRegression.fit
    def audited_fit(self, x, y, **kwargs):
        np.testing.assert_array_equal(x, np.stack([encoder.vector(r['text']) for r in train]))
        assert y == [r['label'] for r in train]
        return original_fit(self, x, y, **kwargs)
    monkeypatch.setattr(emb.LogisticRegression, 'fit', audited_fit)
    emb.run(root, manifest, output, root / 'cache')
    meta = read_json(output / 'metadata.json')
    assert meta['status'] == 'completed'
    assert meta['configuration']['logistic_regression'] == LOGISTIC
    assert read_json(output / 'samples.json') == record['samples']
    assert encoder.calls == [[r['text'] for r in train], [r['text'] for r in validation]]
    assert meta['frozen_encoder_check']['state_sha256_before'] == meta['frozen_encoder_check']['state_sha256_after']
    for name, digest in meta['artifacts_sha256'].items():
        assert sha256((output / name).read_bytes()) == digest
    for key in ('training_embedding_seconds', 'validation_embedding_seconds',
                'classifier_fit_seconds', 'classifier_predict_seconds'):
        assert isinstance(meta['resources'][key], float) and meta['resources'][key] >= 0
    predictions = read_json(output / 'predictions.json')
    truth, predicted = [r['true_label'] for r in predictions], [r['predicted_label'] for r in predictions]
    metrics = read_json(output / 'metrics.json')
    assert metrics['evaluation_split'] == 'validation'
    assert metrics[emb.METRIC_KEY] == classification_metrics(truth, predicted, record['samples']['labels'])
    with np.load(output / 'validation_embeddings.npz', allow_pickle=False) as saved:
        classifier = joblib.load(output / 'classifier.joblib')
        assert classifier.predict(saved['values']).tolist() == predicted
    probabilities = read_json(output / 'probabilities.json')
    assert probabilities['validation_ids'] == record['samples']['validation_ids']
    np.testing.assert_allclose(np.sum(probabilities['probabilities'], axis=1), 1)
    comparison = read_json(output / 'comparison.json')
    assert comparison['difference_percentage_points']['macro_f1'] == 100 * (metrics[emb.METRIC_KEY]['macro_f1'] - 1)
    assert emb.verify_run(root, manifest, output)['status'] == 'passed'
    assert len(encoder.calls) == 2  # Verification never encodes or trains again.
    with pytest.raises(FileExistsError):
        emb.run(root, manifest, output, root / 'cache')
    (output / 'predictions.json').write_text('[]')
    with pytest.raises(ValueError, match='checksum mismatch'):
        emb.verify_run(root, manifest, output)


def test_mutating_encoder_fails_and_is_logged(fixture, monkeypatch):
    root, manifest, *_, encoder = fixture
    original = encoder.encode
    def mutate(*args, **kwargs):
        with torch.no_grad():
            encoder.weight.add_(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(encoder, 'encode', mutate)
    with pytest.raises(ValueError, match='state changed'):
        emb.run(root, manifest, root / 'mutating', root / 'cache')
    assert read_json(root / 'mutating/metadata.json')['status'] == 'failed'
    assert not (root / 'mutating/metrics.json').exists()


def test_classifier_nonconvergence_fails_and_is_logged(fixture, monkeypatch):
    root, manifest, *_ = fixture
    def failed_fit(*args):
        warnings.warn('synthetic convergence failure', ConvergenceWarning)
    monkeypatch.setattr(emb.LogisticRegression, 'fit', failed_fit)
    with pytest.raises(RuntimeError, match='did not converge'):
        emb.run(root, manifest, root / 'failed', root / 'cache')
    meta = read_json(root / 'failed/metadata.json')
    assert meta['status'] == 'failed' and 'ConvergenceWarning' in meta['warnings'][0]
    assert not (root / 'failed/metrics.json').exists()
