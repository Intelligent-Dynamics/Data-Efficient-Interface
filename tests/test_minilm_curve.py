"""Synthetic study/cache checks; no real BANKING77 fits or encoder downloads."""
from copy import deepcopy
import statistics
import warnings

import numpy as np
import pytest
import torch
from sklearn.exceptions import ConvergenceWarning

from baseline import embeddings as emb, minilm_curve as curve
from baseline.data import PROTOCOL_ID, json_bytes, read_json, sha256, text_key, write_json
from baseline.experiment import classification_metrics


@pytest.fixture
def references():
    labels = [f'intent_{i:02}' for i in range(77)]
    truth = [label for label in labels for _ in range(10)]
    result = []
    for n, seed in curve.GRID:
        correct = 2 + n // 10 + seed // 11
        pred = [label if i % 10 < correct else labels[(labels.index(label) + 1) % 77]
                for i, label in enumerate(truth)]
        samples = {'shots': n, 'seed': seed, 'labels': labels,
                   'train_ids': [f'train:{seed}:{label}:{j}' for label in labels for j in range(n)],
                   'validation_ids': [f'val:{j}' for j in range(770)]}
        result.append({'metadata': {'status': 'completed', 'evaluation_split': 'validation',
                       'protocol_id': PROTOCOL_ID, 'run_id': f'reference-n{n}-s{seed}', 'shots': n, 'seed': seed,
                       'source': {'synthetic': True}, 'split_manifest_sha256': 'fixed', 'warnings': [],
                       'label_budgets': {'training': n * 77, 'validation': 770, 'test_evaluation': 0}},
                       'samples': samples,
                       'predictions': [{'id': samples['validation_ids'][i], 'true_label': t,
                                        'predicted_label': pred[i], 'dummy_predicted_label': labels[0]}
                                       for i, t in enumerate(truth)],
                       'metrics': {'evaluation_split': 'validation',
                                   curve.TFIDF_MODEL: classification_metrics(truth, pred, labels),
                                   'dummy_most_frequent': classification_metrics(truth, [labels[0]] * 770, labels)}})
    return result


@pytest.fixture
def records(references):
    records = []
    for ref in references:
        record = deepcopy(ref)
        record['metadata'].update(role='primary', configuration=deepcopy(curve.CONFIG),
                                 samples_sha256=sha256(json_bytes(ref['samples'])),
                                 reference_sha256=sha256(json_bytes(ref)),
                                 embedding_source={'model_id': emb.MODEL_ID, 'revision': emb.MODEL_REVISION},
                                 frozen_encoder_check={'trainable_parameters': 0, 'gradients_present': False,
                                                       'evaluation_mode': True, 'inference_mode': True})
        # Improve one case/class for every seed, leaving nontrivial seed variance.
        for i, row in enumerate(record['predictions']):
            if i % 10 == 9:
                row['predicted_label'] = row['true_label']
            del row['dummy_predicted_label']
        truth = [p['true_label'] for p in record['predictions']]
        pred = [p['predicted_label'] for p in record['predictions']]
        record['metrics'] = {'evaluation_split': 'validation',
                            emb.METRIC_KEY: classification_metrics(truth, pred, record['samples']['labels'])}
        records.append(record)
    return records


def test_all_seeds_sample_sd_and_matching_seed_improvements(records, references):
    summary = curve.aggregate(records, references)
    assert summary['primary_run_count'] == 15
    assert summary['additional_validation_labels'] == 770
    for regime in summary['regimes']:
        for metric in ('accuracy', 'macro_f1'):
            values = regime['minilm'][metric]['by_seed']
            assert set(values) == {str(s) for s in curve.SEEDS}
            assert regime['minilm'][metric]['sample_std'] == statistics.stdev(values.values())
            assert regime['minilm'][metric]['mean'] == statistics.mean(values.values())
        for seed in curve.SEEDS:
            gain = regime['macro_f1_improvement_pp']['by_seed'][str(seed)]
            assert gain == pytest.approx(10)
    assert curve.aggregate(list(reversed(records)), list(reversed(references))) == summary


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'samples', 'config', 'revision',
                                    'metric', 'labels', 'role', 'unfrozen'])
def test_aggregate_rejects_incomplete_or_incomparable_records(records, references, change):
    r = records[-1]
    if change == 'missing':
        records.pop()
    elif change == 'duplicate':
        records[-1] = records[0]
    elif change == 'samples':
        r['samples']['train_ids'].reverse()
    elif change == 'config':
        r['metadata']['configuration']['logistic_regression']['C'] = 2
    elif change == 'revision':
        r['metadata']['embedding_source']['revision'] = 'wrong'
    elif change == 'metric':
        r['metrics'][emb.METRIC_KEY]['macro_f1'] += .001
    elif change == 'labels':
        r['predictions'][0]['true_label'] = 'wrong'
    elif change == 'role':
        r['metadata']['role'] = 'reproduction'
    else:
        r['metadata']['frozen_encoder_check']['trainable_parameters'] = 1
    with pytest.raises(ValueError):
        curve.aggregate(records, references)


@pytest.fixture
def cache_fixture(tmp_path, monkeypatch):
    labels = ['a', 'b', 'c']
    def rows(start, count):
        return [{'id': f'train:{c}-{i}', 'label': c, 'text': f'{c} example {i}',
                 'key': text_key(f'{c} example {i}')}
                for c in labels for i in range(start, start + count)]
    train, val, rest = rows(0, 5), rows(10, 10), rows(100, 1)
    groups = [('train', train), ('validation', val), ('remaining_training_union', rest)]
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = []
        def encode(self, texts, **kwargs):
            assert torch.is_inference_mode_enabled() and not self.training
            assert not self.weight.requires_grad
            self.calls.append(texts)
            values = np.zeros((len(texts), 384), dtype=np.float32)
            for i, t in enumerate(texts):
                values[i, ord(t[0]) - ord('a')] = 1
            return values
    encoder = Encoder()
    source = {'model_id': emb.MODEL_ID, 'revision': emb.MODEL_REVISION, 'files_sha256': {'model': 'fake'}}
    environment = {'packages': curve.cache_signature()['packages']}
    smoke = {'configuration': curve.CONFIG, 'embedding_source': source,
             'environment': environment, 'frozen_encoder_check': {'state_sha256_before': emb.state_digest(encoder)}}
    write_json(tmp_path / 'metadata.json', smoke)
    monkeypatch.setattr(curve, 'SMOKE_RECORD', tmp_path)
    monkeypatch.setattr(emb, 'environment', lambda: deepcopy(environment))
    monkeypatch.setattr(emb, 'load_encoder', lambda _: (encoder, deepcopy(source)))
    monkeypatch.setattr(curve, 'code_provenance', lambda _: {'source_files_sha256': {}})
    monkeypatch.setattr(emb, 'verify_run', lambda *args: {'status': 'passed'})
    emb.freeze_encoder(encoder)
    for name, group in groups[:2]:
        features, _ = emb.encode_rows(encoder, group)
        features.save(tmp_path / f'{name}_embeddings.npz')
    encoder.calls.clear()
    return tmp_path, groups, encoder


def test_cache_reuses_smoke_and_reencodes_independently(cache_fixture):
    root, groups, encoder = cache_fixture
    all_rows = [r for _, rows in groups for r in rows]
    curve.build_cache(root / 'primary', groups, reuse_smoke=root)
    meta, cached = curve.load_cache(root / 'primary', all_rows)
    assert encoder.calls == [[r['text'] for r in groups[2][1]]]
    assert meta['groups'][0]['cache_hit'] and meta['groups'][0]['encoding_seconds'] is None
    encoder.calls.clear()
    curve.build_cache(root / 'independent', groups)
    _, recomputed = curve.load_cache(root / 'independent', all_rows)
    assert encoder.calls == [[r['text'] for r in rows] for _, rows in groups]
    np.testing.assert_array_equal(cached.values, recomputed.values)
    shuffled = list(reversed(groups[0][1]))
    selected = curve.select_features(cached, shuffled)
    assert selected.ids == [r['id'] for r in shuffled]
    for i, row in enumerate(shuffled):
        assert selected.values[i, ord(row['label']) - ord('a')] == 1
    changed = deepcopy(shuffled)
    changed[0]['text'] = 'other text'
    with pytest.raises(ValueError, match='aligned'):
        curve.select_features(cached, changed)
    with pytest.raises(FileExistsError):
        curve.build_cache(root / 'primary', groups)


@pytest.mark.parametrize('change', ['runtime', 'revision', 'bytes', 'rows', 'state'])
def test_cache_invalidations(cache_fixture, change):
    root, groups, _ = cache_fixture
    rows = [r for _, group in groups for r in group]
    path = root / 'cache'
    curve.build_cache(path, groups)
    meta = read_json(path / 'metadata.json')
    if change == 'runtime':
        meta['signature']['packages']['torch'] = 'wrong'
    elif change == 'revision':
        meta['signature']['revision'] = 'wrong'
    elif change == 'bytes':
        with (path / 'features.npz').open('ab') as f:
            f.write(b'changed')
    elif change == 'rows':
        rows = list(reversed(rows))
    else:
        meta['state_sha256_after'] = 'wrong'
    write_json(path / 'metadata.json', meta)
    with pytest.raises(ValueError):
        curve.load_cache(path, rows)


def test_fit_uses_only_selected_training_rows_and_replays(cache_fixture, monkeypatch):
    root, groups, _ = cache_fixture
    train, val = groups[0][1], groups[1][1]
    all_rows = [r for _, rows in groups for r in rows]
    path = root / 'cache'
    curve.build_cache(path, groups)
    cache_meta, features = curve.load_cache(path, all_rows)
    samples = {'shots': 5, 'seed': 11, 'labels': ['a', 'b', 'c'],
               'train_ids': [r['id'] for r in train], 'validation_ids': [r['id'] for r in val]}
    reference = {'metadata': {'run_id': 'synthetic', 'source': {}, 'split_manifest_sha256': 'fake',
                             'label_budgets': {'training': 15, 'validation': 30, 'test_evaluation': 0}},
                 'samples': samples, 'predictions': [{'id': r['id'], 'true_label': r['label']} for r in val]}
    original = curve.LogisticRegression.fit
    def audited(self, x, y):
        expected = curve.select_features(features, train)
        np.testing.assert_array_equal(x, expected.values)
        assert y == expected.labels and len(y) == 15
        return original(self, x, y)
    monkeypatch.setattr(curve.LogisticRegression, 'fit', audited)
    output = root / 'fit'
    curve.fit_run(output, reference, train, val, path, cache_meta, features, 'primary')
    record, _ = curve.verify_run(output, reference, train, val, path, cache_meta, features)
    assert record['metadata']['resources']['end_to_end_inference_seconds'] is None
    assert record['metadata']['resources']['embedding_cache_used']
    assert read_json(output / 'samples.json') == samples
    (output / 'probabilities.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        curve.verify_run(output, reference, train, val, path, cache_meta, features)
    def failed(*args):
        warnings.warn('synthetic nonconvergence', ConvergenceWarning)
    monkeypatch.setattr(curve.LogisticRegression, 'fit', failed)
    with pytest.raises(RuntimeError, match='converge'):
        curve.fit_run(root / 'failed', reference, train, val, path, cache_meta, features, 'primary')
    assert read_json(root / 'failed/metadata.json')['status'] == 'failed'
    assert not (root / 'failed/metrics.json').exists()
