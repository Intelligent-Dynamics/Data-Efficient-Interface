"""EXP-007 synthetic specialist checks: no official dataset or real encoder access."""
from collections import UserDict
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from baseline import final_specialists as final
from baseline.data import json_bytes, read_json, sha256, write_json
from baseline.embeddings import ENCODING, MODEL_ID, MODEL_REVISION, state_digest
from baseline.experiment import LOGISTIC


class SyntheticEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = []
        self.max_seq_length = 256
        self.prompts = {}
        self.default_prompt_name = None
        self.mutate = False

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **settings):
        assert torch.is_inference_mode_enabled()
        assert not self.training and not self.weight.requires_grad
        assert settings == {'batch_size': 32, 'precision': 'float32', 'device': 'cpu',
                            'normalize_embeddings': True, 'convert_to_numpy': True,
                            'show_progress_bar': False, 'prompt': None, 'prompt_name': None}
        self.calls.append(list(texts))
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        for index, text in enumerate(texts):
            matrix[index, int(text)] = 1.0
        if self.mutate:
            self.weight.add_(1)
        return matrix


class SyntheticClassifier(LogisticRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classes_ = np.array(['a', 'm', 'z'])
        self.n_features_in_ = 384
        self.coef_ = np.zeros((3, 384))
        self.intercept_ = np.zeros(3)
        self.mutate = False
        self.calls = []

    def get_params(self, deep=True):
        # Match the exact existing LR configuration without fitting synthetic data.
        return {name: getattr(self, name) for name in LogisticRegression().get_params()}

    def fit(self, *_args, **_kwargs):
        raise AssertionError('EXP-007 must never fit')

    def predict_proba(self, values):
        self.calls.append(values.copy())
        if self.mutate:
            self.coef_[0, 0] += 1
        table = np.array([[.2, .2, .6], [.9, .05, .05], [.3, .25, .45], [.5, .0, .5]],
                         dtype=np.float64)
        return table[values.argmax(axis=1)]

    def predict(self, values):
        return self.classes_[self.predict_proba(values).argmax(axis=1)]


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    encoder = SyntheticEncoder()
    classifier = SyntheticClassifier(**LOGISTIC)
    protocol = {
        'population': {'labels': ['z', 'a', 'm'], 'expected_count_from_existing_metadata': 4},
        'specialists': {'seeds': final.SEEDS, 'shots': 20, 'refit': False, 'recalibrate': False,
                        'tune': False, 'embedding': {
                            'model_id': MODEL_ID, 'revision': MODEL_REVISION,
                            'settings': deepcopy(ENCODING), 'logistic_regression': deepcopy(LOGISTIC),
                            'model_files_sha256': {}, 'recorded_packages': {'synthetic-library': '1.0'},
                            'encoder_state_sha256': state_digest(encoder)}},
        'thresholds': {}, 'input_files_sha256': {},
    }
    snapshot = final._snapshot_path(protocol, tmp_path)
    snapshot.mkdir(parents=True)
    for filename in ('model.safetensors', 'config.json'):
        (snapshot / filename).write_text('synthetic fixture, not model weights')
        protocol['specialists']['embedding']['model_files_sha256'][filename] = final._file_sha256(snapshot / filename)
    for seed in final.SEEDS:
        relative = f'artifacts/synthetic-s{seed}/classifier.joblib'
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.write_text(f'synthetic classifier marker {seed}; not a pickle')
        digest = final._file_sha256(path)
        threshold = .6
        protocol['thresholds'][str(seed)] = {
            'classifier_path': relative, 'classifier_sha256': digest,
            'value': threshold, 'decimal': repr(threshold), 'binary64_hex': threshold.hex()}
        protocol['input_files_sha256'][relative] = digest
    monkeypatch.setattr(final.importlib.metadata, 'version', lambda _name: '1.0')
    def load_encoder(_snapshot):
        final.freeze_encoder(encoder)
        return encoder
    monkeypatch.setattr(final, '_load_local_encoder', load_encoder)
    monkeypatch.setattr(final, '_load_classifier', lambda _path: classifier)
    rows = [{'id': f'test:{index:05d}', 'text': str(index)} for index in range(4)]
    return SimpleNamespace(root=tmp_path, protocol=protocol, encoder=encoder, classifier=classifier,
                           rows=rows, output=tmp_path / 'output')


def run(fixture, **kwargs):
    return final.run_specialists(kwargs.pop('rows', fixture.rows), fixture.protocol,
                                 kwargs.pop('output', fixture.output), root=fixture.root, **kwargs)


def test_five_frozen_classifiers_label_free_alignment_and_boundary(synthetic):
    result = run(synthetic)
    assert set(result) == {'11', '22', '33', '44', '55'}
    for seed, predictions in result.items():
        assert [r['id'] for r in predictions] == [r['id'] for r in synthetic.rows]
        assert [r['predicted_label'] for r in predictions] == ['z', 'a', 'z', 'a']
        assert [r['use_specialist'] for r in predictions] == [True, True, False, False]
        assert all(set(r) == {'id', 'predicted_label', 'confidence', 'use_specialist'} for r in predictions)
        directory = synthetic.output / 'specialists' / seed
        probabilities = read_json(directory / 'probabilities.json')
        assert probabilities['labels'] == ['z', 'a', 'm']
        assert probabilities['values'][0] == [.6, .2, .2]
        assert read_json(directory / 'metadata.json')['label_access'] is False
    assert synthetic.encoder.calls == [['0', '1', '2', '3']]
    assert all(not param.requires_grad and param.grad is None for param in synthetic.encoder.parameters())


def test_local_encoder_loader_forces_offline_frozen_settings(synthetic, monkeypatch):
    # Test the real loader with a synthetic SentenceTransformer constructor.
    import importlib
    reloaded = importlib.reload(final)
    calls = []
    def constructor(*args, **kwargs):
        calls.append((args, kwargs))
        return synthetic.encoder
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=constructor))
    loaded = reloaded._load_local_encoder('/synthetic/local/snapshot')
    assert loaded is synthetic.encoder
    assert calls[0][1]['local_files_only'] is True
    assert calls[0][1]['trust_remote_code'] is False
    assert calls[0][1]['model_kwargs'] == {'use_safetensors': True, 'attn_implementation': 'eager'}
    assert not loaded.training and not loaded.weight.requires_grad


def test_gates_independent_of_request_permutation_or_batch_coverage(synthetic):
    first = run(synthetic)
    permuted = list(reversed(synthetic.rows))
    second = run(synthetic, rows=permuted, output=synthetic.root / 'permuted')
    for seed in first:
        assert {r['id']: r for r in first[seed]} == {r['id']: r for r in second[seed]}
        assert sum(r['use_specialist'] for r in first[seed]) == 2  # 50%, never forced to 90%.


@pytest.mark.parametrize('field', ['label', 'true_label', 'truth', 'specialist_confidence'])
def test_labels_or_extra_fields_cannot_enter_inference(synthetic, field):
    rows = deepcopy(synthetic.rows)
    rows[0][field] = 'secret label'
    with pytest.raises(ValueError, match='labels are forbidden'):
        run(synthetic, rows=rows)
    assert not synthetic.encoder.calls
    assert not synthetic.output.exists()


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'empty_text'])
def test_rejects_population_drift(synthetic, mutation):
    rows = deepcopy(synthetic.rows)
    if mutation == 'duplicate':
        rows[1]['id'] = rows[0]['id']
    elif mutation == 'missing':
        rows.pop()
    else:
        rows[0]['text'] = ''
    with pytest.raises(ValueError):
        run(synthetic, rows=rows)
    assert not synthetic.encoder.calls


@pytest.mark.parametrize('mutation', ['classifier_bytes', 'encoder_bytes', 'package', 'settings', 'threshold', 'refit'])
def test_rejects_frozen_input_drift_before_model_load(synthetic, mutation, monkeypatch):
    if mutation == 'classifier_bytes':
        item = synthetic.protocol['thresholds']['11']
        (synthetic.root / item['classifier_path']).write_text('changed')
    elif mutation == 'encoder_bytes':
        (final._snapshot_path(synthetic.protocol, synthetic.root) / 'model.safetensors').write_text('changed')
    elif mutation == 'package':
        monkeypatch.setattr(final.importlib.metadata, 'version', lambda _name: 'changed')
    elif mutation == 'settings':
        synthetic.protocol['specialists']['embedding']['settings']['normalize_embeddings'] = False
    elif mutation == 'threshold':
        synthetic.protocol['thresholds']['11']['value'] = .9
    else:
        synthetic.protocol['specialists']['refit'] = True
    with pytest.raises(ValueError):
        run(synthetic)
    assert not synthetic.encoder.calls


@pytest.mark.parametrize('target', ['encoder', 'classifier'])
def test_rejects_state_mutation_during_inference(synthetic, target):
    getattr(synthetic, target).mutate = True
    with pytest.raises(ValueError, match='state changed'):
        run(synthetic)
    assert not (synthetic.output / 'specialists/11/predictions.json').exists()


def test_rejects_loaded_classifier_hyperparameter_drift(synthetic):
    synthetic.classifier.C = 2
    with pytest.raises(ValueError, match='hyperparameters changed'):
        run(synthetic)


def test_rejects_loaded_encoder_state_drift(synthetic):
    synthetic.protocol['specialists']['embedding']['encoder_state_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='state mismatch before inference'):
        run(synthetic)


def test_resume_reuses_complete_checkpoints_without_inference(synthetic, monkeypatch):
    expected = run(synthetic)
    before = {str(p.relative_to(synthetic.output)): p.read_bytes()
              for p in synthetic.output.rglob('*.json')}
    def forbidden(*_args):
        raise AssertionError('Completed specialist checkpoints must not rerun inference')
    monkeypatch.setattr(final, '_load_local_encoder', forbidden)
    monkeypatch.setattr(final, '_load_classifier', forbidden)
    assert run(synthetic) == expected
    assert before == {str(p.relative_to(synthetic.output)): p.read_bytes()
                      for p in synthetic.output.rglob('*.json')}


def test_partial_checkpoint_resumes_without_overwriting_predictions(synthetic):
    expected = run(synthetic)
    directory = synthetic.output / 'specialists/55'
    before = (directory / 'predictions.json').read_bytes()
    (directory / 'metadata.json').unlink()
    assert run(synthetic) == expected
    assert (directory / 'predictions.json').read_bytes() == before


@pytest.mark.parametrize('mutation', ['probabilities', 'predictions', 'input_text', 'threshold', 'metadata'])
def test_resume_rejects_provenance_or_output_drift(synthetic, mutation):
    run(synthetic)
    directory = synthetic.output / 'specialists/11'
    if mutation in ('probabilities', 'predictions'):
        path = directory / f'{mutation}.json'
        path.write_text('{}')
    elif mutation == 'input_text':
        synthetic.rows[0]['text'] = '1'
    elif mutation == 'threshold':
        item = synthetic.protocol['thresholds']['11']
        item.update(value=.9, decimal=repr(.9), binary64_hex=(.9).hex())
    else:
        path = directory / 'metadata.json'
        record = read_json(path)
        record['frozen_encoder']['trainable_parameters'] = 1
        write_json(path, record)
    with pytest.raises(ValueError):
        run(synthetic)


def test_probability_ties_use_frozen_classifier_class_order():
    probabilities = np.asarray([[.5, .5, 0]], dtype=np.float64)
    records = final._predictions(['x'], ['z', 'a', 'm'], probabilities, .5)
    assert records == [{'id': 'x', 'predicted_label': 'a', 'confidence': .5, 'use_specialist': True}]


@pytest.mark.parametrize('values', [[[.2, .2, .2]], [[float('nan'), .5, .5]], [[-.1, .6, .5]]])
def test_invalid_probability_matrices_rejected(values):
    with pytest.raises(ValueError, match='probabilities'):
        final._predictions(['x'], ['a', 'b', 'c'], np.asarray(values, dtype=np.float64), .5)


def test_verify_saved_specialists_is_read_only_and_requires_no_model_loading(synthetic, monkeypatch):
    expected = run(synthetic)
    def forbidden(*_args):
        raise AssertionError('Checkpoint verification must not load or run models')
    monkeypatch.setattr(final, '_load_local_encoder', forbidden)
    monkeypatch.setattr(final, '_load_classifier', forbidden)
    before = {str(path): path.read_bytes() for path in synthetic.output.rglob('*.json')}
    assert final.verify_saved_specialists(synthetic.rows, synthetic.protocol, synthetic.output,
                                          root=synthetic.root) == expected
    assert before == {str(path): path.read_bytes() for path in synthetic.output.rglob('*.json')}


@pytest.mark.parametrize('seed', final.SEEDS)
def test_verify_saved_specialists_rejects_any_missing_receipt(synthetic, seed):
    run(synthetic)
    (synthetic.output / 'specialists' / str(seed) / 'metadata.json').unlink()
    with pytest.raises(ValueError, match='Missing completed specialist checkpoint'):
        final.verify_saved_specialists(synthetic.rows, synthetic.protocol, synthetic.output,
                                        root=synthetic.root)


def test_immutable_checkpoint_fsyncs_parent_directory(synthetic, monkeypatch):
    calls = []
    original = final._fsync_directory
    def recorded(path):
        calls.append(Path(path))
        original(path)
    monkeypatch.setattr(final, '_fsync_directory', recorded)
    path = synthetic.root / 'durable/checkpoint.json'
    final._write_once(path, {'synthetic': True})
    assert calls == [path.parent, path.parent]
    final._write_once(path, {'synthetic': True})
    assert calls[-1] == path.parent


@pytest.mark.parametrize('prompts', [{}, {'query': '', 'document': ''}, {'custom': ''},
                                   UserDict({'query': '', 'document': ''})])
def test_original_loader_accepts_only_inert_mapping_without_mutation(prompts, monkeypatch):
    encoder = SyntheticEncoder()
    encoder.prompts = prompts
    original_mapping = encoder.prompts
    before = dict(prompts)
    calls = []
    def constructor(*args, **kwargs):
        calls.append((args, kwargs))
        return encoder
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=constructor))
    loaded = final._load_local_encoder('/synthetic/local/snapshot')
    assert loaded is encoder and loaded.prompts is original_mapping
    assert dict(loaded.prompts) == before and loaded.default_prompt_name is None
    assert calls == [(('/synthetic/local/snapshot',), {
        'device': 'cpu', 'backend': 'torch', 'local_files_only': True,
        'trust_remote_code': False, 'prompts': {}, 'default_prompt_name': None,
        'model_kwargs': {'use_safetensors': True, 'attn_implementation': 'eager'},
    })]
    assert not loaded.training and not loaded.weight.requires_grad


@pytest.mark.parametrize('prompts', [
    None, [], (), '', 0, [('query', '')],
    {'query': 'prefix'}, {'query': ' '}, {'query': '\n'},
    {'query': None}, {'query': False}, {'query': 0}, {'query': b''},
    {'query': []}, {'query': {}}, {1: ''}, {None: ''},
    {'query': '', 'document': 'prefix'},
])
def test_original_loader_rejects_nonempty_or_malformed_prompts(prompts, monkeypatch):
    encoder = SyntheticEncoder()
    encoder.prompts = prompts
    monkeypatch.setitem(sys.modules, 'sentence_transformers',
                        SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: encoder))
    with pytest.raises(ValueError, match='prompts must remain disabled'):
        final._load_local_encoder('/synthetic/local/snapshot')
    assert encoder.prompts is prompts


@pytest.mark.parametrize('default', ['query', 'document', '', False, 0])
@pytest.mark.parametrize('prompts', [{}, {'query': '', 'document': ''}])
def test_original_loader_rejects_any_non_none_default(default, prompts, monkeypatch):
    encoder = SyntheticEncoder()
    encoder.prompts = prompts
    encoder.default_prompt_name = default
    monkeypatch.setitem(sys.modules, 'sentence_transformers',
                        SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: encoder))
    with pytest.raises(ValueError, match='prompts must remain disabled'):
        final._load_local_encoder('/synthetic/local/snapshot')
    assert encoder.prompts is prompts and encoder.default_prompt_name is default
