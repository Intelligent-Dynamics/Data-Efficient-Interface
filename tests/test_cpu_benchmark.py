"""Synthetic CPU timing mechanics only; no real models/data/network."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from baseline import cpu_benchmark as cpu
from baseline.data import json_bytes, sha256, write_json


class Encoder:
    def __init__(self):
        self.seen = []

    def encode(self, texts, **settings):
        self.seen.append((list(texts), settings))
        assert torch.is_inference_mode_enabled()
        return np.asarray([[len(t), 1] for t in texts], dtype=np.float32)


class Classifier:
    def predict_proba(self, values):
        # Deliberately return an exact threshold boundary and alphabetical tie.
        return np.tile(np.asarray([0.5, 0.5], dtype=np.float64), (len(values), 1))


def test_raw_text_full_path_exact_gate_no_cached_features():
    encoder = Encoder()
    values, predicted, confidence, gates = cpu.infer_batch(
        encoder, Classifier(), ['first', 'second'], ['z', 'a'], [0, 1], 0.5, 32)
    assert predicted == ['a', 'a']
    assert gates.tolist() == [True, True]
    assert confidence.tolist() == [0.5, 0.5]
    assert values.shape == (2, 2)
    assert encoder.seen[0][0] == ['first', 'second']
    assert encoder.seen[0][1] == dict(batch_size=32, precision='float32', device='cpu',
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False, prompt=None, prompt_name=None)


def test_timed_pass_keeps_all_rows_and_partial_batch():
    encoder = Encoder()
    clock = iter([0, 10, 20, 45]).__next__
    rows = [{'id': str(i), 'text': f'text-{i}'} for i in range(3)]
    times, values, prediction, confidence, gates = cpu.timed_pass(
        encoder, Classifier(), rows, ['z', 'a'], [0, 1], 0.6, 2, clock)
    assert times == [10, 25]
    assert values.shape == (3, 2)
    assert prediction == ['a'] * 3
    assert gates == [False] * 3
    assert [t for texts, _ in encoder.seen for t in texts] == ['text-0', 'text-1', 'text-2']


def test_single_latency_and_batch_throughput_distinct():
    single = cpu.summarize([{'durations_ns': [1_000_000, 3_000_000]} for _ in range(5)], 2, 1)
    assert single['median_request_latency_ms'] == 2
    assert single['p95_request_latency_ms'] == 3
    assert single['pooled_throughput_requests_per_second'] == 500
    batch = cpu.summarize([{'durations_ns': [4_000_000]} for _ in range(5)], 2, 32)
    assert batch['pooled_throughput_requests_per_second'] == 500
    assert 'median_request_latency_ms' not in batch


@pytest.mark.parametrize('passes,count,batch', [([], 2, 1), ([{'durations_ns': [1]}] * 5, 2, 1),
                                              ([{'durations_ns': [0, 1]}] * 5, 2, 1)])
def test_summary_rejects_missing_or_invalid_measurements(passes, count, batch):
    with pytest.raises(ValueError):
        cpu.summarize(passes, count, batch)


def test_numerical_boundary_difference_reported_without_threshold_adjustment():
    reference = np.asarray([[0.5, 0.5]], dtype=np.float64)
    actual = np.asarray([[0.5000000001, 0.4999999999]], dtype=np.float64)
    result = cpu.comparison(['train:00001'], actual, ['a'], [0.5000000001], [True],
                            reference, ['a', 'b'], 0.50000000005)
    assert result['gate_mismatch_ids'] == ['train:00001']
    assert result['prediction_mismatch_ids'] == []
    assert result['reference_accepted_count'] == 0
    assert result['accepted_count'] == 1
    assert result['maximum_absolute_probability_difference'] > 0


def test_guard_denies_test_and_network_without_opening(monkeypatch, tmp_path):
    hooks = []
    monkeypatch.setattr(cpu.sys, 'addaudithook', hooks.append)
    counts = cpu.offline_guard(tmp_path)
    hooks[0]('open', (str(tmp_path / 'data/raw/banking77/train.csv'),))
    with pytest.raises(PermissionError, match='official-test'):
        hooks[0]('open', (str(tmp_path / 'data/raw/banking77/test.csv'),))
    with pytest.raises(PermissionError, match='network'):
        hooks[0]('socket.connect', ())
    assert counts == {'official_test_access_attempts': 1, 'network_access_attempts': 1}


def test_validation_loader_only_opens_training_source(monkeypatch, tmp_path):
    labels = [f'label-{i}' for i in range(77)]
    ids = [f'train:{i:05d}' for i in range(770)]
    path = tmp_path / 'data/raw/banking77/train.csv'
    path.parent.mkdir(parents=True)
    path.write_text('text,category\n' + ''.join(f'text-{i},{labels[i // 10]}\n' for i in range(770)))
    source = {'files': {'train.csv': {'sha256': sha256(path.read_bytes())}}}
    record = {'samples': {'labels': labels, 'validation_ids': ids, 'train_ids': ['train:00999']},
              'metadata': {'source': source},
              'predictions': [{'id': i, 'true_label': labels[n // 10]} for n, i in enumerate(ids)]}
    (tmp_path / cpu.REFERENCE).parent.mkdir(parents=True)
    (tmp_path / cpu.PROBABILITIES).parent.mkdir(parents=True)
    write_json(tmp_path / cpu.REFERENCE, record)
    write_json(tmp_path / cpu.PROBABILITIES, {'validation_ids': ids, 'labels': labels, 'probabilities': [[1 / 77] * 77] * 770})
    protocol = {'input_files_sha256': {str(p): sha256((tmp_path / p).read_bytes()) for p in (cpu.REFERENCE, cpu.PROBABILITIES)}}
    opened = []
    original = Path.open
    def guarded(path, *args, **kwargs):
        opened.append(path)
        assert path.name != 'test.csv'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    rows, actual_labels, probability, digest = cpu.validation_inputs(protocol, tmp_path)
    assert [r['id'] for r in rows] == ids
    assert all(set(r) == {'id', 'text'} for r in rows)
    assert actual_labels == labels and probability.shape == (770, 77)
    assert all('test.csv' not in str(p) for p in opened)


@pytest.mark.parametrize('prompts,default,accepted', [({'query': '', 'document': ''}, None, True),
                                                   ({'query': 'prefix'}, None, False),
                                                   ({'query': ''}, 'query', False)])
def test_empty_library_prompt_names_are_inert(monkeypatch, prompts, default, accepted):
    import types
    class LocalEncoder(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            assert kwargs['local_files_only'] is True
            assert kwargs['prompts'] == {} and kwargs['default_prompt_name'] is None
            self.prompts = prompts
            self.default_prompt_name = default
            self.max_seq_length = 256
        def get_sentence_embedding_dimension(self):
            return 384
    monkeypatch.setitem(cpu.sys.modules, 'sentence_transformers', types.SimpleNamespace(SentenceTransformer=LocalEncoder))
    if accepted:
        model = cpu.load_benchmark_encoder('/synthetic/local/model')
        assert not model.training
    else:
        with pytest.raises(ValueError, match='prompt'):
            cpu.load_benchmark_encoder('/synthetic/local/model')
