"""Frozen MiniLM smoke baseline: reuse EXP-002 IDs; never prepare new splits."""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import warnings

import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_info, threadpool_limits

from .data import (DEFAULT_MANIFEST, PROTOCOL_ID, json_bytes, load_development,
                   read_json, sha256, verify_isolation, write_json)
from .experiment import LOGISTIC, classification_metrics, code_provenance

REFERENCE = Path('experiments/exp002-learning-curve/runs/exp002-v2-n5-s11.json')
SAMPLE_SHA256 = '018f481053a04611d7171083b1235e8108a26a1a4ac8d4a3eb3cba210e481af8'
MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL_FILES = [
    'model.safetensors', 'config.json', 'modules.json', 'config_sentence_transformers.json',
    'sentence_bert_config.json', '1_Pooling/config.json', 'tokenizer.json',
    'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'README.md',
]
ENCODING = {
    'device': 'cpu', 'backend': 'torch', 'precision': 'float32', 'dimensions': 384,
    'batch_size': 32, 'max_seq_length': 256, 'normalize_embeddings': True,
    'pooling': 'attention-mask mean pooling', 'prompt': None,
    'attention_implementation': 'eager', 'compute_threads': 1,
    'trust_remote_code': False, 'use_safetensors': True,
}
METRIC_KEY = 'frozen_minilm_logistic_regression'


def reuse_reference(raw, manifest_path, reference):
    """Resolve existing sample IDs verbatim; the loader only verifies the fixed split."""
    record = read_json(reference)
    meta, samples = record['metadata'], record['samples']
    if (meta['status'] != 'completed' or meta['protocol_id'] != PROTOCOL_ID
            or meta['evaluation_split'] != 'validation'
            or (samples['shots'], samples['seed']) != (5, 11)
            or (meta['shots'], meta['seed']) != (5, 11)
            or meta['configuration']['logistic_regression'] != LOGISTIC):
        raise ValueError('Expected completed v2 5-shot seed-11 reference with unchanged LR')
    if sha256(json_bytes(samples)) != SAMPLE_SHA256 or meta['samples_sha256'] != SAMPLE_SHA256:
        raise ValueError('Reference sample IDs changed')
    manifest, pool, validation = load_development(raw, manifest_path)
    if (sha256(manifest_path.read_bytes()) != meta['split_manifest_sha256']
            or manifest['source'] != meta['source'] or manifest['labels'] != samples['labels']
            or [r['id'] for r in validation] != samples['validation_ids']):
        raise ValueError('Reference and existing dataset/split disagree')
    by_id = {row['id']: row for row in pool}
    if not set(samples['train_ids']) <= by_id.keys():
        raise ValueError('Reference training IDs outside existing training pool')
    train = [by_id[row_id] for row_id in samples['train_ids']]
    verify_isolation(train, validation, manifest['sealed_test'])
    labels = manifest['labels']
    if (Counter(r['label'] for r in train) != Counter({label: 5 for label in labels})
            or Counter(r['label'] for r in validation) != Counter({label: 10 for label in labels})):
        raise ValueError('Incorrect per-class label budget')
    predictions = record['predictions']
    if ([(p['id'], p['true_label']) for p in predictions]
            != [(r['id'], r['label']) for r in validation]):
        raise ValueError('Reference prediction rows/labels are not aligned')
    recomputed = classification_metrics([r['label'] for r in validation],
                                       [p['predicted_label'] for p in predictions], labels)
    if recomputed != record['metrics']['tfidf_logistic_regression']:
        raise ValueError('Reference metrics do not match saved predictions')
    return record, manifest, train, validation


def freeze_encoder(encoder):
    encoder.eval()
    encoder.requires_grad_(False)
    assert_frozen(encoder)


def assert_frozen(encoder):
    if any(module.training for module in encoder.modules()):
        raise ValueError('Encoder must remain in evaluation mode')
    if any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
        raise ValueError('Encoder must have no trainable parameters or gradients')


def state_digest(encoder):
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        digest.update(json_bytes([name, str(tensor.dtype), list(tensor.shape)]))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_encoder(cache):
    # Download only inference files at an immutable revision, then load locally.
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    start = time.perf_counter()
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                     cache_dir=str(cache), allow_patterns=MODEL_FILES))
    if snapshot.name != MODEL_REVISION or any(not (snapshot / f).is_file() for f in MODEL_FILES):
        raise ValueError('Incomplete or unexpected model snapshot')
    source = {'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'license': 'Apache-2.0',
              'files_sha256': {f: sha256((snapshot / f).read_bytes()) for f in MODEL_FILES},
              'snapshot_resolution_seconds': time.perf_counter() - start}
    start = time.perf_counter()
    encoder = SentenceTransformer(
        str(snapshot), device='cpu', backend='torch', local_files_only=True,
        trust_remote_code=False, prompts={}, default_prompt_name=None,
        model_kwargs={'use_safetensors': True, 'attn_implementation': 'eager'},
    )
    encoder.float()
    if encoder.max_seq_length != ENCODING['max_seq_length']:
        raise ValueError('Unexpected encoder sequence length')
    if encoder.get_sentence_embedding_dimension() != ENCODING['dimensions']:
        raise ValueError('Unexpected encoder dimension')
    source['model_load_seconds'] = time.perf_counter() - start
    source['architecture'] = str(encoder)
    return encoder, source


@dataclass
class Features:
    values: np.ndarray
    ids: list
    labels: list
    text_sha256: list

    def aligned(self, rows):
        if (self.ids != [r['id'] for r in rows] or self.labels != [r['label'] for r in rows]
                or self.text_sha256 != [sha256(r['text'].encode()) for r in rows]
                or len(set(self.ids)) != len(rows)
                or self.values.shape != (len(rows), ENCODING['dimensions'])
                or self.values.dtype != np.float32 or not np.isfinite(self.values).all()):
            raise ValueError('Embedding rows, labels, texts, or dimensions are not aligned')
        return self.values, self.labels

    def save(self, path):
        np.savez_compressed(path, values=self.values, ids=self.ids, labels=self.labels,
                            text_sha256=self.text_sha256)


def encode_rows(encoder, rows):
    assert_frozen(encoder)
    start = time.perf_counter()
    # SentenceTransformer.encode returns vectors in input order after internal batching.
    with torch.inference_mode():
        values = encoder.encode(
            [r['text'] for r in rows], batch_size=ENCODING['batch_size'],
            precision='float32', device='cpu', normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False, prompt=None, prompt_name=None,
        )
    seconds = time.perf_counter() - start
    assert_frozen(encoder)
    features = Features(values, [r['id'] for r in rows], [r['label'] for r in rows],
                        [sha256(r['text'].encode()) for r in rows])
    features.aligned(rows)
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5):
        raise ValueError('Expected normalized embeddings')
    return features, seconds


def environment():
    hardware = {'machine': platform.machine(), 'processor': platform.processor(),
                'logical_cpus': os.cpu_count(), 'device_used': 'cpu'}
    if sys.platform == 'darwin':
        for key in ('machdep.cpu.brand_string', 'hw.memsize', 'hw.logicalcpu'):
            hardware[key] = subprocess.check_output(['sysctl', '-n', key], text=True).strip()
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'hardware': hardware,
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            'torch_build': torch.__config__.show()}


def run(raw, manifest_path, output, cache):
    output.mkdir(parents=True, exist_ok=False)
    meta = {
        'schema_version': 1, 'run_id': output.name, 'status': 'running',
        'purpose': 'single validation smoke check; not a final research result',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_split': 'validation', 'shots': 5, 'seed': 11,
        'command': shlex.join([sys.executable, '-m', 'baseline.embeddings', *sys.argv[1:]]),
        'configuration': {'embedding': ENCODING, 'logistic_regression': LOGISTIC,
                          'multiclass_loss': 'multinomial', 'classifier_changes_from_tfidf': []},
        'warnings': [],
    }
    write_json(output / 'metadata.json', meta)
    try:
        meta['code'] = code_provenance(output)
        meta['environment'] = environment()
        reference, manifest, train, validation = reuse_reference(raw, manifest_path, REFERENCE)
        meta.update(protocol_id=manifest['protocol_id'], source=manifest['source'],
                    data_audit=manifest['audit'], label_budgets=reference['metadata']['label_budgets'],
                    samples_sha256=SAMPLE_SHA256,
                    split_manifest_sha256=sha256(manifest_path.read_bytes()))
        meta['reference'] = {'path': str(REFERENCE), 'sha256': sha256(REFERENCE.read_bytes()),
                             'run_id': reference['metadata']['run_id']}
        meta['test_access_this_run'] = 'byte-integrity verification only; no test rows parsed'
        write_json(output / 'samples.json', reference['samples'])
        shutil.copyfile(manifest_path, output / 'split_manifest.json')
        torch.set_num_threads(1)
        torch.manual_seed(11)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            encoder, meta['embedding_source'] = load_encoder(cache)
            freeze_encoder(encoder)
            before = state_digest(encoder)
            train_features, train_seconds = encode_rows(encoder, train)
            val_features, val_seconds = encode_rows(encoder, validation)
            x_train, y_train = train_features.aligned(train)
            x_val, truth = val_features.aligned(validation)
            classifier = LogisticRegression(**LOGISTIC)
            start = time.perf_counter()
            classifier.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - start
            meta['warnings'] = [f'{w.category.__name__}: {w.message}' for w in caught]
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError('Logistic regression did not converge; run is incomplete')
            start = time.perf_counter()
            predicted = classifier.predict(x_val).tolist()
            prediction_seconds = time.perf_counter() - start
            labels = manifest['labels']
            order = [classifier.classes_.tolist().index(label) for label in labels]
            probabilities = classifier.predict_proba(x_val)[:, order].tolist()
            assert_frozen(encoder)
            after = state_digest(encoder)
            if before != after:
                raise ValueError('Frozen encoder state changed during the experiment')
            meta['frozen_encoder_check'] = {
                'state_sha256_before': before, 'state_sha256_after': after,
                'trainable_parameters': 0, 'gradients_present': False,
                'evaluation_mode': True, 'inference_mode': True,
                'parameters': sum(p.numel() for p in encoder.parameters()),
                'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'torch_threads': torch.get_num_threads(),
            }
            meta['environment']['threadpools'] = threadpool_info()
            meta['warnings'] = [f'{w.category.__name__}: {w.message}' for w in caught]
        score = classification_metrics(truth, predicted, labels)
        write_json(output / 'metrics.json', {'schema_version': 1, 'evaluation_split': 'validation',
                                            METRIC_KEY: score})
        write_json(output / 'predictions.json', [
            {'id': r['id'], 'true_label': truth[i], 'predicted_label': predicted[i]}
            for i, r in enumerate(validation)])
        write_json(output / 'probabilities.json', {'labels': labels,
                   'validation_ids': val_features.ids, 'probabilities': probabilities})
        baseline = reference['metrics']['tfidf_logistic_regression']
        write_json(output / 'comparison.json', {
            'schema_version': 1, 'purpose': 'paired smoke check, one seed; no uncertainty estimate',
            'reference_run_id': reference['metadata']['run_id'], 'embedding_run_id': output.name,
            'protocol_id': manifest['protocol_id'], 'samples_sha256': SAMPLE_SHA256,
            'tfidf': {k: baseline[k] for k in ('accuracy', 'macro_f1')},
            'frozen_minilm': {k: score[k] for k in ('accuracy', 'macro_f1')},
            'difference_percentage_points': {k: 100 * (score[k] - baseline[k])
                                            for k in ('accuracy', 'macro_f1')},
        })
        train_features.save(output / 'train_embeddings.npz')
        val_features.save(output / 'validation_embeddings.npz')
        joblib.dump(classifier, output / 'classifier.joblib')
        meta['resources'] = {
            'training_embedding_seconds': train_seconds, 'validation_embedding_seconds': val_seconds,
            'classifier_fit_seconds': fit_seconds, 'classifier_predict_seconds': prediction_seconds,
            'timing_method': 'one cold pass each; perf_counter wall seconds; CPU, one compute thread',
            'timing_exclusions': 'state hashing, serialization and predict_proba; model loading recorded separately',
            'classifier_bytes': (output / 'classifier.joblib').stat().st_size,
        }
        meta['optimizer_iterations'] = classifier.n_iter_.tolist()
        meta['artifacts_sha256'] = {p.name: sha256(p.read_bytes()) for p in sorted(output.iterdir())
                                    if p.is_file() and p.name != 'metadata.json'}
        meta['status'] = 'completed'
        return score
    except Exception as exc:
        meta['status'] = 'failed'
        meta['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        meta['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'metadata.json', meta)


def verify_run(raw, manifest_path, output):
    """Replay only saved validation features/classifier; no encoder execution or refit."""
    meta = read_json(output / 'metadata.json')
    if meta['status'] != 'completed':
        raise ValueError('Cannot verify an incomplete run')
    for name, digest in meta['artifacts_sha256'].items():
        if sha256((output / name).read_bytes()) != digest:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    for name, digest in meta['code']['source_files_sha256'].items():
        if sha256((output / 'source' / name).read_bytes()) != digest:
            raise ValueError(f'Source snapshot mismatch: {name}')
    reference, manifest, train, validation = reuse_reference(raw, manifest_path, REFERENCE)
    if (read_json(output / 'samples.json') != reference['samples']
            or meta['reference']['sha256'] != sha256(REFERENCE.read_bytes())
            or meta['split_manifest_sha256'] != sha256(manifest_path.read_bytes())
            or meta['configuration']['logistic_regression'] != LOGISTIC):
        raise ValueError('Run does not match reference protocol')
    frozen = meta['frozen_encoder_check']
    if (frozen['state_sha256_before'] != frozen['state_sha256_after']
            or frozen['trainable_parameters'] != 0 or frozen['gradients_present']):
        raise ValueError('Encoder freeze check failed')
    arrays = []
    for name, rows in [('train', train), ('validation', validation)]:
        with np.load(output / f'{name}_embeddings.npz', allow_pickle=False) as saved:
            features = Features(saved['values'], saved['ids'].tolist(), saved['labels'].tolist(),
                                saved['text_sha256'].tolist())
        features.aligned(rows)
        arrays.append(features.values)
    classifier = joblib.load(output / 'classifier.joblib')
    if {k: classifier.get_params()[k] for k in LOGISTIC} != LOGISTIC:
        raise ValueError('Serialized classifier settings changed')
    with threadpool_limits(limits=1):
        predicted = classifier.predict(arrays[1]).tolist()
        labels = manifest['labels']
        order = [classifier.classes_.tolist().index(label) for label in labels]
        probabilities = classifier.predict_proba(arrays[1])[:, order]
    expected_predictions = [{'id': row['id'], 'true_label': row['label'], 'predicted_label': predicted[i]}
                            for i, row in enumerate(validation)]
    if read_json(output / 'predictions.json') != expected_predictions:
        raise ValueError('Saved classifier predictions differ')
    score = classification_metrics([r['label'] for r in validation], predicted, labels)
    if read_json(output / 'metrics.json')[METRIC_KEY] != score:
        raise ValueError('Saved metrics differ from replay')
    saved_probabilities = read_json(output / 'probabilities.json')
    if (saved_probabilities['labels'] != labels
            or saved_probabilities['validation_ids'] != reference['samples']['validation_ids']):
        raise ValueError('Probability labels/IDs are not aligned')
    np.testing.assert_allclose(probabilities, saved_probabilities['probabilities'], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    comparison = read_json(output / 'comparison.json')
    for key in ('accuracy', 'macro_f1'):
        baseline = reference['metrics']['tfidf_logistic_regression'][key]
        if (comparison['frozen_minilm'][key] != score[key] or comparison['tfidf'][key] != baseline
                or comparison['difference_percentage_points'][key] != 100 * (score[key] - baseline)):
            raise ValueError('Comparison does not match both saved runs')
    return {'schema_version': 1, 'run_id': meta['run_id'], 'status': 'passed',
            'checks': ['artifact/source hashes', 'reference sample IDs and budgets',
                       'ID/text isolation', 'feature row/text/label alignment',
                       'unchanged frozen encoder state', 'unchanged classifier settings',
                       'classifier prediction/probability replay', 'metric/comparison recomputation'],
            'training_examples': len(train), 'validation_examples': len(validation),
            'classifier_refitted': False, 'encoder_executed': False,
            'test_rows_parsed': False,
            'note': 'Artifact consistency check; not an independent training reproduction'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/raw/banking77'))
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--cache', type=Path, default=Path('.cache/huggingface/hub'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true', help='Verify an existing run without refitting')
    args = parser.parse_args()
    if args.verify:
        print(json_bytes(verify_run(args.raw, args.manifest, args.output)).decode(), end='')
        return
    score = run(args.raw, args.manifest, args.output, args.cache)
    print({k: score[k] for k in ('accuracy', 'macro_f1', 'n_examples')})


if __name__ == '__main__':
    main()
