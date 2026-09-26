"""Label-free inference using the five already-frozen EXP-007 specialists.

This module has no dataset loader, downloader, fitting method, or scoring code.
The guarded CLI must authorize unsealing before constructing its ID/text rows.
All writes are immutable checkpoints; resumed checkpoints are fully revalidated.
"""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .data import ROOT, json_bytes, read_json, sha256
from .embeddings import (ENCODING, MODEL_ID, MODEL_REVISION, assert_frozen,
                         freeze_encoder, state_digest)
from .experiment import LOGISTIC
from .thresholds import use_specialist


SEEDS = [11, 22, 33, 44, 55]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_path(protocol, root):
    embedding = protocol['specialists']['embedding']
    model_dir = 'models--' + embedding['model_id'].replace('/', '--')
    return root / '.cache/huggingface/hub' / model_dir / 'snapshots' / embedding['revision']


def verify_specialist_files(protocol, root=ROOT):
    """Hash pinned local files and check packages; never deserialize or download."""
    root = Path(root)
    config = protocol['specialists']
    embedding = config['embedding']
    _require(config['seeds'] == SEEDS and config['shots'] == 20 and
             not config['refit'] and not config['tune'] and not config['recalibrate'],
             'Frozen specialist configuration changed')
    _require(embedding['model_id'] == MODEL_ID and embedding['revision'] == MODEL_REVISION and
             embedding['settings'] == ENCODING and embedding['logistic_regression'] == LOGISTIC,
             'Pinned embedding model/settings or logistic regression changed')
    _require(set(protocol['thresholds']) == {str(seed) for seed in SEEDS},
             'Expected exactly the five frozen thresholds')
    hashes = {}
    for seed in SEEDS:
        item = protocol['thresholds'][str(seed)]
        threshold = item['value']
        _require(type(threshold) is float and threshold.hex() == item['binary64_hex'] and
                 repr(threshold) == item['decimal'], 'Frozen threshold representations disagree')
        use_specialist(0.0, threshold)
        relative = Path(item['classifier_path'])
        _require(not relative.is_absolute() and '..' not in relative.parts,
                 'Classifier must be a repository-relative frozen artifact')
        actual = _file_sha256(root / relative)
        _require(actual == item['classifier_sha256'] ==
                 protocol['input_files_sha256'][str(relative)], 'Frozen classifier hash mismatch')
        hashes[str(relative)] = actual
    snapshot = _snapshot_path(protocol, root)
    for name, expected in embedding['model_files_sha256'].items():
        relative = Path(name)
        _require(not relative.is_absolute() and '..' not in relative.parts,
                 'Unexpected pinned encoder file path')
        actual = _file_sha256(snapshot / relative)
        _require(actual == expected, 'Pinned local encoder file hash mismatch')
        hashes[str((snapshot / relative).relative_to(root))] = actual
    installed = {}
    for name, version in embedding['recorded_packages'].items():
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f'Pinned package unavailable: {name}') from error
        _require(installed[name] == version, f'Pinned package version mismatch: {name}')
    return {'files_sha256': hashes, 'packages': installed, 'snapshot': str(snapshot)}


def _load_local_encoder(snapshot):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        str(snapshot), device='cpu', backend='torch', local_files_only=True,
        trust_remote_code=False, prompts={}, default_prompt_name=None,
        model_kwargs={'use_safetensors': True, 'attn_implementation': 'eager'},
    )
    encoder.float()
    _require(encoder.max_seq_length == ENCODING['max_seq_length'] and
             encoder.get_sentence_embedding_dimension() == ENCODING['dimensions'],
             'Pinned encoder architecture/settings mismatch')
    _require(not encoder.prompts and encoder.default_prompt_name is None,
             'Encoder prompts must remain disabled')
    freeze_encoder(encoder)
    return encoder


def _load_classifier(path):
    # Call only after the frozen artifact digest is checked. No user pickle path.
    return joblib.load(path)


def _validate_classifier(classifier, labels, expected):
    _require(isinstance(classifier, LogisticRegression), 'Expected saved logistic regression')
    _require(classifier.get_params(deep=False) == LogisticRegression(**expected).get_params(deep=False),
             'Saved logistic regression hyperparameters changed')
    classes = classifier.classes_.tolist()
    _require(len(classes) == len(labels) and len(set(classes)) == len(classes) and
             classes == sorted(labels), 'Saved classifier label mapping changed')
    _require(classifier.n_features_in_ == ENCODING['dimensions'] and
             classifier.coef_.shape == (len(labels), ENCODING['dimensions']) and
             classifier.intercept_.shape == (len(labels),) and
             np.isfinite(classifier.coef_).all() and np.isfinite(classifier.intercept_).all(),
             'Saved classifier dimensions or parameters invalid')
    return [classes.index(label) for label in labels]


def _classifier_state(classifier):
    return sha256(json_bytes({'parameters': classifier.get_params(deep=False),
                             'classes': classifier.classes_.tolist(),
                             'coefficients': classifier.coef_.tolist(),
                             'intercept': classifier.intercept_.tolist()}))


def _validate_rows(rows, protocol):
    _require(isinstance(rows, list) and len(rows) ==
             protocol['population']['expected_count_from_existing_metadata'],
             'Expected the complete frozen request population')
    _require(all(isinstance(row, dict) and set(row) == {'id', 'text'} and
                 isinstance(row['id'], str) and row['id'] and
                 isinstance(row['text'], str) and row['text'].strip() for row in rows),
             'Inference rows must contain only nonempty ID and text; labels are forbidden')
    _require(len({row['id'] for row in rows}) == len(rows), 'Duplicate inference IDs')


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path, value):
    """Atomic creation only; a prior checkpoint must agree byte-for-byte."""
    payload = json_bytes(value)
    if path.exists():
        _require(path.read_bytes() == payload, f'Existing specialist evidence changed: {path.name}')
        _fsync_directory(path.parent)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink()
        _fsync_directory(path.parent)


def _check_saved(seed_dir, binding, ids, labels, threshold):
    meta_path = seed_dir / 'metadata.json'
    if not meta_path.exists():
        return None
    meta = read_json(meta_path)
    _require(meta['binding'] == binding, 'Resumed specialist provenance changed')
    for name in ('predictions.json', 'probabilities.json'):
        _require(_file_sha256(seed_dir / name) == meta['files_sha256'][name],
                 'Resumed specialist artifact hash mismatch')
    probabilities = read_json(seed_dir / 'probabilities.json')
    _require(set(probabilities) == {'ids', 'labels', 'values'} and
             probabilities['ids'] == ids and probabilities['labels'] == labels,
             'Resumed probability rows/columns changed')
    expected = _predictions(ids, labels, np.asarray(probabilities['values'], dtype=np.float64), threshold)
    predictions = read_json(seed_dir / 'predictions.json')
    _require(predictions == expected, 'Resumed predictions, confidences or routing changed')
    _require(meta['frozen_encoder']['state_sha256_before'] == binding['encoder_state_sha256'] ==
             meta['frozen_encoder']['state_sha256_after'] and
             meta['frozen_encoder']['trainable_parameters'] == 0 and
             meta['classifier_state_sha256_before'] == meta['classifier_state_sha256_after'],
             'Resumed freeze checks changed')
    return predictions


def _predictions(ids, labels, probabilities, threshold):
    _require(probabilities.shape == (len(ids), len(labels)) and
             probabilities.dtype == np.float64 and np.isfinite(probabilities).all() and
             (probabilities >= 0).all() and (probabilities <= 1).all() and
             np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12),
             'Invalid binary64 classifier probabilities')
    # Argmax is within one request only. No request ranks/quantiles are computed.
    class_order = np.asarray(sorted(range(len(labels)), key=lambda index: labels[index]))
    best = class_order[probabilities[:, class_order].argmax(axis=1)]
    return [{'id': row_id, 'predicted_label': labels[int(best[index])],
             'confidence': float(probabilities[index, best[index]]),
             'use_specialist': use_specialist(float(probabilities[index, best[index]]), threshold)}
            for index, row_id in enumerate(ids)]


def _checkpoint_bindings(rows, protocol):
    ids = [row['id'] for row in rows]
    labels = protocol['population']['labels']
    _require(len(set(labels)) == len(labels), 'Duplicate frozen labels')
    input_digest = sha256(json_bytes([{'id': row['id'], 'text_sha256': sha256(row['text'].encode())}
                                     for row in rows]))
    protocol_digest = sha256(json_bytes(protocol))
    state_expected = protocol['specialists']['embedding']['encoder_state_sha256']
    bindings = {}
    for seed in SEEDS:
        key = str(seed)
        bindings[key] = {'protocol_sha256': protocol_digest, 'inputs_sha256': input_digest,
                         'seed': seed, 'threshold_hex': protocol['thresholds'][key]['binary64_hex'],
                         'classifier_sha256': protocol['thresholds'][key]['classifier_sha256'],
                         'encoder_state_sha256': state_expected}
    return ids, labels, bindings


def verify_saved_specialists(rows, protocol, output, root=ROOT):
    """Require all five completed label-free checkpoints, without any inference.

    The guarded collector uses this receipt immediately before API collection;
    labels and model deserialization are unavailable to this replay path.
    """
    _validate_rows(rows, protocol)
    output = Path(output)
    before = verify_specialist_files(protocol, root=root)
    ids, labels, bindings = _checkpoint_bindings(rows, protocol)
    results = {}
    for seed in SEEDS:
        key = str(seed)
        saved = _check_saved(output / 'specialists' / key, bindings[key], ids, labels,
                             protocol['thresholds'][key]['value'])
        _require(saved is not None, f'Missing completed specialist checkpoint for seed {seed}')
        results[key] = saved
    _require(verify_specialist_files(protocol, root=root) == before,
             'Frozen model files/packages changed during checkpoint verification')
    return results


def run_specialists(rows, protocol, output, *, root=ROOT):
    """Persist label-free predictions/gates; only the guarded caller may supply test rows.

    Return five lists keyed by seed string. Complete checkpoints replay without
    inference; a partial seed can only be completed with identical predictions.
    """
    _validate_rows(rows, protocol)
    root, output = Path(root), Path(output)
    checked = verify_specialist_files(protocol, root=root)
    ids, labels, bindings = _checkpoint_bindings(rows, protocol)
    state_expected = protocol['specialists']['embedding']['encoder_state_sha256']
    results = {}
    for seed in SEEDS:
        key = str(seed)
        saved = _check_saved(output / 'specialists' / key, bindings[key], ids, labels,
                             protocol['thresholds'][key]['value'])
        if saved is not None:
            results[key] = saved
    pending = [str(seed) for seed in SEEDS if str(seed) not in results]
    if pending:
        torch.set_num_threads(1)
        torch.manual_seed(11)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):
            encoder = _load_local_encoder(checked['snapshot'])
            assert_frozen(encoder)
            before = state_digest(encoder)
            _require(before == state_expected, 'Frozen encoder state mismatch before inference')
            started = time.perf_counter()
            with torch.inference_mode():
                values = encoder.encode(
                    [row['text'] for row in rows], batch_size=ENCODING['batch_size'],
                    precision='float32', device='cpu', normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False, prompt=None, prompt_name=None,
                )
            encoding_seconds = time.perf_counter() - started
            _require(isinstance(values, np.ndarray) and
                     values.shape == (len(rows), ENCODING['dimensions']) and
                     values.dtype == np.float32 and np.isfinite(values).all() and
                     np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5),
                     'Encoded rows/dimensions/precision/normalization invalid')
            assert_frozen(encoder)
            after = state_digest(encoder)
            _require(before == after, 'Frozen encoder state changed during inference')
            for key in pending:
                item = protocol['thresholds'][key]
                path = root / item['classifier_path']
                _require(_file_sha256(path) == item['classifier_sha256'],
                         'Frozen classifier changed before deserialization')
                classifier = _load_classifier(path)
                order = _validate_classifier(classifier, labels,
                                             protocol['specialists']['embedding']['logistic_regression'])
                classifier_before = _classifier_state(classifier)
                started = time.perf_counter()
                raw_probabilities = classifier.predict_proba(values)
                _require(raw_probabilities.dtype == np.float64,
                         'Saved classifier must produce binary64 probabilities')
                probabilities = raw_probabilities[:, order]
                predictions = _predictions(ids, labels, probabilities, item['value'])
                _require(classifier.predict(values).tolist() ==
                         [row['predicted_label'] for row in predictions],
                         'Classifier predictions and probability class ordering disagree')
                classifier_seconds = time.perf_counter() - started
                classifier_after = _classifier_state(classifier)
                _require(classifier_before == classifier_after, 'Classifier state changed during inference')
                assert_frozen(encoder)
                _require(state_digest(encoder) == before, 'Frozen encoder changed during classifier inference')
                seed_dir = output / 'specialists' / key
                _write_once(seed_dir / 'probabilities.json', {'ids': ids, 'labels': labels,
                                                             'values': probabilities.tolist()})
                _write_once(seed_dir / 'predictions.json', predictions)
                _write_once(seed_dir / 'metadata.json', {
                    'binding': bindings[key], 'settings': ENCODING, 'label_access': False,
                    'files_sha256': {name: _file_sha256(seed_dir / name)
                                     for name in ('predictions.json', 'probabilities.json')},
                    'frozen_encoder': {'state_sha256_before': before, 'state_sha256_after': after,
                                       'trainable_parameters': 0, 'inference_mode': True},
                    'classifier_state_sha256_before': classifier_before,
                    'classifier_state_sha256_after': classifier_after,
                    'timings': {'shared_encoding_seconds': encoding_seconds,
                                'classifier_prediction_seconds': classifier_seconds,
                                'note': 'Encoding measured once per invocation and shared across pending seeds; '
                                        'do not add it five times or treat classifier-only time as end-to-end.'},
                })
                results[key] = predictions
    _require(verify_specialist_files(protocol, root=root) == checked,
             'Frozen model files/packages changed during specialist inference')
    return {str(seed): results[str(seed)] for seed in SEEDS}
