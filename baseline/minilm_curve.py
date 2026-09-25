"""Frozen MiniLM validation curve: existing IDs, audited caches, independent refits."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shlex
import statistics
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from . import embeddings as emb
from .data import (DEFAULT_MANIFEST, PROTOCOL_ID, SEEDS, SHOTS, json_bytes,
                   load_development, read_json, sha256, verify_isolation, write_json)
from .experiment import LOGISTIC, classification_metrics, code_provenance
from .learning_curve import MODEL as TFIDF_MODEL, require, validate_record as validate_tfidf

GRID = [(n, s) for n in SHOTS for s in SEEDS]
REFERENCES = Path('experiments/exp002-learning-curve/runs')
SMOKE_RECORD = Path('experiments/exp003-minilm-v2-n5-s11')
CONFIG = {'embedding': emb.ENCODING, 'logistic_regression': LOGISTIC,
          'multiclass_loss': 'multinomial', 'classifier_changes_from_tfidf': []}
RUNTIME_PACKAGES = ('sentence-transformers', 'torch', 'transformers', 'numpy',
                    'scikit-learn', 'scipy', 'tokenizers', 'safetensors')


def now():
    return datetime.now(timezone.utc).isoformat()


def reference_path(shots, seed):
    return REFERENCES / f'exp002-v2-n{shots}-s{seed}.json'


def read_references():
    records = [read_json(reference_path(n, s)) for n, s in GRID]
    protocol_checks = read_json(Path('experiments/protocol-val10-v2-check.json'))
    hashes = {(c['shots'], c['seed']): c['train_ids_sha256'] for c in protocol_checks['checks']}
    for (shots, seed), record in zip(GRID, records):
        validate_tfidf(record)
        meta, samples = record['metadata'], record['samples']
        require((meta['shots'], meta['seed']) == (shots, seed), 'Reference identity mismatch')
        require(meta['configuration']['logistic_regression'] == LOGISTIC, 'Reference LR changed')
        require(sha256(json_bytes(samples)) == meta['samples_sha256'], 'Reference sample hash mismatch')
        require(sha256(json_bytes(samples['train_ids'])) == hashes[(shots, seed)],
                'Reference IDs differ from the preexisting protocol check')
    return records


def resolve_rows(records, manifest, pool, validation):
    """Resolve recorded IDs only. Never call few_shot or prepare."""
    by_id = {r['id']: r for r in pool}
    trains = []
    for record in records:
        meta, samples = record['metadata'], record['samples']
        require(meta['source'] == manifest['source'], 'Dataset source changed')
        require(meta['split_manifest_sha256'] == sha256(json_bytes(manifest)), 'Split hash changed')
        require(samples['labels'] == manifest['labels'], 'Label order changed')
        require(samples['validation_ids'] == [r['id'] for r in validation], 'Validation IDs changed')
        require([p['true_label'] for p in record['predictions']] == [r['label'] for r in validation],
                'Validation labels changed')
        require(set(samples['train_ids']) <= by_id.keys(), 'Training ID outside existing pool')
        train = [by_id[i] for i in samples['train_ids']]
        require(Counter(r['label'] for r in train) ==
                {label: samples['shots'] for label in manifest['labels']}, 'Class budget mismatch')
        verify_isolation(train, validation, manifest['sealed_test'])
        trains.append(train)
    return trains


def cache_signature():
    return {'model_id': emb.MODEL_ID, 'revision': emb.MODEL_REVISION,
            'encoding': emb.ENCODING,
            'packages': {p: importlib.metadata.version(p) for p in RUNTIME_PACKAGES}}


def load_features(path):
    with np.load(path, allow_pickle=False) as saved:
        return emb.Features(saved['values'], saved['ids'].tolist(), saved['labels'].tolist(),
                            saved['text_sha256'].tolist())


def select_features(features, rows):
    require(len(set(features.ids)) == len(features.ids), 'Duplicate cache IDs')
    positions = {row_id: i for i, row_id in enumerate(features.ids)}
    require(all(r['id'] in positions for r in rows), 'Missing cache IDs')
    index = [positions[r['id']] for r in rows]
    selected = emb.Features(features.values[index], [features.ids[i] for i in index],
                            [features.labels[i] for i in index], [features.text_sha256[i] for i in index])
    selected.aligned(rows)
    return selected


def load_cache(path, rows):
    meta = read_json(path / 'metadata.json')
    require(meta['status'] == 'completed', 'Incomplete embedding cache')
    require(not meta['warnings'], 'Embedding cache warnings require review')
    frozen = meta['frozen_encoder_check']
    require(frozen['trainable_parameters'] == 0 and not frozen['gradients_present']
            and frozen['evaluation_mode'] and frozen['inference_mode'], 'Unfrozen embedding cache')
    require(meta['signature'] == cache_signature(), 'Cache revision/settings/runtime changed')
    require(sha256((path / 'features.npz').read_bytes()) == meta['features_sha256'], 'Cache checksum mismatch')
    features = load_features(path / 'features.npz')
    features.aligned(rows)
    require(np.allclose(np.linalg.norm(features.values, axis=1), 1, atol=1e-5), 'Cache normalization changed')
    require(meta['state_sha256_before'] == meta['state_sha256_after'], 'Encoder state changed')
    features.values.setflags(write=False)
    return meta, features


def build_cache(path, groups, reuse_smoke=None):
    """Groups preserve EXP-003 batching for train/validation; encode the missing union once."""
    path.mkdir(parents=True, exist_ok=False)
    meta = {'status': 'running', 'started_utc': now(), 'signature': cache_signature(),
            'encoding_seed': 11, 'groups': [], 'warnings': [],
            'timing_note': 'Encoding wall times only; not end-to-end inference or repeated latency benchmarks'}
    write_json(path / 'metadata.json', meta)
    try:
        meta['code'] = code_provenance(path)
        meta['environment'] = emb.environment()
        smoke = read_json(SMOKE_RECORD / 'metadata.json')
        require(smoke['configuration'] == CONFIG, 'EXP-003 configuration changed')
        require(smoke['embedding_source']['revision'] == emb.MODEL_REVISION, 'EXP-003 revision changed')
        require(all(smoke['environment']['packages'][p] == v for p, v in meta['signature']['packages'].items()),
                'Runtime differs from EXP-003')
        if reuse_smoke is not None:
            emb.verify_run(Path('data/raw/banking77'), DEFAULT_MANIFEST, reuse_smoke)
            require(read_json(reuse_smoke / 'metadata.json') == smoke, 'Unrecognized smoke cache metadata')
        torch.set_num_threads(1)
        torch.manual_seed(11)
        torch.use_deterministic_algorithms(True)
        chunks = []
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            encoder, meta['embedding_source'] = emb.load_encoder(Path('.cache/huggingface/hub'))
            require(meta['embedding_source']['files_sha256'] == smoke['embedding_source']['files_sha256'],
                    'Model files differ from EXP-003')
            emb.freeze_encoder(encoder)
            meta['state_sha256_before'] = emb.state_digest(encoder)
            require(meta['state_sha256_before'] == smoke['frozen_encoder_check']['state_sha256_before'],
                    'Loaded encoder state differs from EXP-003')
            for group_name, rows in groups:
                if reuse_smoke is not None and group_name in ('train', 'validation'):
                    file = reuse_smoke / f'{group_name}_embeddings.npz'
                    features = load_features(file)
                    features.aligned(rows)
                    timing = {'cache_hit': True, 'encoded_rows': 0, 'encoding_seconds': None,
                              'source_path': str(file), 'source_sha256': sha256(file.read_bytes())}
                else:
                    features, seconds = emb.encode_rows(encoder, rows)
                    timing = {'cache_hit': False, 'encoded_rows': len(rows), 'encoding_seconds': seconds}
                chunks.append(features.values)
                meta['groups'].append({'name': group_name, 'rows': len(rows),
                                       'ids_sha256': sha256(json_bytes([r['id'] for r in rows])), **timing})
            emb.assert_frozen(encoder)
            meta['state_sha256_after'] = emb.state_digest(encoder)
            require(meta['state_sha256_before'] == meta['state_sha256_after'], 'Frozen encoder mutated')
            meta['frozen_encoder_check'] = {'trainable_parameters': 0, 'gradients_present': False,
                                            'evaluation_mode': True, 'inference_mode': True,
                                            'torch_threads': torch.get_num_threads(),
                                            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
            meta['environment']['threadpools'] = threadpool_info()
            meta['warnings'] = [f'{w.category.__name__}: {w.message}' for w in caught]
        rows = [r for _, group in groups for r in group]
        features = emb.Features(np.concatenate(chunks), [r['id'] for r in rows],
                                [r['label'] for r in rows], [sha256(r['text'].encode()) for r in rows])
        features.aligned(rows)
        features.save(path / 'features.npz')
        meta['features_sha256'] = sha256((path / 'features.npz').read_bytes())
        meta['rows'] = len(rows)
        meta['status'] = 'completed'
    except Exception as exc:
        meta['status'] = 'failed'
        meta['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        meta['finished_utc'] = now()
        write_json(path / 'metadata.json', meta)


def fit_run(output, reference, train, validation, cache_path, cache_meta, features, role):
    output.mkdir(parents=True, exist_ok=False)
    source_meta, samples = reference['metadata'], reference['samples']
    meta = {'schema_version': 1, 'run_id': output.name, 'status': 'running', 'role': role,
            'started_utc': now(), 'protocol_id': PROTOCOL_ID, 'evaluation_split': 'validation',
            'shots': samples['shots'], 'seed': samples['seed'], 'configuration': CONFIG,
            'source': source_meta['source'], 'split_manifest_sha256': source_meta['split_manifest_sha256'],
            'label_budgets': source_meta['label_budgets'], 'samples_sha256': sha256(json_bytes(samples)),
            'reference_run_id': source_meta['run_id'], 'reference_sha256': sha256(json_bytes(reference)),
            'embedding_cache': {'path': str(cache_path), 'metadata_sha256': sha256((cache_path / 'metadata.json').read_bytes()),
                                'features_sha256': cache_meta['features_sha256']},
            'embedding_source': cache_meta['embedding_source'],
            'environment': cache_meta['environment'], 'frozen_encoder_check': cache_meta['frozen_encoder_check'],
            'command': shlex.join([sys.executable, '-m', 'baseline.minilm_curve', *sys.argv[1:]]),
            'test_access': 'byte-integrity checks only; no test rows parsed', 'warnings': []}
    write_json(output / 'metadata.json', meta)
    try:
        meta['code'] = code_provenance(output)
        write_json(output / 'samples.json', samples)
        train_features = select_features(features, train)
        val_features = select_features(features, validation)
        x_train, y_train = train_features.aligned(train)
        x_val, truth = val_features.aligned(validation)
        model = LogisticRegression(**LOGISTIC)
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            start = time.perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - start
            meta['warnings'] = [f'{w.category.__name__}: {w.message}' for w in caught]
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError('Logistic regression did not converge')
            start = time.perf_counter()
            predicted = model.predict(x_val).tolist()
            predict_seconds = time.perf_counter() - start
            labels = samples['labels']
            order = [model.classes_.tolist().index(label) for label in labels]
            probabilities = model.predict_proba(x_val)[:, order].tolist()
            meta['warnings'] = [f'{w.category.__name__}: {w.message}' for w in caught]
        write_json(output / 'metrics.json', {'evaluation_split': 'validation',
                   emb.METRIC_KEY: classification_metrics(truth, predicted, labels)})
        write_json(output / 'predictions.json', [{'id': row['id'], 'true_label': truth[i],
                   'predicted_label': predicted[i]} for i, row in enumerate(validation)])
        write_json(output / 'probabilities.json', {'labels': labels, 'validation_ids': samples['validation_ids'],
                                                   'probabilities': probabilities})
        joblib.dump(model, output / 'classifier.joblib')
        meta['optimizer_iterations'] = model.n_iter_.tolist()
        meta['resources'] = {'classifier_fit_seconds': fit_seconds, 'classifier_predict_seconds': predict_seconds,
                             'embedding_cache_used': True, 'end_to_end_inference_seconds': None,
                             'timing_note': 'One classifier-only pass on cached features; excludes encoding/cache I/O, not end-to-end speed',
                             'classifier_bytes': (output / 'classifier.joblib').stat().st_size}
        meta['artifacts_sha256'] = {p.name: sha256(p.read_bytes()) for p in output.iterdir()
                                    if p.is_file() and p.name != 'metadata.json'}
        meta['status'] = 'completed'
    except Exception as exc:
        meta['status'] = 'failed'
        meta['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        meta['finished_utc'] = now()
        write_json(output / 'metadata.json', meta)


def validate_record(record, reference):
    meta, samples = record['metadata'], record['samples']
    require(meta['status'] == 'completed', 'Incomplete run')
    require(meta['evaluation_split'] == record['metrics']['evaluation_split'] == 'validation', 'Test evaluation prohibited')
    require(meta['protocol_id'] == PROTOCOL_ID, 'Protocol changed')
    require(not meta['warnings'], 'Run warnings require review')
    require((meta['shots'], meta['seed']) == (samples['shots'], samples['seed']), 'Wrong seed/regime')
    require(samples == reference['samples'], 'Existing train/validation IDs not reused')
    require(meta['samples_sha256'] == sha256(json_bytes(samples)), 'Sample hash mismatch')
    require(meta['reference_sha256'] == sha256(json_bytes(reference)), 'Reference changed')
    require(meta['configuration'] == CONFIG, 'Model configuration changed')
    require(meta['embedding_source']['model_id'] == emb.MODEL_ID
            and meta['embedding_source']['revision'] == emb.MODEL_REVISION, 'Encoder revision changed')
    frozen = meta['frozen_encoder_check']
    require(frozen['trainable_parameters'] == 0 and not frozen['gradients_present']
            and frozen['evaluation_mode'] and frozen['inference_mode'], 'Unfrozen encoder')
    for key in ('source', 'split_manifest_sha256', 'label_budgets'):
        require(meta[key] == reference['metadata'][key], f'Reference {key} changed')
    predictions = record['predictions']
    require([(p['id'], p['true_label']) for p in predictions] ==
            [(p['id'], p['true_label']) for p in reference['predictions']], 'Prediction rows/labels changed')
    score = classification_metrics([p['true_label'] for p in predictions],
                                    [p['predicted_label'] for p in predictions], samples['labels'])
    require(score == record['metrics'][emb.METRIC_KEY], 'Saved metrics differ from predictions')


def verify_run(path, reference, train, validation, cache_path, cache_meta, features):
    record = {k: read_json(path / f'{k}.json') for k in ('metadata', 'samples', 'metrics', 'predictions')}
    validate_record(record, reference)
    meta = record['metadata']
    require(meta['embedding_cache']['metadata_sha256'] == sha256((cache_path / 'metadata.json').read_bytes()), 'Cache metadata changed')
    require(meta['embedding_cache']['features_sha256'] == cache_meta['features_sha256'], 'Cache changed')
    for name, digest in meta['artifacts_sha256'].items():
        require(sha256((path / name).read_bytes()) == digest, f'Artifact checksum mismatch: {name}')
    for name, digest in meta['code']['source_files_sha256'].items():
        require(sha256((path / 'source' / name).read_bytes()) == digest, f'Source checksum mismatch: {name}')
    select_features(features, train).aligned(train)
    values, truth = select_features(features, validation).aligned(validation)
    model = joblib.load(path / 'classifier.joblib')
    require({k: model.get_params()[k] for k in LOGISTIC} == LOGISTIC, 'Classifier settings changed')
    with threadpool_limits(limits=1):
        require(model.predict(values).tolist() == [p['predicted_label'] for p in record['predictions']], 'Prediction replay differs')
        labels = record['samples']['labels']
        order = [model.classes_.tolist().index(label) for label in labels]
        probs = read_json(path / 'probabilities.json')
        require(probs['labels'] == labels and probs['validation_ids'] == record['samples']['validation_ids'], 'Probability order changed')
        np.testing.assert_allclose(probs['probabilities'], model.predict_proba(values)[:, order], atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.sum(probs['probabilities'], axis=1), 1, atol=1e-12, rtol=0)
    return record, model


def aggregate(records, references):
    require(len(records) == len(GRID) == len(references), 'Expected exactly 15 runs and references')
    ref_by_pair = {(r['metadata']['shots'], r['metadata']['seed']): r for r in references}
    by_pair = {(r['metadata']['shots'], r['metadata']['seed']): r for r in records}
    require(set(by_pair) == set(ref_by_pair) == set(GRID), 'Duplicate or missing seed/regime')
    fixed = references[0]['samples']
    for pair in GRID:
        ref, record = ref_by_pair[pair], by_pair[pair]
        validate_tfidf(ref)
        validate_record(record, ref)
        require(record['metadata']['role'] == 'primary', 'Reproduction included in aggregate')
        require(ref['samples']['validation_ids'] == fixed['validation_ids'] and
                ref['samples']['labels'] == fixed['labels'], 'Mixed validation sets')
    regimes = []
    for shots in SHOTS:
        regime = {'shots': shots, 'seeds': list(SEEDS), 'training_examples_per_run': 77 * shots,
                  'validation_examples': 770}
        for name, key, source in [('minilm', emb.METRIC_KEY, by_pair), ('tfidf', TFIDF_MODEL, ref_by_pair)]:
            regime[name] = {}
            for metric in ('accuracy', 'macro_f1'):
                values = {str(s): source[(shots, s)]['metrics'][key][metric] for s in SEEDS}
                regime[name][metric] = {'mean': statistics.mean(values.values()),
                                        'sample_std': statistics.stdev(values.values()),
                                        'min': min(values.values()), 'max': max(values.values()), 'by_seed': values}
        gains = {str(s): 100 * (regime['minilm']['macro_f1']['by_seed'][str(s)] -
                               regime['tfidf']['macro_f1']['by_seed'][str(s)]) for s in SEEDS}
        regime['macro_f1_improvement_pp'] = {'by_seed': gains, 'mean': statistics.mean(gains.values()),
                                            'sample_std': statistics.stdev(gains.values())}
        regimes.append(regime)
    train_ids = set().union(*(set(r['samples']['train_ids']) for r in records))
    return {'study_id': 'EXP-004', 'protocol_id': PROTOCOL_ID, 'evaluation_split': 'validation',
            'primary_run_count': 15, 'regimes': regimes, 'metric_scale': 'fraction 0..1; gains in percentage points',
            'standard_deviation': 'sample SD over five training seeds (ddof=1), not a confidence interval',
            'unique_training_examples': len(train_ids), 'additional_validation_labels': 770,
            'unique_training_plus_validation_labels': len(train_ids) + 770,
            'source_training_labels_read_for_audit': 10003, 'test_evaluation_labels': 0,
            'validation_ids_sha256': sha256(json_bytes(fixed['validation_ids']))}


def save_summary(records, references, output):
    summary = aggregate(records, references)
    write_json(output / 'summary.json', summary)
    with (output / 'per_seed.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['shots', 'seed', 'minilm_accuracy', 'minilm_macro_f1', 'tfidf_accuracy', 'tfidf_macro_f1', 'macro_f1_gain_pp'])
        for regime in summary['regimes']:
            for seed in SEEDS:
                writer.writerow([regime['shots'], seed, *[regime[name][metric]['by_seed'][str(seed)]
                                  for name in ('minilm', 'tfidf') for metric in ('accuracy', 'macro_f1')],
                                 regime['macro_f1_improvement_pp']['by_seed'][str(seed)]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for ax, metric in zip(axes, ('macro_f1', 'accuracy')):
        for name, label, color in [('minilm', 'Frozen MiniLM + LR', '#175b9c'), ('tfidf', 'TF-IDF + LR', '#af5832')]:
            regimes = summary['regimes']
            ax.errorbar(SHOTS, [r[name][metric]['mean'] for r in regimes],
                        yerr=[r[name][metric]['sample_std'] for r in regimes], label=label,
                        fmt='o-', capsize=4, color=color)
            for regime in regimes:
                ax.scatter([regime['shots'] + (i - 2) * .13 for i in range(5)],
                           list(regime[name][metric]['by_seed'].values()), s=15, alpha=.5, color=color)
        ax.set(xlabel='Training examples per class', ylabel=f'Validation {metric.replace("_", " ")}',
               xticks=SHOTS, ylim=(0, 1), xlim=(3, 22))
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='lower right')
    fig.suptitle('BANKING77 · same 770 validation examples\nFive seeds · mean ± sample SD (not a confidence interval)')
    fig.savefig(output / 'learning_curve.png', dpi=180)
    plt.close(fig)
    return summary


def execute(output, reuse_smoke=None):
    output.mkdir(parents=True, exist_ok=False)
    study = {'study_id': 'EXP-004', 'status': 'running', 'started_utc': now(), 'completed_runs': [],
             'command': shlex.join([sys.executable, '-m', 'baseline.minilm_curve', *sys.argv[1:]])}
    write_json(output / 'study.json', study)
    try:
        refs = read_references()
        manifest, pool, validation = load_development(Path('data/raw/banking77'), DEFAULT_MANIFEST)
        trains = resolve_rows(refs, manifest, pool, validation)
        unique_train = {r['id']: r for train in trains for r in train}
        base_ids = {r['id'] for r in trains[0]}
        rest = [unique_train[i] for i in sorted(unique_train.keys() - base_ids)]
        groups = [('train', trains[0]), ('validation', validation), ('remaining_training_union', rest)]
        rows = [r for _, group in groups for r in group]
        collections, caches = [], []
        for role in ('primary', 'reproduction'):
            cache_path = output / f'{role}-cache'
            build_cache(cache_path, groups, reuse_smoke if role == 'primary' else None)
            cache_meta, features = load_cache(cache_path, rows)
            caches.append((cache_meta, features))
            folder = output / ('runs' if role == 'primary' else 'reproductions')
            folder.mkdir()
            records = []
            for (n, seed), ref, train in zip(GRID, refs, trains):
                name = f'exp004-minilm-v2-n{n}-s{seed}' + ('-reproduction' if role == 'reproduction' else '')
                fit_run(folder / name, ref, train, validation, cache_path, cache_meta, features, role)
                record, _ = verify_run(folder / name, ref, train, validation, cache_path, cache_meta, features)
                records.append(record)
                study['completed_runs'].append(name)
                write_json(output / 'study.json', study)
                score = record['metrics'][emb.METRIC_KEY]
                print(f'{name}: accuracy={score["accuracy"]:.6f}, macro_f1={score["macro_f1"]:.6f}', flush=True)
            collections.append(records)
        primary, repeated = collections
        feature_diff = float(np.max(np.abs(caches[0][1].values - caches[1][1].values)))
        require(np.allclose(caches[0][1].values, caches[1][1].values, atol=1e-6, rtol=0), 'Recomputed embeddings differ beyond 1e-6')
        checks = []
        for original, repeat in zip(primary, repeated):
            name, other = original['metadata']['run_id'], repeat['metadata']['run_id']
            require(original['samples'] == repeat['samples'], 'Reproduction sample mismatch')
            require(original['predictions'] == repeat['predictions'], 'Reproduction predictions differ')
            require(original['metrics'] == repeat['metrics'], 'Reproduction scores differ')
            a = joblib.load(output / 'runs' / name / 'classifier.joblib')
            b = joblib.load(output / 'reproductions' / other / 'classifier.joblib')
            delta = float(max(np.max(np.abs(a.coef_ - b.coef_)), np.max(np.abs(a.intercept_ - b.intercept_))))
            checks.append({'run_id': name, 'reproduction_id': other, 'identical_samples_predictions_metrics': True,
                           'classifier_parameter_max_abs_difference': delta})
        smoke_metrics = read_json(SMOKE_RECORD / 'metrics.json')
        smoke_predictions = read_json(SMOKE_RECORD / 'predictions.json')
        report = {'primary_runs': 15, 'independent_refits': 15, 'checks': checks,
                  'independently_reencoded_rows': len(rows), 'embedding_max_abs_difference': feature_diff,
                  'embedding_comparison_atol': 1e-6, 'embedding_comparison_rtol': 0,
                  'primary_reused_smoke_rows': sum(g['rows'] for g in caches[0][0]['groups'] if g['cache_hit']),
                  'exp003_identical_predictions_and_metrics': primary[0]['predictions'] == smoke_predictions and
                     primary[0]['metrics'][emb.METRIC_KEY] == smoke_metrics[emb.METRIC_KEY],
                  'all_artifact_source_cache_hashes_verified': True, 'all_existing_sample_ids_verified': True,
                  'no_test_rows_parsed': True, 'official_test_sha256': manifest['source']['files']['test.csv']['sha256'],
                  'split_manifest_sha256': sha256(json_bytes(manifest)),
                  'scope': 'Independent local re-encoding and refits; not external or cross-platform replication'}
        write_json(output / 'verification.json', report)
        save_summary(primary, refs, output)
        for folder, records in [('records', primary), ('reproduction-records', repeated)]:
            target = output / folder
            target.mkdir()
            for record in records:
                (target / f'{record["metadata"]["run_id"]}.json').write_text(
                    json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
        study['status'] = 'completed'
    except Exception as exc:
        study['status'] = 'failed'
        study['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        study['finished_utc'] = now()
        write_json(output / 'study.json', study)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-smoke', type=Path, help='Validated EXP-003 artifact directory to reuse')
    parser.add_argument('--records-dir', type=Path, help='Only reaggregate compact records; no fitting or encoding')
    args = parser.parse_args()
    if args.records_dir:
        records = [read_json(p) for p in sorted(args.records_dir.glob('*.json'))]
        args.output.mkdir(parents=True, exist_ok=False)
        save_summary(records, read_references(), args.output)
    else:
        execute(args.output, args.reuse_smoke)


if __name__ == '__main__':
    main()
