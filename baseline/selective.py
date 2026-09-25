"""EXP-005: exploratory selective prediction from saved validation probabilities only."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import statistics
import sys

import numpy as np

from .data import PROTOCOL_ID, SEEDS, SHOTS, json_bytes, read_json, sha256, write_json
from .experiment import classification_metrics, code_provenance

TARGETS = (.25, .50, .75, .90, 1.0)
GRID = {(n, s) for n in SHOTS for s in SEEDS}
STUDY = Path('experiments/exp004-minilm-learning-curve')
MODEL = 'frozen_minilm_logistic_regression'
ORDER_RULE = 'descending exact max class probability; ascending SHA256(UTF-8 row ID), then row ID'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def confidence_order(ids, confidence):
    require(len(ids) > 0 and len(ids) == len(confidence), 'Nonempty aligned IDs and confidence required')
    require(all(isinstance(i, str) for i in ids) and len(set(ids)) == len(ids), 'Unique string IDs required')
    require(all(math.isfinite(c) and 0 <= c <= 1 for c in confidence), 'Invalid confidence score')
    # No true or predicted labels are passed to this function.
    return sorted(range(len(ids)), key=lambda i: (-confidence[i], sha256(ids[i].encode()), ids[i]))


def probability_rows(record, probabilities):
    samples, predictions = record['samples'], record['predictions']
    labels, ids = samples['labels'], samples['validation_ids']
    require(labels and len(set(labels)) == len(labels), 'Unique class labels required')
    require(ids and len(set(ids)) == len(ids), 'Unique validation IDs required')
    require(probabilities['labels'] == labels, 'Probability class order differs')
    require(probabilities['validation_ids'] == ids, 'Probability row order differs')
    require([p['id'] for p in predictions] == ids, 'Prediction row order differs')
    values = np.asarray(probabilities['probabilities'], dtype=float)
    require(values.shape == (len(ids), len(labels)), 'Probability shape differs')
    require(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all(), 'Invalid probabilities')
    require(np.allclose(values.sum(axis=1), 1, atol=1e-12, rtol=0), 'Probabilities must sum to one')
    maximum = values.max(axis=1)
    label_index = {label: i for i, label in enumerate(labels)}
    for i, prediction in enumerate(predictions):
        require(prediction['true_label'] in label_index and prediction['predicted_label'] in label_index,
                'Unknown prediction/true label')
        # Equal class-probability maxima are allowed; class order need not be sklearn's order.
        require(values[i, label_index[prediction['predicted_label']]] == maximum[i],
                'Predicted label is not a maximum-probability class')
    computed = classification_metrics([p['true_label'] for p in predictions],
                                      [p['predicted_label'] for p in predictions], labels)
    require(record['metrics'][MODEL] == computed, 'EXP-004 metrics differ from predictions')
    return [{**p, 'confidence': float(maximum[i])} for i, p in enumerate(predictions)]


def selective_curve(rows, labels, full_accuracy):
    require(labels and len(set(labels)) == len(labels), 'Unique labels required')
    order = confidence_order([r['id'] for r in rows], [r['confidence'] for r in rows])
    require(all(r['true_label'] in labels and r['predicted_label'] in labels for r in rows), 'Unknown label')
    ranked = [dict(rows[i]) for i in order]
    n = len(ranked)
    errors = 0
    points = [{'accepted_count': 0, 'correct_count': 0, 'error_count': 0,
               'coverage': 0.0, 'accepted_accuracy': None, 'selective_risk': None}]
    for k, row in enumerate(ranked, 1):
        errors += row['predicted_label'] != row['true_label']
        points.append({'accepted_count': k, 'correct_count': k - errors, 'error_count': errors,
                       'coverage': k / n, 'accepted_accuracy': (k - errors) / k,
                       'selective_risk': errors / k})
    require(points[-1]['accepted_accuracy'] == full_accuracy, '100% accuracy differs from EXP-004')
    landmarks = []
    for target in TARGETS:
        k = math.ceil(target * n)
        accepted = ranked[:k]
        counts = Counter(r['true_label'] for r in accepted)
        class_errors = Counter(r['true_label'] for r in accepted if r['true_label'] != r['predicted_label'])
        split_tie = k < n and ranked[k - 1]['confidence'] == ranked[k]['confidence']
        landmarks.append({'target_coverage': target, **points[k],
                          'per_class_acceptance': {label: counts[label] for label in labels},
                          'per_class_errors': {label: class_errors[label] for label in labels},
                          'zero_acceptance_classes': [label for label in labels if counts[label] == 0],
                          'boundary_splits_exact_tie': split_tie})
    tied = Counter(r['confidence'] for r in ranked)
    return {'labels': list(labels), 'n_validation': n, 'full_accuracy': full_accuracy,
            'ranked_rows': ranked, 'curve': points, 'landmarks': landmarks,
            'confidence_score': 'maximum predicted class probability; UNCALIBRATED',
            'ordering': ORDER_RULE, 'coverage_count_rule': 'ceil(target coverage * n_validation)',
            'exact_tie_groups': sum(v > 1 for v in tied.values()),
            'requests_in_exact_ties': sum(v for v in tied.values() if v > 1)}


def stats(values):
    require(len(values) == len(SEEDS), 'Five seed values required')
    result = {'by_seed': {str(s): v for s, v in zip(SEEDS, values)}}
    if all(v is None for v in values):
        return {**result, 'mean': None, 'sample_std': None, 'min': None, 'max': None}
    require(all(v is not None for v in values), 'Mixed defined and undefined statistics')
    return {**result, 'mean': statistics.mean(values), 'sample_std': statistics.stdev(values),
            'min': min(values), 'max': max(values)}


def aggregate(runs):
    require(len(runs) == 15, 'Expected exactly 15 primary runs')
    by_pair = {(r['shots'], r['seed']): r for r in runs}
    require(set(by_pair) == GRID, 'Duplicate or missing seed/regime')
    fixed = by_pair[(5, 11)]
    fixed_truth = {r['id']: r['true_label'] for r in fixed['ranked_rows']}
    for run in runs:
        require(run['labels'] == fixed['labels'], 'Mixed class order')
        require({r['id']: r['true_label'] for r in run['ranked_rows']} == fixed_truth, 'Mixed validation IDs/labels')
        rebuilt = selective_curve(run['ranked_rows'], run['labels'], run['full_accuracy'])
        for key in ('curve', 'landmarks', 'ranked_rows'):
            require(run[key] == rebuilt[key], f'Corrupted {key}')
    metrics = ('coverage', 'accepted_count', 'correct_count', 'error_count', 'accepted_accuracy', 'selective_risk')
    regimes = []
    for n in SHOTS:
        selected = [by_pair[(n, s)] for s in SEEDS]
        landmarks = []
        for i, target in enumerate(TARGETS):
            values = [r['landmarks'][i] for r in selected]
            point = {'target_coverage': target,
                     **{metric: stats([v[metric] for v in values]) for metric in metrics},
                     'zero_acceptance_class_count': stats([len(v['zero_acceptance_classes']) for v in values])}
            point['per_class_acceptance'] = {
                label: {**stats([v['per_class_acceptance'][label] for v in values]),
                        'zero_acceptance_seed_count': sum(v['per_class_acceptance'][label] == 0 for v in values)}
                for label in fixed['labels']}
            point['never_accepted_classes_across_five_seeds'] = [
                label for label, counts in point['per_class_acceptance'].items() if counts['max'] == 0]
            landmarks.append(point)
        curve = []
        for k in range(len(fixed_truth) + 1):
            curve.append({'accepted_count': k, 'coverage': k / len(fixed_truth),
                          'accepted_accuracy': stats([r['curve'][k]['accepted_accuracy'] for r in selected]),
                          'selective_risk': stats([r['curve'][k]['selective_risk'] for r in selected])})
        regimes.append({'shots': n, 'seeds': list(SEEDS), 'landmarks': landmarks, 'curve': curve})
    return {'study_id': 'EXP-005', 'protocol_id': PROTOCOL_ID, 'evaluation_split': 'validation',
            'purpose': 'exploratory confidence-ranking diagnostic; not calibration or a deployment policy',
            'confidence_score': fixed['confidence_score'], 'ordering': ORDER_RULE,
            'regimes': regimes, 'primary_run_count': 15, 'unique_validation_examples': len(fixed_truth),
            'additional_validation_label_budget': len(fixed_truth), 'new_labels': 0,
            'sample_sd_definition': 'ddof=1 over five models on the SAME validation examples, not a confidence interval',
            'zero_acceptance_definition': 'accuracy and risk undefined (JSON null)',
            'per_class_definition': 'counts by true intent; labels used only after ranking',
            'model_fits': 0, 'encoder_executions': 0, 'calibration_fits': 0, 'official_test_access': False}


def load_verified_inputs(artifacts):
    """Hash-verified archived probabilities; never loads a model, encoder, or raw dataset."""
    from .minilm_curve import read_references, validate_record
    refs = read_references()
    audit = read_json(STUDY / 'analysis_audit.json')
    cache = read_json(STUDY / 'primary-cache.json')
    cache_path = artifacts / 'primary-cache'
    require(read_json(cache_path / 'metadata.json') == cache, 'Archived cache metadata changed')
    require(sha256((cache_path / 'features.npz').read_bytes()) == cache['features_sha256'], 'Archived feature cache changed')
    for name, digest in cache['code']['source_files_sha256'].items():
        require(sha256((cache_path / 'source' / name).read_bytes()) == digest, 'Cache source snapshot changed')
    inputs = []
    for ref in refs:
        n, s = ref['samples']['shots'], ref['samples']['seed']
        name = f'exp004-minilm-v2-n{n}-s{s}'
        relative = f'runs/{name}.json'
        record_path = STUDY / relative
        require(sha256(record_path.read_bytes()) == audit['compact_records_sha256'][relative], 'EXP-004 record hash changed')
        record = read_json(record_path)
        validate_record(record, ref)
        meta = record['metadata']
        require(meta['role'] == 'primary' and meta['run_id'] == name, 'Only expected EXP-004 primary runs allowed')
        path = artifacts / 'runs' / name
        for field in ('metadata', 'samples', 'metrics', 'predictions'):
            require(read_json(path / f'{field}.json') == record[field], f'Archived {field} differs from versioned EXP-004')
        for file, digest in meta['artifacts_sha256'].items():
            require(sha256((path / file).read_bytes()) == digest, f'Artifact checksum mismatch: {name}/{file}')
        for file, digest in meta['code']['source_files_sha256'].items():
            require(sha256((path / 'source' / file).read_bytes()) == digest, f'Source checksum mismatch: {name}/{file}')
        require(meta['embedding_cache']['metadata_sha256'] == sha256((cache_path / 'metadata.json').read_bytes()), 'Cache metadata hash changed')
        require(meta['embedding_cache']['features_sha256'] == cache['features_sha256'], 'Cache source differs')
        probabilities = read_json(path / 'probabilities.json')
        rows = probability_rows(record, probabilities)
        inputs.append({'run_id': name, 'shots': n, 'seed': s, 'labels': record['samples']['labels'],
                       'rows': rows, 'full_accuracy': record['metrics'][MODEL]['accuracy'],
                       'provenance': {'exp004_record_sha256': sha256(record_path.read_bytes()),
                                      'probabilities_sha256': sha256((path / 'probabilities.json').read_bytes()),
                                      'classifier_sha256': meta['artifacts_sha256']['classifier.joblib'],
                                      'samples_sha256': meta['samples_sha256'],
                                      'split_manifest_sha256': meta['split_manifest_sha256'],
                                      'cache_metadata_sha256': meta['embedding_cache']['metadata_sha256'],
                                      'cache_features_sha256': cache['features_sha256'],
                                      'dataset_revision': meta['source']['revision'],
                                      'encoder_revision': meta['embedding_source']['revision'],
                                      'exp004_code': meta['code']['git_head'],
                                      'probabilities_regenerated': False,
                                      'checks': ['versioned primary record hashes', 'source/artifact/cache hashes',
                                                 'original train/validation IDs', 'probability class/row order and simplex',
                                                 'predicted class attains max probability', 'complete metrics from predictions']}})
    return inputs


def compact(run):
    # The full prefix curve is exactly reconstructible from ordered correctness flags.
    return {k: v for k, v in run.items() if k != 'curve'}


def restore(record):
    rebuilt = selective_curve(record['ranked_rows'], record['labels'], record['full_accuracy'])
    for key, value in rebuilt.items():
        if key != 'curve':
            require(record[key] == value, f'Compact record {key} changed')
    return {**record, 'curve': rebuilt['curve']}


def save_outputs(runs, output):
    summary = aggregate(runs)
    # Full curves are kept locally; compact ranked records regenerate every point.
    write_json(output / 'full_curves.json', {str(r['shots']): r['curve'] for r in summary['regimes']})
    small = {**summary, 'regimes': [{k: v for k, v in r.items() if k != 'curve'} for r in summary['regimes']]}
    write_json(output / 'summary.json', small)
    metrics = ('accepted_count', 'error_count', 'correct_count', 'coverage', 'accepted_accuracy', 'selective_risk')
    with (output / 'per_run.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['run_id', 'shots', 'seed', 'target_coverage', *metrics, 'zero_acceptance_classes'])
        for r in sorted(runs, key=lambda r: (r['shots'], r['seed'])):
            for point in r['landmarks']:
                writer.writerow([r['run_id'], r['shots'], r['seed'], point['target_coverage'],
                                 *[point[m] for m in metrics], len(point['zero_acceptance_classes'])])
    with (output / 'per_class_acceptance.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['run_id', 'shots', 'seed', 'target_coverage', 'actual_coverage', 'true_intent',
                         'validation_support', 'accepted_count', 'rejected_count', 'accepted_errors'])
        for r in sorted(runs, key=lambda r: (r['shots'], r['seed'])):
            supports = Counter(row['true_label'] for row in r['ranked_rows'])
            for point in r['landmarks']:
                for label in r['labels']:
                    count = point['per_class_acceptance'][label]
                    writer.writerow([r['run_id'], r['shots'], r['seed'], point['target_coverage'], point['coverage'],
                                     label, supports[label], count, supports[label] - count, point['per_class_errors'][label]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout='constrained', sharey=True)
    highest = max(max(point['selective_risk']['max'],
                      point['selective_risk']['mean'] + point['selective_risk']['sample_std'])
                  for regime in summary['regimes'] for point in regime['curve'][1:])
    risk_upper = min(1.0, max(.35, highest * 1.05))
    for ax, regime, color in zip(axes, summary['regimes'], ('#2e6c9b', '#2b7e55', '#9c5035')):
        curve = regime['curve'][1:]  # Risk is undefined at zero coverage: do not plot it as zero.
        x = np.array([p['coverage'] for p in curve])
        y = np.array([p['selective_risk']['mean'] for p in curve])
        sd = np.array([p['selective_risk']['sample_std'] for p in curve])
        for seed in SEEDS:
            ax.plot(x, [p['selective_risk']['by_seed'][str(seed)] for p in curve], color=color, alpha=.2, lw=.7)
        ax.plot(x, y, color=color, lw=2, label='Mean selective risk')
        ax.fill_between(x, np.maximum(0, y - sd), np.minimum(1, y + sd), color=color, alpha=.15, label='±1 sample SD')
        ax.scatter([p['coverage']['mean'] for p in regime['landmarks']],
                   [p['selective_risk']['mean'] for p in regime['landmarks']], color=color, s=25)
        ax.set(title=f"{regime['shots']}-shot · five seeds", xlabel='Actual validation coverage',
               xlim=(0, 1), ylim=(0, risk_upper), xticks=[0, .25, .5, .75, 1])
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Selective risk (accepted error fraction)')
    axes[-1].legend(fontsize=8)
    fig.suptitle('UNCALIBRATED max-probability ranking · same 770 validation requests\nExploratory diagnostic · SD is not a confidence interval', fontsize=12)
    fig.savefig(output / 'risk_coverage.png', dpi=180)
    plt.close(fig)
    return small


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--artifacts', type=Path, help='Existing EXP-004 full artifact root, read-only')
    group.add_argument('--records-dir', type=Path, help='Regenerate analysis from compact EXP-005 records')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {'study_id': 'EXP-005', 'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
            'command': shlex.join([sys.executable, '-m', 'baseline.selective', *sys.argv[1:]]),
            'model_fits': 0, 'calibration_fits': 0, 'encoder_executions': 0, 'official_test_access': False}
    write_json(args.output / 'metadata.json', meta)
    try:
        meta['code'] = code_provenance(args.output)
        if args.artifacts:
            inputs = load_verified_inputs(args.artifacts)
            runs = [{k: v for k, v in item.items() if k != 'rows'} |
                    selective_curve(item['rows'], item['labels'], item['full_accuracy']) for item in inputs]
        else:
            runs = [restore(read_json(p)) for p in sorted(args.records_dir.glob('*.json'))]
        target = args.output / 'runs'
        target.mkdir()
        for run in runs:
            (target / f'{run["run_id"]}.json').write_text(json.dumps(compact(run), sort_keys=True,
                 separators=(',', ':'), allow_nan=False) + '\n')
        summary = save_outputs(runs, args.output)
        meta['run_count'] = len(runs)
        meta['all_full_coverage_accuracies_match_exp004'] = True
        meta['probabilities_regenerated'] = False
        meta['analysis_artifact_sha256'] = {str(p.relative_to(args.output)): sha256(p.read_bytes())
                                            for p in sorted(args.output.rglob('*'))
                                            if p.is_file() and 'source' not in p.relative_to(args.output).parts
                                            and p.name != 'metadata.json'}
        meta['status'] = 'completed'
        for regime in summary['regimes']:
            print(regime['shots'], [(p['coverage']['mean'], p['accepted_accuracy']['mean'],
                                    p['error_count']['mean']) for p in regime['landmarks']])
    except Exception as exc:
        meta['status'] = 'failed'
        meta['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        meta['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / 'metadata.json', meta)


if __name__ == '__main__':
    main()
