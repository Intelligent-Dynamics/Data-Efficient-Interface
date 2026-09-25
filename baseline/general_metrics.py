"""Pure full-workload scoring for one general-model response set and fixed prefixes.

No requests, model execution, fitting, file I/O, or threshold selection occurs here.
Every unresolved response remains an error in the full evaluation denominator.
"""

from collections import Counter
from hashlib import sha256
import math
import statistics


SHOTS = (5, 10, 20)
SEEDS = (11, 22, 33, 44, 55)
TARGETS = (.25, .5, .75, .9, 1.0)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _labels(labels):
    _require(labels and all(isinstance(label, str) and label for label in labels),
             'Nonempty string class labels required')
    _require(len(set(labels)) == len(labels), 'Duplicate class labels')


def general_metrics(truth, predictions, labels):
    """Score all cases and all declared classes; None contributes a false negative.

Undefined per-class precision/recall/F1 uses zero, matching the established
baseline. A failure sentinel is never introduced as an additional macro class.
    """
    _labels(labels)
    _require(len(truth) > 0 and len(truth) == len(predictions), 'Nonempty aligned truth/predictions required')
    allowed = set(labels)
    _require(all(isinstance(value, str) and value in allowed for value in truth), 'Unknown true label')
    _require(all(value is None or isinstance(value, str) and value in allowed for value in predictions),
             'Predictions must be valid labels or None')
    support = Counter(truth)
    predicted = Counter(predictions)
    correct = Counter(t for t, p in zip(truth, predictions) if t == p)
    per_class = {}
    for label in labels:
        tp = correct[label]
        fp = predicted[label] - tp
        fn = support[label] - tp
        per_class[label] = {
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'support': support[label], 'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
        }
    n = len(truth)
    n_correct = sum(correct.values())
    return {
        'labels': list(labels), 'n_examples': n, 'correct_count': n_correct,
        'error_count': n - n_correct, 'unresolved_count': predicted[None],
        'accuracy': n_correct / n, 'macro_f1': statistics.mean(p['f1'] for p in per_class.values()),
        'per_class': per_class,
        'unresolved_policy': 'None counts as incorrect; all cases retained; macro averages declared classes only',
    }


def _unique_rows(rows, name):
    _require(rows, f'Nonempty {name} rows required')
    ids = [row['id'] for row in rows]
    _require(all(isinstance(value, str) and value for value in ids), f'Invalid {name} ID')
    _require(len(set(ids)) == len(ids), f'Duplicate {name} IDs')
    return {row['id']: row for row in rows}


def _validate_selective(run):
    """Validate fixed EXP-005 order/counts, without selecting a different prefix."""
    labels, rows = run['labels'], run['ranked_rows']
    _labels(labels)
    by_id = _unique_rows(rows, 'specialist')
    metrics = general_metrics([r['true_label'] for r in rows], [r['predicted_label'] for r in rows], labels)
    _require(metrics['unresolved_count'] == 0, 'Specialist predictions cannot be unresolved')
    _require(metrics['accuracy'] == run['full_accuracy'], 'Specialist full coverage accuracy changed')
    _require(all(isinstance(r['confidence'], (int, float)) and not isinstance(r['confidence'], bool)
                 and math.isfinite(r['confidence']) and 0 <= r['confidence'] <= 1 for r in rows),
             'Invalid specialist confidence')
    expected_order = sorted(rows, key=lambda r: (-r['confidence'], sha256(r['id'].encode('utf-8')).hexdigest(), r['id']))
    _require([r['id'] for r in rows] == [r['id'] for r in expected_order], 'EXP-005 confidence order changed')
    _require(len(run['landmarks']) == len(TARGETS), 'Exactly five EXP-005 landmarks required')
    n = len(rows)
    for target, point in zip(TARGETS, run['landmarks']):
        k = math.ceil(target * n)
        accepted = rows[:k]
        errors = sum(r['true_label'] != r['predicted_label'] for r in accepted)
        expected = {'target_coverage': target, 'accepted_count': k, 'coverage': k / n,
                    'correct_count': k - errors, 'error_count': errors,
                    'accepted_accuracy': (k - errors) / k, 'selective_risk': errors / k}
        _require(all(point.get(key) == value for key, value in expected.items()), 'EXP-005 landmark counts or metrics changed')
        if 'per_class_acceptance' in point:
            counts = Counter(r['true_label'] for r in accepted)
            _require(point['per_class_acceptance'] == {label: counts[label] for label in labels},
                     'EXP-005 per-class acceptance changed')
    return by_id, metrics


def _stats(values):
    """All entries correspond to the same fixed cases, varying only training seed."""
    _require(len(values) == len(SEEDS), 'Exactly five seed values required')
    mapped = {str(seed): value for seed, value in zip(SEEDS, values)}
    if all(value is None for value in values):
        return {'mean': None, 'sample_std': None, 'by_seed': mapped}
    _require(all(value is not None for value in values), 'Mixed defined/undefined seed metrics')
    return {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values), 'by_seed': mapped}


def evaluate(general_rows, selective_runs):
    """Replay one ID-aligned general response set on all 15 fixed specialist runs.

Status 'ok' requires one valid label. Every other nonempty status is unresolved
regardless of any accompanying label. All IDs must have explicit records; absent
records are rejected instead of being silently dropped or treated as successes.
    """
    _require(len(selective_runs) == 15, 'Exactly 15 specialist runs required')
    by_pair = {(run['shots'], run['seed']): run for run in selective_runs}
    _require(set(by_pair) == {(shots, seed) for shots in SHOTS for seed in SEEDS},
             'Duplicate or missing specialist seed/regime')
    run_ids = [run['run_id'] for run in selective_runs]
    _require(all(isinstance(value, str) and value for value in run_ids) and len(set(run_ids)) == 15,
             'Unique specialist run IDs required')
    fixed = by_pair[(5, 11)]
    fixed_by_id, _ = _validate_selective(fixed)
    labels = fixed['labels']
    truth_by_id = {row_id: row['true_label'] for row_id, row in fixed_by_id.items()}
    general_by_id = _unique_rows(general_rows, 'general')
    _require(set(general_by_id) == set(truth_by_id), 'General IDs must exactly match validation IDs')
    general_predictions = {}
    for row_id, row in general_by_id.items():
        status = row.get('status')
        _require(isinstance(status, str) and status, 'Explicit response status required')
        prediction = row.get('predicted_label')
        if status == 'ok':
            _require(isinstance(prediction, str) and prediction in labels, 'Successful response needs a valid class label')
            general_predictions[row_id] = prediction
        else:
            general_predictions[row_id] = None
    # Canonical output order is label-blind and independent of response arrival order.
    validation_ids = sorted(truth_by_id)
    truth = [truth_by_id[row_id] for row_id in validation_ids]
    baseline = general_metrics(truth, [general_predictions[row_id] for row_id in validation_ids], labels)
    combinations = []
    for shots in SHOTS:
        for seed in SEEDS:
            run = by_pair[(shots, seed)]
            specialist, specialist_metrics = _validate_selective(run)
            _require(run['labels'] == labels, 'Mixed class label order')
            _require({row_id: row['true_label'] for row_id, row in specialist.items()} == truth_by_id,
                     'Mixed validation IDs or true labels')
            n = len(validation_ids)
            points = [{'target_coverage': 0.0, 'accepted_count': 0, 'coverage': 0.0}, *run['landmarks']]
            for point in points:
                k = point['accepted_count']
                accepted_ids = [row['id'] for row in run['ranked_rows'][:k]]
                accepted = set(accepted_ids)
                rejected_ids = [row['id'] for row in run['ranked_rows'][k:]]
                combined_predictions = [specialist[row_id]['predicted_label'] if row_id in accepted
                                        else general_predictions[row_id] for row_id in validation_ids]
                combined = general_metrics(truth, combined_predictions, labels)
                correct_fallback = sum(general_predictions[row_id] == truth_by_id[row_id] for row_id in rejected_ids)
                n_fallback = len(rejected_ids)
                fallback = {
                    'n_examples': n_fallback, 'correct_count': correct_fallback,
                    'error_count': n_fallback - correct_fallback,
                    'unresolved_count': sum(general_predictions[row_id] is None for row_id in rejected_ids),
                    'accuracy': correct_fallback / n_fallback if n_fallback else None,
                }
                _require(k + n_fallback == n, 'Accepted/rejected cases do not partition validation')
                if k == 0:
                    _require(combined == baseline, 'All-fallback result differs from general baseline')
                if k == n:
                    _require(combined == specialist_metrics, 'Full coverage differs from specialist')
                combinations.append({
                    'run_id': run['run_id'], 'shots': shots, 'seed': seed,
                    'target_coverage': point['target_coverage'], 'coverage': k / n,
                    'accepted_count': k, 'rejected_count': n_fallback,
                    'accepted_ids': accepted_ids, 'rejected_ids': rejected_ids,
                    'combined': combined, 'fallback': fallback,
                })
    regimes = []
    for shots in SHOTS:
        points = []
        for target in (0.0, *TARGETS):
            selected = [row for row in combinations if row['shots'] == shots and row['target_coverage'] == target]
            points.append({
                'target_coverage': target,
                'coverage': _stats([row['coverage'] for row in selected]),
                'accepted_count': _stats([row['accepted_count'] for row in selected]),
                'rejected_count': _stats([row['rejected_count'] for row in selected]),
                'accuracy': _stats([row['combined']['accuracy'] for row in selected]),
                'macro_f1': _stats([row['combined']['macro_f1'] for row in selected]),
                'fallback_accuracy': _stats([row['fallback']['accuracy'] for row in selected]),
                'fallback_error_count': _stats([row['fallback']['error_count'] for row in selected]),
                'fallback_unresolved_count': _stats([row['fallback']['unresolved_count'] for row in selected]),
            })
        regimes.append({'shots': shots, 'seeds': list(SEEDS), 'landmarks': points})
    return {
        'evaluation_split': 'validation', 'general': baseline, 'validation_ids': validation_ids,
        'general_response_count': len(general_rows), 'status_counts': dict(sorted(Counter(row['status'] for row in general_rows).items())),
        'combinations': combinations, 'regimes': regimes, 'primary_run_count': 15,
        'unique_validation_examples': len(validation_ids), 'additional_validation_label_budget': len(validation_ids),
        'general_response_sets': 1, 'new_labels': 0,
        'summary_sd': 'sample SD across five specialist training seeds on the same validation cases; not a confidence interval',
    }
