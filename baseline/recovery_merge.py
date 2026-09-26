"""Pure EXP-006 + EXP-006R replay; never execute requests or alter source evidence.

The caller validates source/cache byte hashes and writes a fresh output directory.
This module only merges saved predictions, reuses the unchanged EXP-006 scoring
function, and accounts for every original and recovery attempt separately.
"""
from collections import Counter
from copy import deepcopy
import re

from .general import accounting, final_result, routed_api_costs
from .general_metrics import evaluate


N_VALIDATION = 770
N_ORIGINAL_OK = 718
N_ELIGIBLE = 52
STUDY = 'EXP-006 + EXP-006R recovery'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rows_by_id(rows, count, name):
    _require(isinstance(rows, list) and len(rows) == count, f'{name}: exactly {count} rows required')
    _require(all(isinstance(row, dict) for row in rows), f'{name}: row objects required')
    ids = [row.get('id') for row in rows]
    _require(all(isinstance(rid, str) and rid for rid in ids), f'{name}: nonempty string IDs required')
    _require(len(set(ids)) == count, f'{name}: duplicate IDs')
    for row in rows:
        status = row.get('status')
        _require(isinstance(status, str) and status, f'{name}: explicit status required')
        label = row.get('predicted_label')
        _require(isinstance(label, str) and bool(label) if status == 'ok' else label is None,
                 f'{name}: ok requires a label; unresolved predictions must be None')
    return dict(zip(ids, rows))


def merge_predictions(original_predictions, recovery_predictions, eligible_ids):
    """Replace exactly the 52 original HTTP-429 outcomes, preserving original order.

The 718 original successes are copied value-for-value, including any additional
metadata. Recovery outcomes may remain unresolved; they are never dropped. No
input is mutated and no original successful case can enter the recovery set.
    """
    original = _rows_by_id(original_predictions, N_VALIDATION, 'EXP-006')
    statuses = Counter(row['status'] for row in original_predictions)
    _require(statuses == {'ok': N_ORIGINAL_OK, 'http_429': N_ELIGIBLE},
             'EXP-006 must contain exactly 718 ok and 52 http_429 outcomes')
    _require(isinstance(eligible_ids, list) and len(eligible_ids) == N_ELIGIBLE and
             all(isinstance(rid, str) and rid for rid in eligible_ids) and
             len(set(eligible_ids)) == N_ELIGIBLE, 'Exactly 52 unique eligible IDs required')
    original_unresolved = {rid for rid, row in original.items() if row['status'] == 'http_429'}
    _require(set(eligible_ids) == original_unresolved, 'Only the original 52 HTTP-429 IDs are eligible')
    recovery = _rows_by_id(recovery_predictions, N_ELIGIBLE, 'EXP-006R')
    _require(set(recovery) == original_unresolved,
             'Recovery IDs must exactly match the original unresolved IDs; successful requests cannot be replaced')
    return [deepcopy(recovery[row['id']] if row['id'] in recovery else row) for row in original_predictions]


def _validate_entries(entries, predictions, name):
    _require(isinstance(entries, dict) and set(entries) == {row['id'] for row in predictions},
             f'{name}: cache entries must match prediction IDs exactly')
    for row in predictions:
        entry = entries[row['id']]
        _require(entry.get('id') == row['id'] and entry.get('kind') == 'live',
                 f'{name}: cache kind/ID mismatch')
        attempts = entry.get('attempts')
        _require(isinstance(attempts, list) and attempts, f'{name}: every request must have a recorded attempt')
        _require(final_result(entry) == {'status': row['status'], 'predicted_label': row['predicted_label']},
                 f'{name}: recorded prediction differs from final attempt')
        _require(row['status'] not in ('reserved', 'not_attempted'),
                 f'{name}: a pending request cannot produce final recovery results')


def _hash(value, name):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), f'Invalid {name} SHA-256')
    return value


def offline_evaluation(source, recovery_report, recovery_entries, selective_runs):
    """Evaluate completed saved recovery evidence, including remaining failures.

Input ``source`` is the already-verified ``recovery_source.load_source`` result.
The caller separately validates recovery files/protocol against the frozen
recovery bundle. Both reports must represent live, finished collections; mocks
are exercised in unit tests by synthetic objects explicitly marked as fixtures.
Nothing here loads raw data, calls an endpoint, or writes a file.
    """
    original_report = source['execution']
    original_protocol = _hash(source['manifest']['protocol_sha256'], 'source protocol')
    tree_sha256 = _hash(source['tree_sha256'], 'source tree')
    _require(original_report.get('kind') == 'live' and original_report.get('study_id') == 'EXP-006' and
             original_report.get('status') == 'finished_with_unresolved' and
             original_report.get('protocol_sha256') == original_protocol,
             'Original EXP-006 execution provenance/status mismatch')
    _require(recovery_report.get('kind') == 'live' and recovery_report.get('study_id') == 'EXP-006R',
             'Only saved live EXP-006R reports can be merged')
    _require(recovery_report.get('status') in ('completed', 'finished_with_unresolved'),
             'Recovery collection is not finished; partial/pending reports cannot be evaluated as final')
    _require(recovery_report.get('source_protocol_sha256') == original_protocol and
             recovery_report.get('source_tree_sha256') == tree_sha256,
             'Recovery source provenance mismatch')
    recovery_protocol = _hash(recovery_report.get('recovery_protocol_sha256'), 'recovery protocol')
    eligible = source['eligible_ids']
    _require(recovery_report.get('eligible_ids') == eligible, 'Recovery eligibility/order changed')
    original_predictions = original_report['predictions']
    recovery_predictions = recovery_report['predictions']
    merged = merge_predictions(original_predictions, recovery_predictions, eligible)
    _require(source['validation_ids'] == [row['id'] for row in original_predictions],
             'Original validation ID order changed')
    _require((recovery_report['status'] == 'completed') == all(row['status'] == 'ok' for row in recovery_predictions),
             'Recovery completion status disagrees with unresolved outcomes')
    original_entries = source['entries']
    _validate_entries(original_entries, original_predictions, 'EXP-006')
    _validate_entries(recovery_entries, recovery_predictions, 'EXP-006R')
    original_accounting = accounting(original_entries)
    recovery_accounting = accounting(recovery_entries)
    _require(original_report.get('accounting') == original_accounting,
             'Original execution accounting disagrees with its recorded attempts')
    _require(recovery_report.get('accounting') == recovery_accounting,
             'Recovery accounting disagrees with its recorded attempts')
    # Include both original failed attempts AND recovery attempts for the 52 IDs.
    # This combined ledger is an in-memory accounting view, never a resumed cache.
    combined_entries = {}
    for rid in source['validation_ids']:
        combined_entries[rid] = {
            'id': rid, 'kind': 'live',
            'attempts': deepcopy(original_entries[rid]['attempts']) +
                        deepcopy(recovery_entries[rid]['attempts'] if rid in recovery_entries else []),
        }
    result = routed_api_costs(evaluate(merged, selective_runs), combined_entries)
    result.update({
        'study_id': STUDY,
        'result_label': 'Exploratory validation: original EXP-006 successes plus separately authorized EXP-006R recovery',
        'status': 'completed' if all(row['status'] == 'ok' for row in merged) else 'finished_with_unresolved',
        'source_protocol_sha256': original_protocol,
        'recovery_protocol_sha256': recovery_protocol,
        'source_tree_sha256': tree_sha256,
        'recovery_eligible_ids': list(eligible),
        'predictions': merged,
        'prediction_source_by_id': {row['id']: 'EXP-006R' if row['id'] in recovery_entries else 'EXP-006'
                                    for row in merged},
        'preserved_original_success_count': N_ORIGINAL_OK,
        'recovery_request_count': N_ELIGIBLE,
        'unresolved_ids': [row['id'] for row in merged if row['status'] != 'ok'],
        'actual_experiment_accounting': {
            'original_exp006': original_accounting,
            'recovery_exp006r': recovery_accounting,
            'combined_collection': accounting(combined_entries),
            'note': 'Original and recovery attempts remain separate; combined collection includes every attempt from both experiments. Usage-priced estimates are not invoice reconciliation.',
        },
        'routed_cost_assumption': 'Replay all observed original and recovery attempts for rejected IDs, including original HTTP-429 attempts. This is an API-component counterfactual, not total-system savings.',
        'official_test_access': False,
        'model_fits': 0,
        'api_calls_by_offline_evaluation': 0,
        'source_artifacts_modified': False,
    })
    return result
