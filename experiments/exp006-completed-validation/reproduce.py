"""Reproduce compact EXP-006 + EXP-006R validation evidence without API/data access."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from baseline.data import json_bytes, read_json, sha256
from baseline.general import accounting, priced_response, routed_api_costs
from baseline.general_metrics import evaluate
from baseline.general_protocol import PRICING, digest
from baseline.selective import restore


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quality_summary(result):
    metric_keys = ('n_examples', 'correct_count', 'error_count', 'unresolved_count', 'accuracy', 'macro_f1')
    return {
        'study_id': 'EXP-006 + EXP-006R recovery', 'evaluation_split': 'validation',
        'general': result['general'], 'status_counts': result['status_counts'],
        'general_response_count': result['general_response_count'],
        'combinations': [{
            **{key: point[key] for key in ('run_id', 'shots', 'seed', 'target_coverage', 'coverage',
                                          'accepted_count', 'rejected_count')},
            'accepted_ids_sha256': digest(point['accepted_ids']),
            'rejected_ids_sha256': digest(point['rejected_ids']),
            'combined': {key: point['combined'][key] for key in metric_keys},
            'fallback': point['fallback'],
        } for point in result['combinations']],
        'regimes': result['regimes'], 'primary_run_count': result['primary_run_count'],
        'unique_validation_examples': result['unique_validation_examples'],
        'additional_validation_label_budget': result['additional_validation_label_budget'],
        'new_labels': 0, 'general_response_sets': 1,
        'summary_sd': result['summary_sd'],
        'limitations': 'Same reused 770-case validation set and same general predictions for every seed. Exploratory; no production threshold, independent-holdout quality, or total-system savings claim.',
    }


def compact_accounting(entries_by_stage, evaluation):
    original, recovery = entries_by_stage['EXP-006'], entries_by_stage['EXP-006R']
    combined = {rid: {'id': rid, 'attempts': deepcopy(entry['attempts']) +
                     deepcopy(recovery.get(rid, {'attempts': []})['attempts'])}
                for rid, entry in original.items()}
    stage = {'original_exp006': accounting(original), 'recovery_exp006r': accounting(recovery),
             'combined_collection': accounting(combined)}
    projected = routed_api_costs(deepcopy(evaluation), combined)
    return {'study_id': 'EXP-006 + EXP-006R recovery', 'pricing': PRICING,
            'actual_experiment_accounting': stage,
            'attempt_status_counts': {name: dict(sorted(Counter(a['status'] for e in entries.values()
                                          for a in e['attempts']).items()))
                                      for name, entries in entries_by_stage.items()},
            'usage_totals_from_reported_attempts': {name: {
                'input_tokens': sum(a['usage']['input_tokens'] for e in entries.values() for a in e['attempts'] if a.get('usage')),
                'output_tokens': sum(a['usage']['output_tokens'] for e in entries.values() for a in e['attempts'] if a.get('usage')),
                'cached_input_tokens': sum(a['usage']['input_tokens_details']['cached_tokens'] for e in entries.values() for a in e['attempts'] if a.get('usage')),
                'cache_write_tokens': sum(a['usage']['input_tokens_details']['cache_write_tokens'] for e in entries.values() for a in e['attempts'] if a.get('usage')),
                'reasoning_tokens': sum(a['usage'].get('output_tokens_details', {}).get('reasoning_tokens', 0) for e in entries.values() for a in e['attempts'] if a.get('usage')),
            } for name, entries in entries_by_stage.items()},
            'hypothetical_routed_api_components': [{
                **{key: point[key] for key in ('run_id', 'shots', 'seed', 'target_coverage', 'coverage')},
                **point['hypothetical_routed_api_charges'],
            } for point in projected['combinations']],
            'note': 'All original and recovery attempts are retained. Missing usage remains unknown, not free. Usage-priced charges are not invoice reconciliation. Routed API estimates replay observed cache behavior and original/recovery retries; specialist deployment cost and total-system savings remain unmeasured.'}


def reproduce():
    verified = read_json(BUNDLE / 'verification.json')
    for name, expected in verified['compact_files_sha256'].items():
        require(sha256((BUNDLE / name).read_bytes()) == expected, f'Compact artifact changed: {name}')
    require(read_json(BUNDLE / 'pricing.json') == PRICING, 'Frozen pricing differs from accounting implementation')
    runs = []
    for name, expected in sorted(verified['specialist_files_sha256'].items()):
        path = ROOT / name
        require(sha256(path.read_bytes()) == expected, 'Versioned EXP-005 specialist changed')
        runs.append(restore(read_json(path)))
    predictions = read_json(BUNDLE / 'predictions.json')
    require(len(predictions) == len({p['id'] for p in predictions}) == 770, 'Expected exactly 770 unique validation IDs')
    require(Counter(p['original_status'] for p in predictions) == {'ok': 718, 'http_429': 52}, 'Original statuses changed')
    require(all(p['status'] == 'ok' for p in predictions), 'Recovery completion changed')
    require(all((p['original_status'] == 'ok' and p['prediction_source'] == 'EXP-006' and p['recovery_status'] is None) or
                (p['original_status'] == 'http_429' and p['prediction_source'] == 'EXP-006R' and p['recovery_status'] == 'ok')
                for p in predictions), 'Stage/source alignment changed')
    rows = [{key: p[key] for key in ('id', 'status', 'predicted_label')} for p in predictions]
    evaluation = evaluate(rows, runs)
    entries = {'EXP-006': {}, 'EXP-006R': {}}
    for item in read_json(BUNDLE / 'accounting_attempts.json'):
        stage, rid = item['stage'], item['id']
        require(stage in entries, 'Unknown collection stage')
        entry = entries[stage].setdefault(rid, {'id': rid, 'attempts': []})
        attempt = {k: v for k, v in item.items() if k not in ('stage', 'id')}
        require(attempt['attempt'] == len(entry['attempts']) + 1, 'Attempt order changed')
        cost = priced_response({'model': attempt.get('returned_model'),
                               'service_tier': attempt.get('returned_service_tier'), 'usage': attempt.get('usage')})
        require(attempt['usage_priced_usd'] == (str(cost) if cost is not None else None), 'Compact usage price differs')
        entry['attempts'].append(attempt)
    require(set(entries['EXP-006']) == {p['id'] for p in predictions}, 'Original accounting IDs changed')
    require(set(entries['EXP-006R']) == {p['id'] for p in predictions if p['original_status'] == 'http_429'}, 'Recovery accounting IDs changed')
    for p in predictions:
        require(entries['EXP-006'][p['id']]['attempts'][-1]['status'] == p['original_status'], 'Original terminal status differs')
        if p['recovery_status'] is not None:
            require(entries['EXP-006R'][p['id']]['attempts'][-1]['status'] == p['recovery_status'], 'Recovery terminal status differs')
    quality = quality_summary(evaluation)
    costs = compact_accounting(entries, evaluation)
    require(quality == read_json(BUNDLE / 'quality_summary.json'), 'Quality summary failed reproduction')
    require(costs == read_json(BUNDLE / 'accounting.json'), 'Accounting summary failed reproduction')
    return quality, costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional fresh directory for regenerated summaries')
    args = parser.parse_args()
    quality, costs = reproduce()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        for name, value in (('quality_summary.json', quality), ('accounting.json', costs)):
            (args.output / name).write_bytes(json_bytes(value))
    print(json.dumps({'status': 'verified', 'validation_ids': 770, 'specialists': 15,
                      'combinations': len(quality['combinations']),
                      'standalone_accuracy': quality['general']['accuracy'],
                      'standalone_macro_f1': quality['general']['macro_f1'],
                      'unresolved': quality['general']['unresolved_count'],
                      'accounting': costs['actual_experiment_accounting']['combined_collection']}, indent=2))


if __name__ == '__main__':
    main()
