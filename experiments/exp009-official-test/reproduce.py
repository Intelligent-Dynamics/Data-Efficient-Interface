"""Replay compact EXP-009 sufficient statistics; no private data or model imports."""
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

ARMS = {'specialist', 'zero_shot_luna', 'zero_shot_hybrid', 'retrieved_luna', 'retrieved_hybrid'}
PROTOCOL = 'b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334'
FREEZE = 'c2963a59ca83911a9ccacf9390390fed18fb9bb5282bf38ae1b33b5c44be2a8c'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    require(isinstance(a, (int, float)) and isinstance(b, (int, float)) and
            math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-12,
            'Numerical mismatch')
    return abs(a-b)


def metrics(classes):
    """Recompute accuracy/macro-F1 from TP/FP/FN; retain unknown predictions."""
    require(isinstance(classes, dict) and classes, 'Class statistics required')
    n = correct = predicted = 0
    f1_values = []
    max_difference = 0.0
    for row in classes.values():
        keys = ('true_positive', 'false_positive', 'false_negative', 'support')
        require(all(type(row[k]) is int and row[k] >= 0 for k in keys), 'Invalid class count')
        tp, fp, fn, support = (row[k] for k in keys)
        require(tp + fn == support, 'Support/count mismatch')
        precision = tp / (tp+fp) if tp+fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        for key, value in (('precision', precision), ('recall', recall), ('f1', f1)):
            max_difference = max(max_difference, close(row[key], value))
        n += support
        correct += tp
        predicted += tp+fp
        f1_values.append(f1)
    require(n > 0 and predicted <= n, 'Predicted population exceeds support')
    return {'accuracy': correct/n, 'macro_f1': sum(f1_values)/len(f1_values),
            'n_examples': n, 'correct_count': correct, 'error_count': n-correct,
            'unresolved_count': n-predicted}, max_difference


def replay(summary, per_class, accounting):
    require(summary['protocol_sha256'] == PROTOCOL and summary['prediction_freeze_sha256'] == FREEZE,
            'Protocol/freeze mismatch')
    require(summary['n_examples'] == 3080 and summary['companion']['seed'] == 11 and
            summary['companion']['seed_count'] == 1 and summary['task_training_labels'] == 1540 and
            summary['additional_validation_labels'] == 770, 'Population/label budget mismatch')
    labels = per_class['labels']
    require(len(labels) == 77 and len(set(labels)) == 77, 'Exactly 77 unique labels required')
    require(set(per_class['metrics']) == set(summary['metrics']) == ARMS, 'Experiment arms changed')
    max_difference = 0.0
    replayed = {}
    for arm in sorted(ARMS):
        classes = per_class['metrics'][arm]
        require(set(classes) == set(labels) and all(r['support'] == 40 for r in classes.values()),
                'Class population changed')
        replayed[arm], difference = metrics(classes)
        max_difference = max(max_difference, difference)
        for key, value in replayed[arm].items():
            max_difference = max(max_difference, close(summary['metrics'][arm][key], value))
        require(replayed[arm]['unresolved_count'] == 0, 'Completed outputs missing')
    for pair, difference in summary['paired_differences'].items():
        left, right = pair.split(' minus ')
        for metric in ('accuracy', 'macro_f1'):
            max_difference = max(max_difference, close(difference[metric], replayed[left][metric]-replayed[right][metric]))
    require(set(summary['paired_differences']) == {
        'retrieved_luna minus zero_shot_luna', 'retrieved_luna minus specialist',
        'retrieved_hybrid minus specialist', 'retrieved_hybrid minus retrieved_luna',
        'retrieved_hybrid minus zero_shot_hybrid'}, 'Paired comparisons changed')
    accepted, rejected = summary['accepted_count'], summary['fallback_count']
    require(type(accepted) is int and type(rejected) is int and accepted+rejected == 3080 and
            accepted == 2812 and rejected == 268, 'Acceptance population changed')
    max_difference = max(max_difference, close(summary['observed_coverage'], accepted/3080),
                         close(summary['fallback_percentage'], 100*rejected/3080))
    correct = summary['fallback_correct_counts']
    require(correct == {'retrieved':202, 'zero_shot':154}, 'Fallback counts changed')
    for name in correct:
        max_difference = max(max_difference, close(summary['fallback_accuracy'][name], correct[name]/rejected))
    require(summary['metrics']['retrieved_hybrid']['correct_count']-correct['retrieved'] ==
            summary['metrics']['zero_shot_hybrid']['correct_count']-correct['zero_shot'] == 2511,
            'Accepted specialist correct count changed')
    require(summary['retrieved_status_counts'] == {'ok':3080}, 'Completed status changed')
    prices = accounting['pricing']['per_million_tokens']
    u = accounting['usage_totals']
    known = (Decimal(u['input_tokens']-u['cached_tokens']-u['cache_write_tokens'])*Decimal(prices['input']) +
             Decimal(u['cached_tokens'])*Decimal(prices['cached_input']) +
             Decimal(u['cache_write_tokens'])*Decimal(prices['cache_write']) +
             Decimal(u['output_tokens'])*Decimal(prices['output']))/1000000
    unknown = sum((Decimal(r['reservation_usd'])*r['count'] for r in accounting['unknown_usage_bins']), Decimal(0))
    saved = accounting['accounting']
    require(known == Decimal(saved['usage_priced_api_charges_usd']) == Decimal('0.576343155') and
            [known, known+unknown] == [Decimal(v) for v in saved['spend_interval_under_frozen_assumptions_usd']] ==
            [Decimal('0.576343155'),Decimal('0.584791155')], 'Accounting mismatch')
    require(accounting['final_status_counts'] == {'ok':3080} and
            accounting['attempt_status_counts'] == {'ok':3080, 'transport_unknown':3} and
            saved['recorded_attempts'] == 3083 and saved['unknown_charge_attempts'] == 3 and
            saved['actual_api_spend_usd'] is None and saved['invoice_reconciled'] is False,
            'Attempt/invoice status changed')
    return {'verified':True, 'n_examples':3080, 'metrics':replayed,
            'maximum_absolute_arithmetic_difference':max_difference,
            'replay_scope':'Saved per-class sufficient statistics; no new row-level scoring.',
            'usage_priced_api_charges_usd':str(known),
            'spend_interval_under_frozen_assumptions_usd':[str(known),str(known+unknown)]}


def reproduce(directory=None):
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
    verification = json.loads((directory/'verification.json').read_text())
    hashes = verification['compact_files_sha256']
    require({'summary.json','per_class.json','accounting.json','provenance.json','reproduce.py','README.md'} <= set(hashes),
            'Incomplete compact inventory')
    for name, expected in hashes.items():
        require(Path(name).name == name and name != 'verification.json', 'Unsafe compact path')
        require(hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected, 'Compact file hash mismatch: '+name)
    result = replay(*(json.loads((directory/name).read_text()) for name in ('summary.json','per_class.json','accounting.json')))
    close(result['maximum_absolute_arithmetic_difference'],verification['maximum_absolute_arithmetic_difference'])
    return result


if __name__ == '__main__':
    print(json.dumps(reproduce(),sort_keys=True,indent=2))
