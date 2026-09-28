"""Replay compact EXP-007 counts/usage without raw data, API access or model code."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics

BUNDLE = Path(__file__).resolve().parent
SEEDS = ('11', '22', '33', '44', '55')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()).hexdigest()


def confusion(truth, predictions, labels):
    require(len(truth) == len(predictions), 'Prediction alignment mismatch')
    indices = {label: i for i, label in enumerate(labels)}
    require(len(indices) == len(labels), 'Duplicate labels')
    require(all(t in indices and (p is None or p in indices) for t, p in zip(truth, predictions)), 'Invalid label')
    counts = Counter((indices[t], -1 if p is None else indices[p]) for t, p in zip(truth, predictions))
    return [[t, p, count] for (t, p), count in sorted(counts.items())]


def metrics(cells, labels):
    support, predicted, correct = Counter(), Counter(), Counter()
    seen = set()
    for t, p, count in cells:
        require(type(t) is int and type(p) is int and type(count) is int and count > 0 and
                0 <= t < len(labels) and -1 <= p < len(labels) and (t, p) not in seen, 'Invalid confusion counts')
        seen.add((t, p)); support[t] += count; predicted[p] += count
        if t == p: correct[t] += count
    n = sum(support.values())
    require(n > 0, 'Empty metric denominator')
    per_class = {}
    for i, label in enumerate(labels):
        tp, fp, fn = correct[i], predicted[i]-correct[i], support[i]-correct[i]
        per_class[label] = {'precision': tp/(tp+fp) if tp+fp else 0.,
            'recall': tp/(tp+fn) if tp+fn else 0., 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'support': support[i], 'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    return {'n_examples': n, 'correct_count': sum(correct.values()), 'error_count': n-sum(correct.values()),
            'unresolved_count': predicted[-1], 'accuracy': sum(correct.values())/n,
            'macro_f1': statistics.mean(c['f1'] for c in per_class.values()), 'per_class': per_class}


def summary(counts):
    labels, n = counts['labels'], counts['n_examples']
    require(n == 3080 and len(labels) == len(set(labels)) == 77, 'Official test population mismatch')
    require(set(counts['seeds']) == set(SEEDS), 'All five seeds required')
    luna = metrics(counts['luna'], labels)
    require(luna['n_examples'] == n, 'Luna denominator mismatch')
    small = lambda m: {k: v for k, v in m.items() if k != 'per_class'}
    result = {'study_id': 'EXP-007', 'evaluation_split': 'official test', 'n_examples': n,
              'luna_only': small(luna), 'seeds': {}, 'aggregate': {}}
    per_class = {'luna_only': luna['per_class'], 'seeds': {}}
    for seed in SEEDS:
        item = counts['seeds'][seed]
        sp, routed = metrics(item['specialist'], labels), metrics(item['routed'], labels)
        fallback, rejected_sp = metrics(item['rejected_luna'], labels), metrics(item['rejected_specialist'], labels)
        accepted = item['per_class_accepted']
        require(len(accepted) == len(labels) and all(type(a) is int and 0 <= a <= 40 for a in accepted), 'Acceptance counts invalid')
        require(sp['n_examples'] == routed['n_examples'] == n and
                fallback['n_examples'] == rejected_sp['n_examples'] == n-sum(accepted), 'Coverage denominator mismatch')
        require(all(m['per_class'][label]['support'] == 40 for m in (luna, sp, routed) for label in labels), 'Class support mismatch')
        paired = item['rejected_complementarity']
        require(sum(paired.values()) == fallback['n_examples'] and
                paired['luna_only_correct']+paired['both_correct'] == fallback['correct_count'] and
                paired['specialist_only_correct']+paired['both_correct'] == rejected_sp['correct_count'] and
                routed['correct_count']-sp['correct_count'] == paired['luna_only_correct']-paired['specialist_only_correct'],
                'Complementary counts inconsistent')
        result['seeds'][seed] = {'seed': int(seed), 'threshold': item['threshold'],
            'specialist_only': small(sp), 'routed': small(routed),
            'accepted_count': sum(accepted), 'observed_coverage': sum(accepted)/n,
            'fallback_count': fallback['n_examples'], 'fallback_percentage': 100*fallback['n_examples']/n,
            'fallback_accuracy_on_rejected': fallback['accuracy'],
            'specialist_accuracy_on_rejected': rejected_sp['accuracy'], 'rejected_complementarity': paired,
            'accuracy_gain_percentage_points': 100*(routed['accuracy']-sp['accuracy']),
            'macro_f1_gain_percentage_points': 100*(routed['macro_f1']-sp['macro_f1'])}
        per_class['seeds'][seed] = {'specialist_only': sp['per_class'], 'routed': routed['per_class'],
            'acceptance': {label: {'accepted_count': a, 'rejected_count': 40-a, 'support': 40}
                           for label, a in zip(labels, accepted)}}
    def stats(values): return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}
    for model in ('specialist_only', 'routed'):
        result['aggregate'][model] = {metric: stats([r[model][metric] for r in result['seeds'].values()])
                                     for metric in ('accuracy', 'macro_f1')}
    for key in ('accepted_count', 'observed_coverage', 'fallback_count', 'fallback_percentage',
                'fallback_accuracy_on_rejected', 'specialist_accuracy_on_rejected',
                'accuracy_gain_percentage_points', 'macro_f1_gain_percentage_points'):
        result['aggregate'][key] = stats([r[key] for r in result['seeds'].values()])
    result['limitations'] = ['Five training seeds share one 3080-case test population and one Luna response set.',
        'Sample SD across seeds is not a confidence interval.', '1540 fitting labels per specialist plus 770 development labels and external encoder pretraining.',
        'Fallback fraction is a policy replay; standalone baseline collection queried all 3080 cases.',
        'Specialist deployment cost, production latency and total-system dollar savings are unmeasured.',
        'Retrieved-example EXP-009 official-test result remains pending; no tuning follows these test results.']
    return result, per_class


def reproduce():
    verified = json.loads((BUNDLE/'verification.json').read_text())
    for name, expected in verified['compact_files_sha256'].items():
        require(hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest() == expected, 'Compact evidence changed: '+name)
    actual, per_class = summary(json.loads((BUNDLE/'counts.json').read_text()))
    require(actual == json.loads((BUNDLE/'summary.json').read_text()), 'Aggregate replay mismatch')
    require(per_class == json.loads((BUNDLE/'per_class.json').read_text()), 'Per-class replay mismatch')
    accounting = json.loads((BUNDLE/'accounting.json').read_text())
    # Accounting export defines its aggregate usage bins and rate components explicitly.
    rates = accounting['pricing']['per_million_tokens']
    totals = accounting['usage_totals']
    charged = (Decimal(totals['input_tokens']-totals['cached_tokens']-totals['cache_write_tokens'])*Decimal(rates['input'])+
               Decimal(totals['cached_tokens'])*Decimal(rates['cached_input'])+
               Decimal(totals['cache_write_tokens'])*Decimal(rates['cache_write'])+
               Decimal(totals['output_tokens'])*Decimal(rates['output']))/1000000
    require(charged == Decimal(accounting['accounting']['usage_priced_api_charges_usd']), 'Usage price replay mismatch')
    unknown = sum((Decimal(b['reservation_usd'])*b['count'] for b in accounting['unknown_usage_bins']), Decimal(0))
    require(unknown == Decimal(accounting['accounting']['unknown_charge_reservation_usd']), 'Unknown reservation replay mismatch')
    require([charged,charged+unknown] == list(map(Decimal,accounting['accounting']['spend_interval_under_frozen_assumptions_usd'])), 'Spend interval replay mismatch')
    return actual


if __name__ == '__main__':
    report = reproduce()
    print(json.dumps({'verified': True, 'n_examples': report['n_examples'],
                      'luna_only': report['luna_only'], 'aggregate': report['aggregate']}, indent=2))
