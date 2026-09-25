"""Independent EXP-005 audit; read only EXP-002/004 artifacts, never raw/test data.

No project imports, model deserialization, inference, fitting, or encoding.
Requires local EXP-004 full artifacts to verify every recorded probability hash.
"""

from pathlib import Path
from collections import Counter
from hashlib import sha256
from statistics import mean, stdev
from math import ceil, isfinite
import argparse
import json


ROOT = Path(__file__).resolve().parents[2]
COMPACT = ROOT / 'experiments/exp004-minilm-learning-curve'
FULL = ROOT / 'artifacts/exp004-minilm-learning-curve'


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def canonical(x):
    return (json.dumps(x, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def audit():
    aud = read(COMPACT / 'analysis_audit.json')
    cache = read(FULL / 'primary-cache/metadata.json')
    assert cache == read(COMPACT / 'primary-cache.json')
    assert cache['state_sha256_before'] == cache['state_sha256_after']
    assert cache['frozen_encoder_check']['trainable_parameters'] == 0
    records = []
    common = None
    for shots in [5, 10, 20]:
        for seed in [11, 22, 33, 44, 55]:
            rid = f'exp004-minilm-v2-n{shots}-s{seed}'
            source = COMPACT / 'runs' / f'{rid}.json'
            assert digest(source) == aud['compact_records_sha256'][f'runs/{rid}.json']
            r = read(source)
            m = r['metadata']
            samples = r['samples']
            p = FULL / 'runs' / rid
            assert m == read(p / 'metadata.json')
            assert m['status'] == 'completed' and m['role'] == 'primary' and (m['evaluation_split'] == 'validation')
            assert m['seed'] == seed and m['shots'] == shots and (samples['shots'] == shots) and (samples['seed'] == seed)
            assert m['protocol_id'] == 'banking77-val10-v2'
            assert m['split_manifest_sha256'] == 'f07ac5a03a3ae444db042dd064db920daa3f6462b1e92c9df36ad6118196030a'
            assert m['embedding_source']['revision'] == '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
            assert m['embedding_source']['files_sha256'] == cache['embedding_source']['files_sha256']
            assert m['frozen_encoder_check']['trainable_parameters'] == 0
            assert digest(FULL / 'primary-cache/metadata.json') == m['embedding_cache']['metadata_sha256']
            assert cache['features_sha256'] == m['embedding_cache']['features_sha256']
            for filename, expected in m['artifacts_sha256'].items():
                assert digest(p / filename) == expected, (rid, filename)
            for key in ['samples', 'predictions', 'metrics']:
                assert r[key] == read(p / f'{key}.json')
                assert sha256(canonical(r[key])).hexdigest() == m['artifacts_sha256'][f'{key}.json']
            refpath = ROOT / 'experiments/exp002-learning-curve/runs' / f"{m['reference_run_id']}.json"
            ref = read(refpath)
            assert sha256(canonical(ref)).hexdigest() == m['reference_sha256']
            assert r['samples'] == ref['samples']
            assert [(x['id'], x['true_label']) for x in r['predictions']] == [(x['id'], x['true_label']) for x in ref['predictions']]
            assert len(samples['train_ids']) == shots * 77 and len(set(samples['train_ids'])) == shots * 77
            assert set(samples['train_ids']).isdisjoint(samples['validation_ids'])
            prob = read(p / 'probabilities.json')
            ids = samples['validation_ids']
            labels = samples['labels']
            preds = r['predictions']
            vals = prob['probabilities']
            assert prob['validation_ids'] == ids and prob['labels'] == labels
            assert len(ids) == len(set(ids)) == 770 and len(vals) == len(preds) == 770 and (len(labels) == len(set(labels)) == 77)
            assert [x['id'] for x in preds] == ids
            truths = [x['true_label'] for x in preds]
            assert Counter(truths) == Counter({label: 10 for label in labels})
            if common is None:
                common = (ids, truths, m['configuration'], m['source'])
            assert (ids, truths, m['configuration'], m['source']) == common
            confidence = []
            for row, pred in zip(vals, preds, strict=True):
                assert len(row) == 77 and all((isfinite(x) and 0 <= x <= 1 for x in row))
                assert abs(sum(row) - 1) < 1e-12
                winner = max(range(77), key=lambda j: row[j])
                assert labels[winner] == pred['predicted_label']
                confidence.append(max(row))
            ordered = sorted(range(770), key=lambda i: (-confidence[i], sha256(ids[i].encode('utf-8')).hexdigest(), ids[i]))
            errors = [pred['predicted_label'] != pred['true_label'] for pred in preds]
            full_acc = sum((not x for x in errors)) / 770
            assert full_acc == r['metrics']['frozen_minilm_logistic_regression']['accuracy']
            points = []
            for target in [0.25, 0.5, 0.75, 0.9, 1.0]:
                k = ceil(target * 770)
                accept = ordered[:k]
                e = sum((errors[i] for i in accept))
                byclass = Counter((truths[i] for i in accept))
                points.append({
                    'target': target, 'accepted': k, 'coverage': k / 770,
                    'errors': e, 'accuracy': (k - e) / k, 'risk': e / k,
                    'zero_accepted_classes': [label for label in labels if byclass[label] == 0],
                    'per_true_class_accepted': {label: byclass[label] for label in labels},
                })
            records.append({
                'id': rid, 'shots': shots, 'seed': seed,
                'confidence_min': min(confidence),
                'confidence_max': max(confidence),
                'exact_tie_groups': sum(n > 1 for n in Counter(confidence).values()),
                'probabilities_sha256': digest(p / 'probabilities.json'),
                'ranked_ids': [ids[i] for i in ordered],
                'ranked_error_flags': [errors[i] for i in ordered],
                'ranked_confidence': [confidence[i] for i in ordered],
                'points': points,
            })
    summary = []
    for shots in [5, 10, 20]:
        for ti in range(5):
            points = [r['points'][ti] for r in records if r['shots'] == shots]
            row = {'shots': shots, 'target': points[0]['target'], 'accepted': points[0]['accepted'], 'coverage': points[0]['coverage']}
            for field in ['accuracy', 'risk', 'errors']:
                values = [p[field] for p in points]
                row[field] = {'mean': mean(values), 'sample_sd': stdev(values), 'values': values}
            zeros = [len(p['zero_accepted_classes']) for p in points]
            row['zero_accepted_class_counts'] = {'mean': mean(zeros), 'sample_sd': stdev(zeros), 'min': min(zeros), 'max': max(zeros), 'values': zeros}
            summary.append(row)
    result = {
        'independent_implementation': 'Python stdlib only; no project imports, model fitting, cache encoding, raw data, or test access',
        'tie_rule': 'descending maximum probability, then SHA256 UTF-8 row ID ascending, then row ID ascending; exact ties only',
        'rounding': 'ceil(target * 770)',
        'all_source_checks_passed': True,
        'runs': records,
        'summary': summary,
    }
    return result


def main():
    if not __debug__:
        raise RuntimeError('Run without -O: integrity checks use assertions')
    parser = argparse.ArgumentParser(description='Independently audit EXP-004 probabilities and calculate EXP-005 using only Python stdlib.')
    parser.add_argument('--output', type=Path, required=True, help='Fresh JSON report path; existing files are refused')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"Verified {len(result['runs'])} primary runs; wrote {args.output}")

if __name__ == '__main__':
    main()
