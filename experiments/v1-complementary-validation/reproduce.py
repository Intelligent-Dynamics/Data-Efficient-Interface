"""Independently reproduce validation quality from versioned, text-free evidence.

No project inference/data/API module is imported. Metric arithmetic is implemented
here to check the earlier evaluator independently. Inputs are fixed versioned JSONs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).resolve().parent
PROTOCOL = 'experiments/exp007-fixed-threshold-preparation/protocol.json'
PROTOCOL_SHA256 = '0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00'
SEEDS = (11, 22, 33, 44, 55)
BUDGETS = (5, 10, 20)
ACCEPTED_COUNTS = (0, 193, 385, 578, 693, 770)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(truth, prediction, labels):
    """All rows and all official classes, including missing/invalid predictions."""
    require(len(truth) == len(prediction) and len(truth) > 0, 'Metric row mismatch')
    require(len(labels) == len(set(labels)) and set(truth) <= set(labels), 'Label mismatch')
    true_counts = Counter(truth)
    predicted_counts = Counter(prediction)
    correct_by_class = Counter(t for t, p in zip(truth, prediction, strict=True) if t == p)
    correct = sum(correct_by_class.values())
    return {
        'n_examples': len(truth), 'correct_count': correct, 'error_count': len(truth) - correct,
        'accuracy': correct / len(truth),
        'macro_f1': statistics.mean(
            2 * correct_by_class[label] / (true_counts[label] + predicted_counts[label])
            if true_counts[label] + predicted_counts[label] else 0.0 for label in labels),
    }


def paired(truth, specialist, luna):
    cells = Counter((s == t, g == t) for t, s, g in zip(truth, specialist, luna, strict=True))
    return {
        'both_correct': cells[True, True], 'specialist_only_correct': cells[True, False],
        'luna_only_correct': cells[False, True], 'both_wrong': cells[False, False],
        'net_correct_added_by_luna': cells[False, True] - cells[True, False],
    }


def summary(values):
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
            'minimum': min(values), 'maximum': max(values)}


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=2e-15), message)


def reproduce():
    sources = {}

    def read(relative, expected_hash=None):
        # Explicit compact-evidence paths only; never discover raw/test/cache inputs.
        require(relative.startswith('experiments/') and '..' not in Path(relative).parts,
                'Only versioned experiment evidence is allowed')
        raw = (ROOT / relative).read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        require(expected_hash is None or actual_hash == expected_hash, f'Hash mismatch: {relative}')
        sources[relative] = actual_hash
        return json.loads(raw)

    frozen = read(PROTOCOL, PROTOCOL_SHA256)
    manifest = read('experiments/exp006-completed-validation/verification.json')
    saved = {}
    for name in ('predictions.json', 'quality_summary.json'):
        saved[name] = read('experiments/exp006-completed-validation/' + name,
                           manifest['compact_files_sha256'][name])
    responses = saved['predictions.json']
    luna_by_id = {r['id']: r['predicted_label'] if r['status'] == 'ok' else None for r in responses}
    require(len(responses) == len(luna_by_id) == 770, 'Expected all 770 unique response IDs')
    require(all(r['id'].startswith('train:') for r in responses), 'Non-training-source validation ID')
    require(Counter(r['status'] for r in responses) == {'ok': 770}, 'Completion status changed')
    require(Counter(r['original_status'] for r in responses) == {'ok': 718, 'http_429': 52},
            'Original/recovery populations changed')
    thresholds = read('experiments/exp007-fixed-threshold-preparation/thresholds.json')
    threshold_by_seed = {r['seed']: r['threshold'] for r in thresholds}
    require(set(threshold_by_seed) == set(SEEDS), 'Threshold seed mismatch')
    require(all(threshold_by_seed[seed].hex() == frozen['thresholds'][str(seed)]['binary64_hex']
                for seed in SEEDS), 'Frozen scalar threshold differs')
    expected_points = {(p['shots'], p['seed'], p['accepted_count']): p
                       for p in saved['quality_summary.json']['combinations']}
    rows, baseline_rows, rejected_rows = [], [], []
    canonical_truth, label_names = None, None
    train_ids_by_seed = {}
    for shots in BUDGETS:
        for seed in SEEDS:
            stem = f'exp004-minilm-v2-n{shots}-s{seed}'
            relative = f'experiments/exp005-selective-diagnostic/runs/{stem}.json'
            diagnostic = read(relative, manifest['specialist_files_sha256'][relative])
            ranked = diagnostic['ranked_rows']
            labels = diagnostic['labels']
            require(len(labels) == len(set(labels)) == 77, 'Expected all 77 classes')
            require(len(ranked) == len({r['id'] for r in ranked}) == 770, 'Ranked IDs mismatch')
            require(set(r['id'] for r in ranked) == set(luna_by_id), 'Response/specialist ID mismatch')
            require(ranked == sorted(ranked, key=lambda r: (-r['confidence'],
                    hashlib.sha256(r['id'].encode('utf-8')).hexdigest(), r['id'])), 'Confidence order mismatch')
            by_id = {r['id']: r for r in ranked}
            truth_by_id = {r['id']: r['true_label'] for r in ranked}
            if canonical_truth is None:
                canonical_truth, label_names = truth_by_id, labels
            require(truth_by_id == canonical_truth and labels == label_names, 'Validation labels drifted')
            primary_path = f'experiments/exp004-minilm-learning-curve/runs/{stem}.json'
            primary = read(primary_path, frozen['input_files_sha256'].get(primary_path))
            require({r['id']: {k: r[k] for k in ('predicted_label', 'true_label')}
                     for r in primary['predictions']} ==
                    {rid: {k: r[k] for k in ('predicted_label', 'true_label')} for rid, r in by_id.items()},
                    'EXP-004/005 prediction mismatch')
            require(len(primary['samples']['train_ids']) == 77 * shots and
                    not (set(primary['samples']['train_ids']) & set(by_id)), 'Training budget or isolation mismatch')
            if shots == 20:
                train_ids_by_seed[str(seed)] = primary['samples']['train_ids']
            truth = [r['true_label'] for r in ranked]
            specialist = [r['predicted_label'] for r in ranked]
            luna = [luna_by_id[r['id']] for r in ranked]
            specialist_metrics = metrics(truth, specialist, labels)
            for key in ('accuracy', 'macro_f1'):
                close(specialist_metrics[key], primary['metrics']['frozen_minilm_logistic_regression'][key],
                      'Primary specialist metric differs')
            baseline_rows.append({'shots': shots, 'seed': seed, **specialist_metrics})
            for accepted in ACCEPTED_COUNTS:
                combined = metrics(truth, specialist[:accepted] + luna[accepted:], labels)
                previous = expected_points[shots, seed, accepted]
                for key, value in combined.items():
                    close(value, previous['combined'][key], 'Previously reported combination differs')
                rows.append({'shots': shots, 'seed': seed, 'accepted_count': accepted,
                             'fallback_count': 770 - accepted, 'fallback_fraction': (770 - accepted) / 770,
                             **combined})
            if shots == 20:
                accepted_ids = {r['id'] for r in ranked if r['confidence'] >= threshold_by_seed[seed]}
                require(accepted_ids == {r['id'] for r in ranked[:693]}, 'Fixed threshold changed acceptance')
                rt, rs, rg = truth[693:], specialist[693:], luna[693:]
                specialist_rejected = metrics(rt, rs, labels)
                luna_rejected = metrics(rt, rg, labels)
                at_90 = next(r for r in rows if r['shots'] == 20 and r['seed'] == seed and r['accepted_count'] == 693)
                cells = paired(rt, rs, rg)
                require(specialist_metrics['correct_count'] + cells['net_correct_added_by_luna'] ==
                        at_90['correct_count'], 'Paired rejected gain does not reproduce hybrid')
                rejected_rows.append({
                    'seed': seed, 'threshold': threshold_by_seed[seed], 'accepted_count': 693,
                    'fallback_count': 77, 'coverage': 0.9,
                    'specialist_all': specialist_metrics, 'hybrid_all':
                        {k: at_90[k] for k in specialist_metrics},
                    'specialist_on_rejected': specialist_rejected, 'luna_on_same_rejected': luna_rejected,
                    'paired_rejected_outcomes': cells,
                    'rejected_accuracy_gain_percentage_points': 100 * (luna_rejected['accuracy'] - specialist_rejected['accuracy']),
                    'hybrid_accuracy_gain_vs_specialist_percentage_points': 100 * (at_90['accuracy'] - specialist_metrics['accuracy']),
                    'hybrid_macro_f1_gain_vs_specialist_percentage_points': 100 * (at_90['macro_f1'] - specialist_metrics['macro_f1']),
                })
    truth = [canonical_truth[rid] for rid in luna_by_id]
    general = metrics(truth, list(luna_by_id.values()), label_names)
    for key, value in general.items():
        close(value, saved['quality_summary.json']['general'][key], 'Luna standalone metric differs')
    aggregates = []
    for shots in BUDGETS:
        for accepted in reversed(ACCEPTED_COUNTS):
            selected = [r for r in rows if r['shots'] == shots and r['accepted_count'] == accepted]
            aggregates.append({'shots': shots, 'accepted_count': accepted,
                'fallback_count': 770 - accepted, 'fallback_fraction': (770 - accepted) / 770,
                'accuracy': summary([r['accuracy'] for r in selected]),
                'macro_f1': summary([r['macro_f1'] for r in selected])})
    rejected_summary = {name: summary([r[name]['accuracy'] for r in rejected_rows])
                        for name in ('specialist_all', 'hybrid_all', 'specialist_on_rejected', 'luna_on_same_rejected')}
    for key in ('hybrid_accuracy_gain_vs_specialist_percentage_points',
                'hybrid_macro_f1_gain_vs_specialist_percentage_points',
                'rejected_accuracy_gain_percentage_points'):
        rejected_summary[key] = summary([r[key] for r in rejected_rows])
    return {
        'analysis': 'v1 complementary routing: independently recomputed saved validation evidence',
        'evaluation_split': 'validation', 'unique_requests': 770, 'seeds': list(SEEDS),
        'source_protocol_sha256': PROTOCOL_SHA256, 'sources_sha256': sources,
        'general': general, 'specialist_baselines': baseline_rows, 'individual_combinations': rows,
        'aggregate_combinations': aggregates, 'fixed_90_percent_per_seed': rejected_rows,
        'fixed_90_percent_summary': rejected_summary,
        'paired_20_shot_rejected_event_totals': {
            key: sum(r['paired_rejected_outcomes'][key] for r in rejected_rows)
            for key in rejected_rows[0]['paired_rejected_outcomes']},
        'labels': {'training_per_20_shot_specialist': 1540, 'additional_validation': 770,
            'training_union_across_20_shot_seeds': len(set().union(*map(set, train_ids_by_seed.values()))),
            'all_source_training_labels_mechanically_audited_in_original_pipeline': 10003,
            'new_labels_for_this_reanalysis': 0, 'external_encoder_pretraining': True},
        'interpretation': [
            'One shared 770-case reused validation set and one shared recovered zero-shot Luna response set.',
            'Five-seed sample SD is training-sample variation, not a confidence interval for new requests.',
            'At 20 shots and 90% specialist coverage, fallback improves over specialist alone by correcting complementary errors.',
            'Luna is less accurate than the 20-shot specialist overall; it is more accurate on each matching rejected subset.',
            'All 90 combinations including unfavorable 5-shot outcomes are retained; more fallback is not always better.',
            'Curves connect existing landmarks for readability and do not evaluate intermediate policies.',
            'Only the marked 90% validation landmark has frozen per-seed deployment thresholds; other points are ranking diagnostics.',
            'Reduced fallback request fraction is not a measurement of total-system dollar savings.',
            'Official test was mechanically opened in prior authorized preflight but remains uninferred and unscored; this reanalysis never accesses it.',
        ],
    }


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'id-v1-complementary'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    colors = {5: '#bf6d28', 10: '#618d63', 20: '#23689b'}
    for ax, metric, title in zip(axes, ('accuracy', 'macro_f1'), ('Accuracy', 'Macro-F1'), strict=True):
        for shots in BUDGETS:
            points = [r for r in result['aggregate_combinations'] if r['shots'] == shots]
            ax.errorbar([r['fallback_fraction'] for r in points], [r[metric]['mean'] for r in points],
                yerr=[r[metric]['sample_sd'] for r in points], color=colors[shots], marker='o',
                markersize=4, capsize=3, linewidth=2 if shots == 20 else 1.2,
                label=f'{shots}-shot: 5-seed mean ± SD', alpha=1 if shots == 20 else .8)
        seed11 = sorted((r for r in result['individual_combinations'] if r['shots'] == 20 and r['seed'] == 11),
                        key=lambda r: r['fallback_fraction'])
        ax.plot([r['fallback_fraction'] for r in seed11], [r[metric] for r in seed11],
                linestyle=':', color='#17475d', linewidth=1.7, marker='x', markersize=4, label='20-shot seed 11 control')
        ax.axhline(result['general'][metric], color='#8c3b54', linestyle='--', linewidth=1.3, label='Zero-shot Luna alone')
        ax.axvline(.1, color='#8f96a2', linestyle=':', linewidth=.9)
        ax.set(xlabel='Requests sent to Luna (fallback fraction)', ylabel=title,
               xlim=(-.025, 1.025), ylim=(.70, .88))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks([0, .1, .25, .5, .75, 1])
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Complementary routing on reused validation data', fontsize=16, x=.07, ha='left')
    fig.text(.07, .90, 'BANKING77 · 770 cases · frozen MiniLM + LR and one shared zero-shot Luna response set', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .02), ncol=3, frameon=False, fontsize=9)
    fig.text(.07, .015, 'Bars: sample SD across training seeds, not a confidence interval. 10% fallback is the frozen validation landmark.', fontsize=8)
    fig.subplots_adjust(top=.83, bottom=.26, left=.07, right=.98, wspace=.22)
    fig.savefig(output / 'quality_vs_fallback_fraction.png', dpi=180, metadata={'Software': 'Matplotlib; deterministic validation evidence'})
    fig.savefig(output / 'quality_vs_fallback_fraction.svg', metadata={'Date': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write regenerated summary/chart into a new directory')
    args = parser.parse_args()
    result = reproduce()
    encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'summary.json').write_bytes(encoded)
        plot(result, args.output)
    else:
        require((BUNDLE / 'summary.json').read_bytes() == encoded, 'Summary is not independently reproducible')
    print(json.dumps({'status': 'verified', 'validation_rows': 770, 'combinations': 90,
                      'luna': result['general'], 'fixed_90_percent': result['fixed_90_percent_summary']}, indent=2))


if __name__ == '__main__':
    main()
