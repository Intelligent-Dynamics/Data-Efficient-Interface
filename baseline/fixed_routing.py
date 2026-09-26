"""EXP-007 offline preparation and pure scoring; no test loader or live API mode."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import subprocess

from .data import ROOT, SEEDS, json_bytes, read_json, sha256
from .general_metrics import general_metrics
from .general_protocol import digest, load_protocol, PRICING
from .selective import require
from .thresholds import derive_threshold, use_specialist
from .threshold_inputs import load_inputs

BUNDLE = ROOT / 'experiments/exp007-fixed-threshold-preparation'
PRICING_VERIFIED_DATE = '2026-09-26'
ACCEPTED_COUNT = 693


def evaluate_fixed(specialist_rows, luna_rows, labels, threshold):
    """Offline scorer: confidence alone chooses each branch; failures keep denominator.

    Input rows are already collected, aligned ID/prediction records. No ranking,
    target coverage, API requests, data loading or threshold fitting occurs here.
    """
    def by_id(rows):
        require(rows and all(isinstance(r.get('id'), str) and r['id'] for r in rows), 'Nonempty ID rows required')
        result = {r['id']: r for r in rows}
        require(len(result) == len(rows), 'Duplicate IDs')
        return result
    specialist, luna = by_id(specialist_rows), by_id(luna_rows)
    require(set(specialist) == set(luna), 'Specialist/Luna ID mismatch')
    ids = list(specialist)
    truth, sp, general, routed, accepted = [], [], [], [], []
    for rid in ids:
        row, fallback = specialist[rid], luna[rid]
        require(row['predicted_label'] in labels, 'Invalid specialist prediction')
        require('true_label' not in fallback or fallback['true_label'] == row['true_label'], 'Conflicting truth')
        status = fallback.get('status')
        require(isinstance(status, str) and status, 'Explicit Luna status required')
        pred = fallback.get('predicted_label')
        require(pred in labels if status == 'ok' else pred is None, 'Luna status/prediction mismatch')
        accept = use_specialist(row['confidence'], threshold)
        truth.append(row['true_label']); sp.append(row['predicted_label']); general.append(pred)
        routed.append(row['predicted_label'] if accept else pred); accepted.append(accept)
    fallback_indices = [i for i, value in enumerate(accepted) if not value]
    count, n = sum(accepted), len(ids)
    return {'n_examples': n, 'threshold': threshold, 'accepted_count': count, 'observed_coverage': count / n,
            'luna_request_count': n-count, 'luna_request_percentage': 100*(n-count)/n,
            'luna_requests_note': 'Policy fallback count, separate from all-case baseline collection calls',
            'specialist_only': general_metrics(truth, sp, labels),
            'luna_only': general_metrics(truth, general, labels),
            'routed': general_metrics(truth, routed, labels),
            'fallback_accuracy_on_rejected': (sum(truth[i] == general[i] for i in fallback_indices) / len(fallback_indices)
                                              if fallback_indices else None),
            'accepted_ids': [rid for rid, value in zip(ids, accepted) if value],
            'fallback_ids': [rid for rid, value in zip(ids, accepted) if not value],
            'predictions': [{'id': rid, 'true_label': truth[i], 'predicted_label': routed[i],
                             'source': 'specialist' if accepted[i] else 'gpt-6-luna'} for i, rid in enumerate(ids)]}


def validate_completed_merge(merged, predictions, source, recovery_protocol):
    # General evaluation lists IDs in its own deterministic order. Match the ID
    # population, while requiring predictions to retain the exact original order.
    ids = merged['validation_ids']
    require(len(ids) == len(set(ids)) == 770 and set(ids) == set(source['validation_ids']),
            'Merged validation ID population differs')
    require(merged['study_id'] == 'EXP-006 + EXP-006R recovery' and merged['status'] == 'completed' and
            merged['predictions'] == predictions and
            merged['source_protocol_sha256'] == source['source_protocol_sha256'] and
            merged['recovery_protocol_sha256'] == digest(recovery_protocol) and
            merged['source_tree_sha256'] == source['tree_sha256'], 'Merged validation provenance mismatch')


def validation_usage():
    """Audit completed VALIDATION evidence only; retain failures in accounting context."""
    from .recovery import checked_protocol, load_completed_recovery, DEFAULT_RECOVERY
    from .recovery_source import load_source
    from .recovery_merge import merge_predictions
    from .general import priced_response
    source = load_source()
    recovery_protocol = checked_protocol(source)
    report, entries = load_completed_recovery(source, recovery_protocol, DEFAULT_RECOVERY)
    require(report['status'] == 'completed' and all(p['status'] == 'ok' for p in report['predictions']), 'Recovery not completed')
    merged_path = ROOT / 'artifacts/exp006-plus-exp006r-evaluation-v1/evaluation.json'
    merged = read_json(merged_path)
    predictions = merge_predictions(source['execution']['predictions'], report['predictions'], source['eligible_ids'])
    validate_completed_merge(merged, predictions, source, recovery_protocol)
    counts = Counter(); priced = Decimal(0); sizes = []
    for rid in source['validation_ids']:
        entry = entries[rid] if rid in entries else source['entries'][rid]
        last = entry['attempts'][-1]
        require(last['status'] == 'ok', 'Nominal projection requires successful final response')
        body = last['response']['body']; usage = body['usage']; cost = priced_response(body)
        require(cost is not None, 'Missing validated usage pricing')
        for key in ('input_tokens', 'output_tokens'):
            counts[key] += usage[key]
        for key in ('cached_tokens', 'cache_write_tokens'):
            counts[key] += usage['input_tokens_details'][key]
        counts['reasoning_tokens'] += usage.get('output_tokens_details', {}).get('reasoning_tokens', 0)
        priced += cost
        sizes.append(len(json.dumps(entry['request'], sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()))
    evidence_paths = [Path(source['directory'])/'execution.json', DEFAULT_RECOVERY/'execution.json', merged_path]
    return {'validation_count': len(predictions), 'successful_response_usage': dict(counts),
            'successful_usage_priced_usd': str(priced), 'largest_validation_payload_bytes': max(sizes),
            'original_unknown_usage_attempts': source['execution']['accounting']['unknown_charge_attempts'],
            'recovery_unknown_usage_attempts': report['accounting']['unknown_charge_attempts'],
            'original_source_tree_sha256': source['tree_sha256'],
            'source_files_sha256': {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in evidence_paths},
            'completed_original_successes_preserved': 718, 'completed_recovery_predictions': 52,
            'new_api_calls': 0, 'official_test_access': False}


def estimate_test_cost(usage, expected_test_count):
    require(expected_test_count == 3080 and usage['validation_count'] == 770, 'Unexpected published population')
    rates = PRICING['per_million_tokens']; counts = usage['successful_response_usage']; scale = Decimal(4)
    nominal = Decimal(usage['successful_usage_priced_usd']) * scale
    write_projection = (Decimal(counts['input_tokens'])*Decimal(rates['cache_write']) +
                        Decimal(counts['output_tokens'])*Decimal(rates['output'])) / 1000000 * scale
    max_bytes = usage['largest_validation_payload_bytes']
    input_envelope, output_envelope, attempts = 2 * max_bytes + 8192, 128, 4
    one = (Decimal(input_envelope)*Decimal(rates['cache_write']) + Decimal(output_envelope)*Decimal(rates['output'])) / 1000000 * expected_test_count
    return {'pricing': PRICING, 'pricing_verified_date': PRICING_VERIFIED_DATE,
            'planned_test_rows': expected_test_count, 'planned_first_attempts': expected_test_count,
            'response_sets_across_all_five_specialists': 1, 'max_attempts_per_request': attempts,
            'maximum_attempts': expected_test_count*attempts,
            'nominal_usage_projection_usd': str(nominal), 'same_tokens_all_input_cache_write_usd': str(write_projection),
            'conservative_one_attempt_usd': str(one), 'conservative_all_attempts_usd': str(one*attempts),
            'assumed_max_payload_bytes': max_bytes, 'input_token_envelope_per_attempt': input_envelope,
            'output_token_envelope_per_attempt': output_envelope,
            'assumptions': ['Nominal: test token distribution equals successful validation responses; one attempt, observed cache behavior.',
                            'Envelope: every future request fits the observed validation byte maximum; bytes are only an engineering token heuristic.',
                            'All-input cache-write rate and full output allowance; no cache-read discount in the conservative estimates.',
                            'All 3080 cases need Luna-only baseline predictions once, reused across seeds; not only rejected cases.',
                            'Test payloads have NOT been read. This estimate is not a guaranteed invoice or approved cap.',
                            'After separate test authorization, recompute all payload reservations before any paid call; halt if cap is insufficient.',
                            'Missing/failure usage remains unknown. Actual collection spend includes failures/retries.',
                            'Specialist deployment cost and total-system savings remain unmeasured.'],
            'live_authorization': False, 'approved_spending_cap_usd': None,
            'api_calls_during_preparation': 0, 'actual_preparation_spend_usd': '0'}


def summarize_thresholds(inputs):
    require([r['seed'] for r in inputs['runs']] == list(SEEDS), 'Five ordered seeds required')
    summaries = []
    for run in inputs['runs']:
        rows = run['selective']['ranked_rows']
        require(len(rows) == 770, 'Expected existing validation population')
        landmark = next(p for p in run['selective']['landmarks'] if p['target_coverage'] == .9)
        require(landmark['accepted_count'] == ACCEPTED_COUNT, 'Frozen acceptance count changed')
        result = derive_threshold([r['id'] for r in rows], [r['confidence'] for r in rows], ACCEPTED_COUNT)
        require(result['original_accepted_ids'] == [r['id'] for r in rows[:ACCEPTED_COUNT]], 'Original prefix mismatch')
        actual = [r['id'] for r in rows if use_specialist(r['confidence'], result['threshold'])]
        require(set(actual) == set(result['fixed_accepted_ids']), 'Independent single-request gate disagrees')
        ids_fields = ('original_accepted_ids', 'fixed_accepted_ids')
        small = {k: v for k, v in result.items() if k not in ids_fields}
        small.update({k+'_sha256': digest(sorted(result[k])) for k in ids_fields})
        small.update(seed=run['seed'], run_id=run['run_id'], shots=20,
                     confidence_below_rank_boundary=rows[ACCEPTED_COUNT]['confidence'],
                     classifier_path=run['classifier_path'], classifier_sha256=run['classifier_sha256'],
                     probabilities_sha256=run['probabilities_sha256'])
        summaries.append(small)
    return summaries


def make_test_protocol(inputs, thresholds, costs):
    from .recovery import POLICY
    luna, _, _ = load_protocol()
    require(inputs['labels'] == luna['labels'], 'Frozen specialist/Luna class names differ')
    require([t['seed'] for t in thresholds] == list(SEEDS), 'Freeze all five seeds in order')
    return {'study_id': 'EXP-007', 'protocol_version': 'exp007-fixed-threshold-test-v1',
            'status': 'FROZEN PREPARATION; OFFICIAL TEST NOT RUN',
            'decision_rule': 'specialist_confidence >= per_seed_threshold -> specialist; otherwise gpt-6-luna',
            'confidence': 'exact binary64 maximum LR class probability; UNCALIBRATED; no rounding or tolerance',
            'threshold_derivation': {'source': 'EXP-005 20-shot 90% validation landmark, five seeds',
                                    'accepted_count': 693, 'validation_count': 770,
                                    'tie_rule': 'At rank-693 score q, include all q ties or exclude all using nextafter(q,+inf); minimize symmetric difference from frozen rank prefix; equal distances include all ties.',
                                    'ranking_tie_rule': 'Original EXP-005 descending confidence, SHA256(UTF-8 ID), then ID; runtime gate never uses IDs or ranks.',
                                    'label_use': 'IDs and confidence only in derivation; 20-shot/90% choice was informed by prior validation results.',
                                    'validation_labels_already_used': 770, 'new_labels': 0, 'calibration_fits': 0},
            'thresholds': {str(t['seed']): {'value': t['threshold'], 'decimal': t['threshold_decimal'], 'binary64_hex': t['threshold_hex'],
                                          'classifier_path': t['classifier_path'], 'classifier_sha256': t['classifier_sha256']} for t in thresholds},
            'specialists': {'shots': 20, 'seeds': list(SEEDS), 'embedding': inputs['embedding'],
                            'refit': False, 'recalibrate': False, 'tune': False, 'training_labels_per_seed': 1540,
                            'runtime': 'Reuse pinned CPU float32 embedding settings and saved classifier. Probabilities/gate use binary64; changing runtime/precision requires revalidation without test tuning.'},
            'population': {'dataset': inputs['dataset']['dataset'], 'revision': inputs['dataset']['revision'],
                           'split': 'official test', 'expected_count_from_existing_metadata': 3080,
                           'sealed_file_sha256_from_existing_metadata': inputs['dataset']['files']['test.csv']['sha256'],
                           'row_ids': 'test:{original_zero_based_csv_row:05d}', 'labels': inputs['labels'],
                           'include_all_rows': True, 'preparation_read_test_file': False,
                           'unseal_requirement': 'Separate explicit user authorization for the one-time official test evaluation; no test bytes, hashes or labels read during preparation.'},
            'fallback': {'source_protocol_sha256': digest(luna), 'endpoint': luna['endpoint'],
                         'settings': luna['settings'], 'prompt_sha256': luna['prompt_sha256'], 'schema_sha256': luna['schema_sha256'],
                         'prompt_path': 'experiments/exp006-general-model-preparation/prompt.txt',
                         'schema_path': 'experiments/exp006-general-model-preparation/schema.json',
                         'alias_limitation': 'No documented fixed snapshot; require returned model ID gpt-6-luna and record timestamp/ID; underlying provider weights may change.',
                         'response_sets': 1, 'planned_all_case_requests': 3080,
                         'payload': 'One independent customer message, unchanged zero-shot prompt/schema; no test label, specialist output, confidence, ID or demonstrations.',
                         'reuse': 'One saved prediction per test ID shared by all five configurations. Identical request payloads may reuse an exact compatible cache entry, with explicit per-ID aliases; never collapse evaluation rows.',
                         'retry_policy': POLICY, 'auth': 'New explicit live authorization, protocol hash, current price acknowledgement and positive numeric spending cap; previous approvals do not transfer.',
                         'preflight': 'Once separately authorized to unseal, compute complete payload/output/retry reservations before paid calls; halt for revised spending approval if insufficient. No silent truncation, excluded rows, prompt or threshold changes.'},
            'one_time_sequence': ['Verify frozen code/protocol/model/package hashes and separate unseal/live approval before opening test data.',
                                  'Verify official source bytes/counts/77 labels after authorization, retaining all 3080 original IDs.',
                                  'Hash-check saved encoders/classifiers; run all five existing specialists without fitting. Persist probability matrices, predictions, exact threshold decisions before scoring.',
                                  'Apply the scalar gate independently to every request. Never rank test scores, recompute percentiles, adjust thresholds, or force 90% test coverage.',
                                  'Collect one capped durable Luna response set for all test IDs; reuse across all seeds. Preserve refusals/incomplete/invalid/failed answers; never retry correctness.',
                                  'After all planned prediction artifacts are frozen, join truth by ID and evaluate all cases once. Resume collection only within the frozen retry allowance.',
                                  'Publish every seed and aggregates, all failures, label budget, observed coverage and API accounting. Never revise this protocol based on test outcomes.'],
            'evaluation': {'primary_metric': 'macro-F1, fixed 77 labels; zero division=0',
                           'metrics': ['specialist-only accuracy/macro-F1', 'Luna-only accuracy/macro-F1', 'routed accuracy/macro-F1',
                                       'observed specialist coverage/count', 'Luna fallback count/percentage', 'fallback accuracy on rejected subset',
                                       'per-class support/precision/recall/F1 and acceptance counts', 'unresolved statuses and attempt counts'],
                           'denominator': 'All 3080 test IDs for all three full-workload metrics; unresolved Luna predictions are None and count as incorrect when used.',
                           'empty_rejected_subset': 'fallback accuracy undefined/null',
                           'seeds': 'Report all five seeds and mean/sample SD (ddof=1), never best seed only. Same test and Luna responses; not five independent test datasets.',
                           'label_budget': '1540 fitting +770 validation labels per specialist; zero new fitting/calibration labels; 3080 test labels used only for eventual scoring.',
                           'selection_limitations': '20-shot/90% selected after exploratory validation; no calibrated confidence, prespecified non-inferiority margin, production-quality guarantee or total-system savings claim.'},
            'accounting': {'cost_estimate_sha256': digest(costs),
                           'separation': 'Actual all-case Luna collection spend including retries is separate from hypothetical routed API charges. Policy fallback count is distinct from baseline collection count.',
                           'specialist_deployment_cost': None, 'total_system_savings': None, 'production_latency': None},
            'outputs': {'root': 'artifacts/exp007-fixed-threshold-test-v1',
                        'files': ['protocol.json', 'manifest.json', 'specialists/<seed>/predictions.json', 'specialists/<seed>/probabilities.json',
                                  'luna/responses/<cache-key>.json', 'luna/reservations.json', 'luna/execution.json', 'evaluation.json', 'verification.json'],
                        'preserve_prior_artifacts': True, 'raw_responses_and_large_arrays_ignored': True},
            'input_files_sha256': inputs['source_files_sha256']}


def prepare(output):
    require(not output.exists(), 'Use a fresh output directory; preserve previous evidence')
    inputs = load_inputs()
    usage = validation_usage()
    thresholds = summarize_thresholds(inputs)
    costs = estimate_test_cost(usage, inputs['dataset']['expected_counts']['test'])
    protocol = make_test_protocol(inputs, thresholds, costs)
    # Recheck all read-only validation inputs before any output write.
    for name, expected in {**inputs['source_files_sha256'], **usage['source_files_sha256']}.items():
        require(sha256((ROOT / name).read_bytes()) == expected, 'Input changed during preparation')
    code_paths = ['baseline/thresholds.py', 'baseline/threshold_inputs.py', 'baseline/fixed_routing.py',
                  'baseline/general_metrics.py', 'pyproject.toml', 'uv.lock']
    manifest = {'study_id': 'EXP-007', 'status': 'PREPARATION ONLY; OFFICIAL TEST NOT RUN',
                'generated_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': digest(protocol),
                'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'code_files_sha256': {name: sha256((ROOT/name).read_bytes()) for name in code_paths},
                'official_test_access': False, 'model_fits': 0, 'model_executions': 0, 'api_calls': 0,
                'observed_validation_coverage_by_seed': {str(t['seed']): t['actual_coverage'] for t in thresholds}}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('thresholds.json', thresholds), ('cost_estimate.json', costs), ('validation_completion.json', usage),
                        ('protocol.json', protocol), ('manifest.json', manifest)]:
        (output/name).write_bytes(json_bytes(value))
    (output/'protocol_sha256.txt').write_text(digest(protocol)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == '__main__':
    main()
