"""Guarded one-time EXP-007 execution. Default dry-run keeps official test sealed."""
import argparse
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import statistics

from .data import ROOT, read_json, sha256
from .general import atomic_json, lock, now, spending_cap
from .general_protocol import digest, PRICING
from .final_protocol import (Authorization, BUNDLE, FROZEN_PROTOCOL_SHA256, load_frozen,
                             runtime_code, safe_output, scoring_truth, unseal_inputs,
                             validate_protocol, verify_preparation)
from .final_collection import collect, collection_plan, verify_completed
from .final_specialists import run_specialists, verify_specialist_files
from .fixed_routing import evaluate_fixed
from .selective import require
from .thresholds import use_specialist


def dry_run(root=ROOT):
    protocol, _, _ = load_frozen(root)
    costs = read_json(Path(root) / BUNDLE / 'cost_estimate.json')
    return {'study_id': 'EXP-007', 'status': 'SEALED METADATA-ONLY DRY RUN',
            'protocol_sha256': FROZEN_PROTOCOL_SHA256, 'official_test_opened': False,
            'api_calls': 0, 'model_executions': 0, 'planned_test_rows': 3080,
            'planned_initial_requests_at_most': 3080, 'shared_luna_response_sets': 1,
            'nominal_usage_projection_usd': costs['nominal_usage_projection_usd'],
            'conditional_conservative_maximum_usd': costs['conservative_all_attempts_usd'],
            'assumed_max_payload_bytes': costs['assumed_max_payload_bytes'],
            'estimate_note': 'Frozen validation-based estimate only. Actual test payloads remain unread; authorized preflight recomputes every reservation before any call.',
            'pricing': PRICING, 'pricing_currently_verified_by_runner': False,
            'authorization_required_by_mode': {
                'dry-run': [],
                'preflight': ['exact protocol hash', 'test access'],
                'live': ['exact protocol hash', 'test access', 'live API', 'positive USD cap',
                         'current UTC date acknowledgement of unchanged official prices']},
            'required_live_environment': ['OPENAI_API_KEY'], 'optional_live_environment': ['OPENAI_PROJECT_ID'],
            'thresholds': {seed: t['value'] for seed, t in protocol['thresholds'].items()}}


def prepare_authorized(authorization, root=ROOT, *, live=True):
    protocol, prompt, schema = load_frozen(root)
    cap = authorization.validate(protocol, live=live)
    verify_preparation(root, protocol)
    # Hashes/packages must be checked before even the authorized CSV unseal.
    verify_specialist_files(protocol, root=root)
    rows = unseal_inputs(authorization, protocol, root) if live else unseal_inputs(authorization, protocol, root, live=False)
    plan = collection_plan(rows, protocol, prompt, schema)
    if live:
        require(Decimal(plan['full_retry_reservation_usd']) <= cap,
                'Full-run reservation exceeds approved cap: USD ' + plan['full_retry_reservation_usd'])
    return protocol, prompt, schema, rows, plan


def preflight(authorization, root=ROOT):
    # Optional cap comparison is planning, never authorization to spend. No date,
    # credentials, paid transport or model execution is needed on this code path.
    proposed_cap = None
    if authorization.spending_cap_usd not in ('', None):
        require(not isinstance(authorization.spending_cap_usd, bool), 'Positive numeric USD cap required')
        proposed_cap = spending_cap(authorization.spending_cap_usd)
    protocol, _, _, _, plan = prepare_authorized(authorization, root, live=False)
    return {'study_id': 'EXP-007', 'status': 'AUTHORIZED PREFLIGHT ONLY; NO INFERENCE',
            'protocol_sha256': digest(protocol), 'official_test_opened': True, 'api_calls': 0,
            'model_executions': 0, 'test_rows': 3080, 'label_count': 77,
            'dataset_revision': protocol['population']['revision'],
            'dataset_sha256': protocol['population']['sealed_file_sha256_from_existing_metadata'],
            'authorization_scope': 'test access only; no API authorization is granted or consumed',
            'required_cap_usd': plan['full_retry_reservation_usd'],
            'provided_cap_usd': str(proposed_cap) if proposed_cap is not None else None,
            'cap_sufficient': Decimal(plan['full_retry_reservation_usd']) <= proposed_cap if proposed_cap is not None else None,
            'pricing_currently_verified_by_runner': False,
            'pricing_note': 'Reservations use frozen rates; live still requires current unchanged-price acknowledgement.',
            'authorization': asdict(authorization), 'plan': plan}


def _persist_exact(path, value):
    if path.exists():
        require(read_json(path) == value, 'Immutable run artifact changed: ' + path.name)
    else:
        atomic_json(path, value)


def _freeze_predictions(output, protocol):
    execution = read_json(output / 'luna/execution.json')
    require(execution['status'] in ('completed', 'finished_with_unresolved'),
            'Collection must finish before any test labels are joined')
    require(len(execution['predictions']) == 3080 and
            [r['id'] for r in execution['predictions']] == [f'test:{i:05d}' for i in range(3080)],
            'Luna prediction population changed')
    require(all(r['status'] not in ('not_attempted', 'reserved')
                for r in execution['predictions']), 'Incomplete collection cannot be scored')
    paths = [output / 'manifest.json', output / 'protocol.json', output / 'preflight.json']
    for name in ['specialists', 'luna']:
        paths.extend(p for p in (output / name).rglob('*.json') if p.name != 'authorizations.json')
    require(all((output / f'specialists/{s}/{name}.json').exists()
                for s in protocol['specialists']['seeds'] for name in ('predictions', 'probabilities', 'metadata')),
            'Missing specialist prediction evidence')
    frozen = {'protocol_sha256': digest(protocol), 'n_examples': 3080,
              'files_sha256': {str(p.relative_to(output)): sha256(p.read_bytes()) for p in sorted(paths)}}
    _persist_exact(output / 'predictions_frozen.json', frozen)
    return frozen


def verify_frozen_predictions(output, protocol):
    frozen = read_json(output / 'predictions_frozen.json')
    require(frozen['protocol_sha256'] == digest(protocol) and frozen['n_examples'] == 3080,
            'Prediction freeze provenance mismatch')
    # Reconstruct both file membership and hashes; an omitted file cannot escape auditing.
    require(_freeze_predictions(output, protocol) == frozen, 'Frozen prediction inventory mismatch')
    return frozen


def evaluate_predictions(specialists, luna, truth, protocol):
    """Scoring-only label join; routes must already be fixed and persisted."""
    validate_protocol(protocol)
    ids = [f'test:{i:05d}' for i in range(3080)]
    require(set(truth) == set(ids), 'Scoring truth population changed')
    require([r['id'] for r in luna] == ids, 'Shared Luna rows must retain every test ID exactly once')
    require(set(specialists) == set(protocol['thresholds']), 'All five frozen specialists required')
    labels, seeds = protocol['population']['labels'], protocol['specialists']['seeds']
    results = {}
    for seed in seeds:
        rows = specialists[str(seed)]
        require([r['id'] for r in rows] == ids, 'Specialist population changed')
        threshold = protocol['thresholds'][str(seed)]['value']
        require(all(set(r) == {'id', 'predicted_label', 'confidence', 'use_specialist'} and
                    type(r['use_specialist']) is bool and r['use_specialist'] == use_specialist(r['confidence'], threshold)
                    for r in rows), 'Persisted scalar gate changed; labels must not influence routing')
        score_rows = [{**r, 'true_label': truth[r['id']]} for r in rows]
        result = evaluate_fixed(score_rows, luna, labels, threshold)
        accepted = Counter(truth[r['id']] for r in rows if r['use_specialist'])
        support = Counter(truth.values())
        result['per_class_acceptance'] = {label: {'accepted_count': accepted[label],
                                                  'rejected_count': support[label] - accepted[label],
                                                  'support': support[label]} for label in labels}
        result['fallback_unresolved_count'] = sum(r['status'] != 'ok' for r, s in zip(luna, rows) if not s['use_specialist'])
        result['seed'] = seed
        results[str(seed)] = result
    aggregate = {}
    for key in ('accepted_count', 'observed_coverage', 'luna_request_count', 'luna_request_percentage', 'fallback_accuracy_on_rejected'):
        values = [r[key] for r in results.values()]
        defined = [v for v in values if v is not None]
        aggregate[key] = {'mean': statistics.mean(defined) if defined else None,
                          'sample_sd': statistics.stdev(defined) if len(defined) > 1 else None,
                          'defined_seeds': len(defined)}
    for model in ('specialist_only', 'routed'):
        aggregate[model] = {metric: {'mean': statistics.mean(r[model][metric] for r in results.values()),
                                    'sample_sd': statistics.stdev(r[model][metric] for r in results.values())}
                            for metric in ('accuracy', 'macro_f1')}
    return {'study_id': 'EXP-007', 'evaluation_split': 'official test', 'n_examples': 3080,
            'protocol_sha256': digest(protocol), 'luna_only': results['11']['luna_only'], 'seeds': results,
            'seed_summary': aggregate, 'sd_definition': 'sample SD, ddof=1; one shared test population and Luna response set',
            'luna_status_counts': dict(Counter(r['status'] for r in luna)),
            'unresolved_count': sum(r['status'] != 'ok' for r in luna),
            'label_budget': {'fitting_per_specialist': 1540, 'additional_validation_labels': 770, 'test_scoring_labels': 3080},
            'specialist_deployment_cost': None, 'total_system_savings': None, 'production_latency': None}


def execute(authorization, root=ROOT, *, transport=None, sleep=None, clock=None):
    protocol, prompt, schema, rows, plan = prepare_authorized(authorization, root)
    output = safe_output(root, protocol)
    cap = str(authorization.validate(protocol, live=True))
    manifest = {'study_id': 'EXP-007', 'protocol_sha256': digest(protocol), 'rows_sha256': digest(rows),
                'code_files_sha256': runtime_code(root), 'cap_usd': cap,
                'test_sha256': protocol['population']['sealed_file_sha256_from_existing_metadata']}
    with lock(output):
        if not (output / 'manifest.json').exists():
            require(not any(p.name != '.lock' for p in output.iterdir()), 'New test output must be empty')
        _persist_exact(output / 'manifest.json', manifest)
        _persist_exact(output / 'protocol.json', protocol)
        _persist_exact(output / 'preflight.json', plan)
        authorizations = read_json(output / 'authorizations.json') if (output / 'authorizations.json').exists() else []
        authorizations.append({'checked_utc': now(), **asdict(authorization)})
        atomic_json(output / 'authorizations.json', authorizations)
        if not (output / 'predictions_frozen.json').exists():
            require(not (output / 'evaluation.json').exists(), 'Scored output without frozen predictions')
            run_specialists(rows, protocol, output, root=root)
            options = {name: value for name, value in [('transport', transport), ('sleep', sleep), ('clock', clock)] if value is not None}
            report, _ = collect(rows, protocol, prompt, schema, output / 'luna', authorization, **options)
            if report['status'] not in ('completed', 'finished_with_unresolved'):
                return {'status': report['status'], 'halt_reason': report.get('halt_reason'),
                        'scoring_performed': False, 'labels_joined': False, 'accounting': report['accounting']}
            verify_completed(rows, protocol, prompt, schema, output / 'luna', authorization)
            _freeze_predictions(output, protocol)
        frozen = verify_frozen_predictions(output, protocol)
        if (output / 'evaluation_checkpoint.json').exists():
            checkpoint = read_json(output / 'evaluation_checkpoint.json')
            require(checkpoint['verification']['evaluation_sha256'] == digest(checkpoint['evaluation']) and
                    checkpoint['verification']['prediction_freeze_sha256'] == digest(frozen) and
                    checkpoint['evaluation']['protocol_sha256'] == digest(protocol) and
                    checkpoint['evaluation']['prediction_freeze_sha256'] == digest(frozen),
                    'Completed evaluation checkpoint changed')
            _persist_exact(output / 'evaluation.json', checkpoint['evaluation'])
            _persist_exact(output / 'verification.json', checkpoint['verification'])
            return checkpoint['evaluation']
        require(not (output / 'evaluation.json').exists() and not (output / 'verification.json').exists(),
                'Scored output without its atomic evaluation checkpoint')
        # The only truth join is below the durable, immutable prediction freeze.
        truth = scoring_truth(authorization, protocol, root, frozen_predictions_verified=True)
        specialists = {str(s): read_json(output / f'specialists/{s}/predictions.json') for s in protocol['specialists']['seeds']}
        luna_report = read_json(output / 'luna/execution.json')
        result = evaluate_predictions(specialists, luna_report['predictions'], truth, protocol)
        result.update(prediction_freeze_sha256=digest(frozen), api_accounting=luna_report['accounting'],
                      collection_status=luna_report['status'], evaluated_utc=now())
        verification = {'protocol_sha256': digest(protocol),
                    'prediction_freeze_sha256': digest(frozen), 'evaluation_sha256': digest(result),
                    'all_3080_ids_retained': True, 'thresholds_recomputed': False,
                    'predictions_frozen_before_label_join': True, 'all_five_seeds_reported': True}
        # One durable transaction lets a crash between the two public files resume
        # without rejoining labels, re-running inference or rescoring.
        _persist_exact(output / 'evaluation_checkpoint.json', {'evaluation': result, 'verification': verification})
        _persist_exact(output / 'evaluation.json', result)
        _persist_exact(output / 'verification.json', verification)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['dry-run', 'preflight', 'live'], nargs='?', default='dry-run')
    parser.add_argument('--approved-protocol-sha256', default='')
    parser.add_argument('--authorize-test-access', action='store_true')
    parser.add_argument('--authorize-live', action='store_true')
    parser.add_argument('--spending-cap-usd', default='', help='Required approved cap for live; optional proposed cap comparison for preflight')
    parser.add_argument('--acknowledge-current-pricing-date', default='',
                        help='UTC date on which operator verified official rates still equal frozen prices')
    args = parser.parse_args(argv)
    authorization = Authorization(args.approved_protocol_sha256, args.authorize_test_access,
                                  args.authorize_live, args.spending_cap_usd, args.acknowledge_current_pricing_date)
    try:
        result = dry_run() if args.mode == 'dry-run' else preflight(authorization) if args.mode == 'preflight' else execute(authorization)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, 'EXP-007 stopped: ' + str(exc) + '\n')
    # Avoid printing request text, response caches or 5x3080 row predictions.
    if args.mode == 'live':
        result = {key: value for key, value in result.items() if key not in ('seeds', 'luna_only')}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
