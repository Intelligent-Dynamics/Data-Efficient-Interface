"""Synthetic exact-source resume receipts; no real EXP-009/data/model/API access."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

import pytest

from baseline import general, recovery, retrieved_collection as collector, retrieved_run
from baseline import retrieved_compatibility as compatibility
from baseline.data import ROOT, json_bytes, read_json, sha256
from baseline.general_protocol import ENDPOINT, PRICING, SETTINGS, cache_key, digest, estimate, payload


def forbidden(*args, **kwargs):
    pytest.fail('Synthetic compatibility check attempted protected data/model/network access')


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setattr(recovery, 'post_recovery', forbidden)
    monkeypatch.setattr(general, 'post_openai', forbidden)
    monkeypatch.setattr('socket.create_connection', forbidden)
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        resolved = str(path.absolute())
        if any(resolved.startswith(str(ROOT / name) + '/') for name in
               ('data/raw', 'data/processed', 'artifacts', '.cache/huggingface')):
            forbidden()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    monkeypatch.setattr(retrieved_run, 'prepare_test', forbidden)
    monkeypatch.setattr(retrieved_run, 'unseal_inputs', forbidden)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def response():
    return {'http_status': 200, 'request_id': 'SYNTHETIC', 'headers': {}, 'body': {
        'model': 'gpt-6-luna', 'service_tier': 'default', 'status': 'completed',
        'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"intent":"a"}'}]}],
        'usage': {'input_tokens': 100, 'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
                  'output_tokens': 10, 'output_tokens_details': {'reasoning_tokens': 0}},
    }}


def approve(case):
    write(case['receipt_path'], case['receipt'])
    case['approval'] = replace(case['approval'],
        resume_compatibility_sha256=sha256(case['receipt_path'].read_bytes()))


@pytest.fixture
def case(tmp_path, monkeypatch):
    schema = {'type': 'object', 'properties': {'intent': {'type': 'string', 'enum': ['a', 'b']}},
              'required': ['intent'], 'additionalProperties': False}
    prompt = 'SYNTHETIC classification only'
    protocol = {'study_id': 'SYNTHETIC EXP-009', 'settings': deepcopy(SETTINGS), 'pricing': deepcopy(PRICING),
                'retry_policy': deepcopy(recovery.POLICY), 'endpoint': ENDPOINT,
                'prompt_sha256': sha256(prompt.encode()), 'schema_sha256': digest(schema),
                'schema': schema, 'labels': ['a', 'b'],
                'outputs': {'test': str(compatibility.RUN)}}
    monkeypatch.setattr(compatibility, 'PROTOCOL_SHA256', digest(protocol))
    ids = [f'test:{i:05d}' for i in range(3080)]
    requests = {rid: payload('SYNTHETIC query ' + rid, prompt, schema) for rid in ids}
    reserve = str(sum((Decimal(estimate(body)['attempt_reservation_usd']) * 4
                       for body in requests.values()), Decimal(0)))
    old_source = {
        'code_sha256': {'retrieved.py': '1' * 64, 'retrieved_collection.py': '2' * 64,
                        'retrieved_run.py': '3' * 64},
        'reused_code': {'git_head': '1' * 40, 'files_sha256': {'baseline/general.py': '4' * 64},
                        'python': 'SYNTHETIC', 'platform': 'SYNTHETIC', 'packages': {}, 'http_client': 'SYNTHETIC'},
        'reused_helper_sha256': {'recovery.py': '5' * 64},
    }
    old = {**compatibility._bound_files(old_source), 'baseline/final_collection.py': '6' * 64}
    new = {**old, **{name: '7' * 64 for name in compatibility.CHANGED}}
    new_source = deepcopy(old_source)
    new_source['code_sha256'].update({Path(name).name: new[name] for name in compatibility.CHANGED})
    new_source['reused_code']['git_head'] = '8' * 40
    new_source['reused_helper_sha256']['final_collection.py'] = old['baseline/final_collection.py']
    monkeypatch.setattr(compatibility, 'runtime_code', lambda root: deepcopy(new))
    monkeypatch.setattr(collector, 'ROOT', tmp_path)
    monkeypatch.setattr(collector, 'source_record', lambda: deepcopy(new_source))
    manifest = {'study_id': 'EXP-009', 'kind': 'synthetic', 'scope': 'test',
                'protocol_sha256': digest(protocol), 'requests_sha256': digest(requests),
                'cap_usd': '100', 'full_retry_reservation_usd': reserve, **old_source}
    run = tmp_path / compatibility.RUN
    output = run / 'collection'
    write(output / 'manifest.json', manifest)
    write(tmp_path / 'experiments/exp009-retrieved-luna/protocol.json', protocol)
    write(run / 'requests.json', requests)
    write(run / 'prepared.json', {'protocol_sha256': digest(protocol), 'scope': 'test', 'ids': ids,
                                 'requests_sha256': digest(requests), 'reservation_usd': reserve})
    for name in ('retrieval_evidence.json', 'specialist.json', 'overhead.json'):
        write(run / name, {'notice': 'SYNTHETIC fixture'})
    counts = {rid: 1 for rid in ids[:2]}
    for rid in counts:
        body = requests[rid]
        saved = response()
        a = {'attempt': 1, 'reservation_usd': estimate(body)['attempt_reservation_usd'],
             'response': saved, 'response_sha256': digest(saved), **recovery.outcome(saved, protocol['labels']),
             'usage_priced_usd': str(general.priced_response(saved['body'])),
             'usage': saved['body']['usage'], 'returned_model': saved['body']['model'],
             'returned_service_tier': saved['body']['service_tier'], 'started_epoch': 100.,
             'finished_epoch': 101., 'retry_after_seconds': 0.}
        write(output / 'responses' / (rid.replace(':', '_') + '.json'),
              {'study_id': 'EXP-009', 'kind': 'synthetic', 'id': rid, 'cache_key': cache_key(body),
               'request': body, 'protocol_sha256': digest(protocol), 'attempts': [a]})
    ledger = {'manifest_sha256': digest(manifest), 'attempt_counts': counts}
    write(output / 'reservations.json', ledger)
    receipt = {
        'study_id': 'EXP-009', 'protocol_sha256': digest(protocol), 'run_directory': str(compatibility.RUN),
        'original_runtime_files_sha256': old, 'patched_runtime_files_sha256': deepcopy(new),
        'changed_source_files': sorted(compatibility.CHANGED),
        'original_manifest_sha256': sha256((output / 'manifest.json').read_bytes()),
        'requests_sha256': digest(requests), 'cap_usd': '100', 'full_retry_reservation_usd': reserve,
        'original_git_head': old_source['reused_code']['git_head'],
        'prepared_files_sha256': {name: sha256((run / name).read_bytes()) for name in compatibility.PREPARED},
        'original_ledger': deepcopy(ledger), 'original_ledger_sha256': digest(ledger),
        'preserved_response_files_sha256': {str(p.relative_to(run)): sha256(p.read_bytes())
                                            for p in (output / 'responses').glob('*.json')},
    }
    value = {'root': tmp_path, 'run': run, 'output': output, 'receipt': receipt,
             'receipt_path': tmp_path / compatibility.RECEIPT, 'manifest': manifest,
             'expected': {**manifest, **new_source}, 'source': new_source, 'new': new,
             'requests': requests, 'protocol': protocol,
             'approval': collector.Approval(digest(protocol), True, '100',
                  datetime.now(timezone.utc).date().isoformat(), True)}
    approve(value)
    return value


def accept(case, expected=None):
    return compatibility.accepted_manifest(case['output'] / 'manifest.json',
        case['expected'] if expected is None else expected, case['approval'], root=case['root'])


def test_exact_approved_transition_is_readonly_and_keeps_historical_manifest(case):
    before = {p: p.read_bytes() for p in case['run'].rglob('*.json')}
    assert accept(case) == case['manifest']
    assert before == {p: p.read_bytes() for p in case['run'].rglob('*.json')}


@pytest.mark.parametrize('approval', ['', '0' * 64])
def test_missing_or_wrong_receipt_approval_rejects_source_drift(case, approval):
    case['approval'] = replace(case['approval'], resume_compatibility_sha256=approval)
    with pytest.raises(ValueError):
        accept(case)


def test_receipt_bytes_cannot_change_after_approval(case):
    case['receipt']['cap_usd'] = '999'
    write(case['receipt_path'], case['receipt'])
    with pytest.raises(ValueError, match='hash mismatch'):
        accept(case)


def test_arbitrary_later_runtime_drift_is_rejected(case):
    case['new']['baseline/general.py'] = '9' * 64
    with pytest.raises(ValueError, match='Patched runtime differs'):
        accept(case)


@pytest.mark.parametrize('field', ['cap_usd', 'requests_sha256', 'full_retry_reservation_usd', 'protocol_sha256', 'scope'])
def test_receipt_cannot_change_experimental_or_budget_binding(case, field):
    expected = deepcopy(case['expected'])
    expected[field] = 'changed'
    with pytest.raises(ValueError):
        accept(case, expected)


def test_receipt_cannot_change_runtime_environment(case):
    expected = deepcopy(case['expected'])
    expected['reused_code']['packages'] = {'SYNTHETIC': 'new'}
    with pytest.raises(ValueError, match='environment'):
        accept(case, expected)


@pytest.mark.parametrize('name', ['requests.json', 'retrieval_evidence.json', 'collection/manifest.json',
                                  'collection/responses/test_00000.json'])
def test_prepared_or_historical_evidence_tampering_is_rejected(case, name):
    path = case['run'] / name
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        accept(case)


@pytest.mark.parametrize('fault', ['truncate', 'retry_success', 'binding', 'bytes', 'unrecorded_response'])
def test_ledger_cannot_be_truncated_tampered_or_give_success_another_attempt(case, fault):
    path = case['output'] / 'reservations.json'
    ledger = read_json(path)
    if fault == 'truncate':
        del ledger['attempt_counts']['test:00000']
    elif fault == 'retry_success':
        ledger['attempt_counts']['test:00000'] = 2
    elif fault == 'binding':
        ledger['manifest_sha256'] = '0' * 64
    elif fault == 'unrecorded_response':
        ledger['attempt_counts']['test:00002'] = 1
    if fault == 'bytes':
        path.write_bytes(path.read_bytes() + b' ')
    else:
        write(path, ledger)
    with pytest.raises(ValueError):
        accept(case)


@pytest.mark.parametrize('fault', ['missing_response_binding', 'other_source_transition', 'other_run'])
def test_even_approved_receipt_cannot_expand_its_exact_scope(case, fault):
    if fault == 'missing_response_binding':
        case['receipt']['preserved_response_files_sha256'].pop('collection/responses/test_00000.json')
    elif fault == 'other_source_transition':
        case['receipt']['changed_source_files'].append('baseline/general.py')
    else:
        case['receipt']['run_directory'] = 'artifacts/replacement'
    approve(case)
    with pytest.raises(ValueError):
        accept(case)


def test_compatibility_cannot_create_a_replacement_run(case):
    target = case['root'] / 'replacement/manifest.json'
    with pytest.raises(ValueError, match='new/replacement'):
        compatibility.accepted_manifest(target, case['expected'], case['approval'], root=case['root'])
    assert not target.parent.exists()


def test_compatibility_cannot_recreate_a_missing_original_manifest(case):
    (case['output'] / 'manifest.json').unlink()
    with pytest.raises(ValueError, match='new/replacement'):
        accept(case)
    assert not (case['output'] / 'manifest.json').exists()


def test_resume_provenance_is_separate_immutable_and_requires_approval(case):
    before = {p: p.read_bytes() for p in case['run'].rglob('*.json')}
    compatibility.record_resume_source(case['output'], case['approval'], case['source'], root=case['root'])
    path = case['output'] / 'source_compatibility.json'
    assert read_json(path) == {'receipt_sha256': case['approval'].resume_compatibility_sha256,
                              'source': case['source']}
    assert all(p.read_bytes() == raw for p, raw in before.items())
    compatibility.record_resume_source(case['output'], case['approval'], case['source'],
                                       root=case['root'], write=False)
    changed = deepcopy(case['source'])
    changed['reused_code']['git_head'] = '9' * 40
    with pytest.raises(ValueError):
        compatibility.record_resume_source(case['output'], case['approval'], changed, root=case['root'])
    with pytest.raises(ValueError):
        compatibility.record_resume_source(case['output'],
            replace(case['approval'], resume_compatibility_sha256=''), case['source'], root=case['root'])


def test_readonly_provenance_check_cannot_create_a_record(case):
    with pytest.raises(ValueError, match='Missing patched'):
        compatibility.record_resume_source(case['output'], case['approval'], case['source'],
                                           root=case['root'], write=False)
    assert not (case['output'] / 'source_compatibility.json').exists()


def test_approved_resume_skips_saved_successes_and_keeps_remaining_retry_allowance(case):
    originals = {p: p.read_bytes() for p in (case['output'] / 'responses').glob('*.json')}
    manifest = (case['output'] / 'manifest.json').read_bytes()
    calls = []
    clock = [200.]

    def sleep(delay):
        if calls:
            raise KeyboardInterrupt('SYNTHETIC stop after exactly one newly completed request')
        clock[0] += delay

    def transport(body, timeout):
        calls.append(cache_key(body))
        return response()

    with pytest.raises(KeyboardInterrupt):
        collector.collect(case['requests'], case['protocol'], case['output'], case['approval'], 'test',
                          transport=transport, sleep=sleep, clock=lambda: clock[0], monotonic=lambda: clock[0])
    assert calls == [cache_key(case['requests']['test:00002'])]
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    assert (case['output'] / 'manifest.json').read_bytes() == manifest
    ledger = read_json(case['output'] / 'reservations.json')
    assert ledger == {'manifest_sha256': digest(case['manifest']),
                      'attempt_counts': {f'test:{i:05d}': 1 for i in range(3)}}
    assert accept(case) == case['manifest']  # Legitimate ledger growth retains the old receipt.
    for rid in ('test:00003', 'test:03079'):
        _, entry = collector._entry(case['output'], rid, case['requests'][rid], case['protocol'], 'synthetic')
        assert entry['attempts'] == [] and recovery.POLICY['max_attempts_per_request'] == 4


def test_cli_approved_resume_reuses_preparation_without_test_or_model_access(case, monkeypatch, capsys):
    monkeypatch.setattr(retrieved_run, 'ROOT', case['root'])
    monkeypatch.setattr(retrieved_run, 'load_protocol', lambda: (case['protocol'], '', case['protocol']['schema']))
    observed = []

    def collect(requests, protocol, output, approval, scope):
        assert requests == case['requests'] and output == case['output'] and scope == 'test'
        assert approval == case['approval']
        observed.append(scope)
        return {'status': 'SYNTHETIC', 'halt_reason': None, 'accounting': {'api_calls': 0}}

    # _load_prepared also checks these exact compact bindings; no preflight is called.
    prepared_path = case['run'] / 'prepared.json'
    prepared = read_json(prepared_path)
    prepared['retrieval_evidence_sha256'] = digest(read_json(case['run'] / 'retrieval_evidence.json'))
    prepared['specialist_predictions_sha256'] = digest(read_json(case['run'] / 'specialist.json'))
    write(prepared_path, prepared)
    case['receipt']['prepared_files_sha256']['prepared.json'] = sha256(prepared_path.read_bytes())
    approve(case)
    monkeypatch.setattr(retrieved_run, 'collect', collect)
    retrieved_run.main(['test-live', '--approved-protocol-sha256', digest(case['protocol']),
                       '--authorize-test-access', '--authorize-live', '--spending-cap-usd', '100',
                       '--acknowledge-model-pricing-date', case['approval'].compatibility_date,
                       '--approved-resume-compatibility-sha256', case['approval'].resume_compatibility_sha256])
    assert observed == ['test'] and json.loads(capsys.readouterr().out)['status'] == 'SYNTHETIC'
