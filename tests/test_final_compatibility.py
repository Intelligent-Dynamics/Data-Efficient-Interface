"""Synthetic receipt-only source migration checks; no real run/data/model access."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from baseline import final_compatibility as compatibility, final_protocol
from baseline.data import json_bytes, read_json, sha256
from baseline.general_protocol import digest


ORIGINAL_HEAD = 'aa58af8461ea3423b849d4e67a24eec93eb4dbb7'
RUN = Path('artifacts/exp007-fixed-threshold-test-v1')
RECEIPT = Path('experiments/exp007-cooldown-compatibility/receipt.json')
CHANGED = ['baseline/final_collection.py', 'baseline/final_compatibility.py',
           'baseline/final_protocol.py', 'baseline/final_test.py']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def approve(fixture):
    write(fixture['receipt_path'], fixture['receipt'])
    fixture['authorization'] = replace(fixture['authorization'],
        resume_compatibility_sha256=sha256(fixture['receipt_path'].read_bytes()))


@pytest.fixture
def case(tmp_path, monkeypatch):
    old = {name: '1' * 64 for name in CHANGED if 'compatibility' not in name}
    old['baseline/unchanged.py'] = '2' * 64
    new = {**old, **{name: '3' * 64 for name in CHANGED}}
    monkeypatch.setattr(final_protocol, 'runtime_code', lambda root=tmp_path: deepcopy(new))
    if hasattr(compatibility, 'runtime_code'):
        monkeypatch.setattr(compatibility, 'runtime_code', lambda root=tmp_path: deepcopy(new))
    outer = {'study_id': 'EXP-007', 'protocol_sha256': final_protocol.FROZEN_PROTOCOL_SHA256,
             'rows_sha256': '4' * 64, 'code_files_sha256': old, 'cap_usd': '28.00',
             'test_sha256': '5' * 64}
    old_code = {'git_head': ORIGINAL_HEAD, 'files_sha256': old, 'python': 'SYNTHETIC',
                'platform': 'SYNTHETIC', 'packages': {}, 'http_client': 'SYNTHETIC'}
    inner = {'study_id': 'EXP-007', 'kind': 'live',
             'protocol_sha256': final_protocol.FROZEN_PROTOCOL_SHA256,
             'requests_sha256': '4' * 64, 'aliases': [], 'pricing': {'SYNTHETIC': True},
             'cap_usd': '28.00', 'full_retry_reservation_usd': '26.00', 'code': old_code,
             'authorization': {'approved_protocol_sha256': final_protocol.FROZEN_PROTOCOL_SHA256,
                               'authorize_test_access': True, 'authorize_live': True}}
    output = tmp_path / RUN
    write(output / 'manifest.json', outer)
    write(output / 'luna/manifest.json', inner)
    counts = {f'{i:064x}': 1 for i in range(509)}
    for key in counts:
        write(output / 'luna/responses' / (key + '.json'),
              {'kind': 'live', 'cache_key': key, 'attempts': [{'attempt': 1, 'status': 'ok'}]})
    write(output / 'luna/reservations.json',
          {'manifest_sha256': digest(inner), 'attempt_counts': counts})
    receipt = {'study_id': 'EXP-007', 'protocol_sha256': final_protocol.FROZEN_PROTOCOL_SHA256,
               'run_directory': str(RUN), 'original_git_head': ORIGINAL_HEAD,
               'original_runtime_files_sha256': old, 'patched_runtime_files_sha256': new,
               'changed_source_files': CHANGED,
               'manifest_files_sha256': {name: sha256((output / name).read_bytes())
                                        for name in ('manifest.json', 'luna/manifest.json')},
               'preserved_files_sha256': {str(p.relative_to(output)): sha256(p.read_bytes())
                                          for p in output.rglob('*.json')
                                          if p.name != 'reservations.json'},
               'preserved_successful_responses': 509,
               'original_attempt_counts': counts}
    authorization = final_protocol.Authorization(final_protocol.FROZEN_PROTOCOL_SHA256,
        True, True, '28.00', datetime.now(timezone.utc).date().isoformat())
    value = {'root': tmp_path, 'output': output, 'receipt_path': tmp_path / RECEIPT,
             'receipt': receipt, 'authorization': authorization, 'old': old, 'new': new,
             'outer': outer, 'inner': inner,
             'expected_outer': {**outer, 'code_files_sha256': new},
             'expected_inner': {**inner, 'code': {**old_code, 'files_sha256': new,
                                                'git_head': '6' * 40}}}
    approve(value)
    return value


def accept(case, name='manifest.json', expected=None):
    expected = expected or case['expected_outer' if name == 'manifest.json' else 'expected_inner']
    return compatibility.accepted_manifest(case['output'] / name, expected,
                                            case['authorization'], root=case['root'])


def test_exact_manifest_needs_no_compatibility_approval_or_receipt(case):
    case['receipt_path'].unlink()
    case['authorization'] = replace(case['authorization'], resume_compatibility_sha256='')
    assert accept(case, expected=case['outer']) == case['outer']


@pytest.mark.parametrize('name', ['manifest.json', 'luna/manifest.json'])
def test_explicit_exact_transition_preserves_historical_manifests_and_509_responses(case, name):
    before = {p: p.read_bytes() for p in case['output'].rglob('*.json')}
    assert accept(case, name) == case['outer' if name == 'manifest.json' else 'inner']
    assert {p: p.read_bytes() for p in case['output'].rglob('*.json')} == before
    assert len(list((case['output'] / 'luna/responses').glob('*.json'))) == 509


@pytest.mark.parametrize('approval', ['', '0' * 64])
def test_source_transition_requires_exact_explicit_receipt_approval(case, approval):
    case['authorization'] = replace(case['authorization'], resume_compatibility_sha256=approval)
    with pytest.raises(ValueError):
        accept(case)


def test_receipt_edit_after_approval_is_rejected(case):
    case['receipt']['original_git_head'] = '0' * 40
    write(case['receipt_path'], case['receipt'])
    with pytest.raises(ValueError):
        accept(case)


@pytest.mark.parametrize('field', ['cap_usd', 'rows_sha256', 'protocol_sha256', 'test_sha256'])
def test_outer_noncode_drift_is_never_covered_by_receipt(case, field):
    expected = deepcopy(case['expected_outer'])
    expected[field] = 'changed'
    with pytest.raises(ValueError):
        accept(case, expected=expected)


@pytest.mark.parametrize('field', ['cap_usd', 'requests_sha256', 'protocol_sha256',
                                  'full_retry_reservation_usd', 'pricing', 'authorization'])
def test_inner_noncode_drift_is_never_covered_by_receipt(case, field):
    expected = deepcopy(case['expected_inner'])
    expected[field] = 'changed'
    with pytest.raises(ValueError):
        accept(case, 'luna/manifest.json', expected)


@pytest.mark.parametrize('field', ['python', 'platform', 'packages', 'http_client'])
def test_environment_changes_are_not_source_compatibility(case, field):
    expected = deepcopy(case['expected_inner'])
    expected['code'][field] = 'changed'
    with pytest.raises(ValueError):
        accept(case, 'luna/manifest.json', expected)


@pytest.mark.parametrize('name', ['manifest.json', 'luna/manifest.json',
                                 'luna/responses/' + '0' * 64 + '.json'])
def test_preserved_original_evidence_tamper_is_rejected(case, name):
    path = case['output'] / name
    content = read_json(path)
    content['tampered'] = True
    write(path, content)
    with pytest.raises(ValueError):
        accept(case)


def test_unapproved_runtime_source_change_is_rejected(case):
    case['new']['baseline/unchanged.py'] = '7' * 64
    with pytest.raises(ValueError):
        accept(case)


@pytest.mark.parametrize('field,value', [
    ('study_id', 'EXP-009'), ('protocol_sha256', '0' * 64),
    ('run_directory', 'artifacts/replacement'), ('original_git_head', '0' * 40),
    ('changed_source_files', ['baseline/unchanged.py']),
])
def test_receipt_cannot_approve_another_experiment_or_transition(case, field, value):
    case['receipt'][field] = value
    approve(case)
    with pytest.raises(ValueError):
        accept(case)


def test_receipt_preserved_paths_cannot_escape_original_run(case):
    case['receipt']['preserved_files_sha256']['../outside.json'] = '0' * 64
    approve(case)
    with pytest.raises(ValueError):
        accept(case)


@pytest.mark.parametrize('fault', ['missing_original', 'retry_original', 'manifest_digest'])
def test_original_reservations_cannot_disappear_change_or_reset(case, fault):
    path = case['output'] / 'luna/reservations.json'
    ledger = read_json(path)
    key = next(iter(ledger['attempt_counts']))
    if fault == 'missing_original':
        del ledger['attempt_counts'][key]
    elif fault == 'retry_original':
        ledger['attempt_counts'][key] = 2
    else:
        ledger['manifest_sha256'] = '0' * 64
    write(path, ledger)
    with pytest.raises(ValueError):
        accept(case)


def test_new_attempts_can_append_without_resetting_original_ledger_binding(case):
    path = case['output'] / 'luna/reservations.json'
    ledger = read_json(path)
    original_counts = deepcopy(ledger['attempt_counts'])
    ledger['attempt_counts']['f' * 64] = 1
    write(path, ledger)
    assert accept(case, 'luna/manifest.json') == case['inner']
    saved = read_json(path)
    assert saved['manifest_sha256'] == digest(case['inner'])
    assert all(saved['attempt_counts'][key] == value for key, value in original_counts.items())


def test_receipt_cannot_create_replacement_run(case):
    with pytest.raises(ValueError):
        compatibility.accepted_manifest(case['root'] / 'replacement/manifest.json',
            case['expected_outer'], case['authorization'], root=case['root'])


def test_new_resume_provenance_is_separate_and_immutable(case):
    output = case['output'] / 'luna'
    before = {p: p.read_bytes() for p in case['output'].rglob('*.json')}
    code = case['expected_inner']['code']
    compatibility.record_resume_source(output, case['authorization'], code, root=case['root'])
    path = output / 'source_compatibility.json'
    assert read_json(path) == {'receipt_sha256': case['authorization'].resume_compatibility_sha256,
                              'code': code}
    assert all(path.read_bytes() == content for path, content in before.items())
    compatibility.record_resume_source(output, case['authorization'], code,
                                       root=case['root'], write=False)
    changed = {**code, 'git_head': '9' * 40}
    with pytest.raises(ValueError):
        compatibility.record_resume_source(output, case['authorization'], changed,
                                           root=case['root'])
    assert read_json(path)['code'] == code


def test_read_only_resume_provenance_verification_never_creates_record(case):
    output = case['output'] / 'luna'
    with pytest.raises(ValueError):
        compatibility.record_resume_source(output, case['authorization'],
            case['expected_inner']['code'], root=case['root'], write=False)
    assert not (output / 'source_compatibility.json').exists()


def test_resume_provenance_cannot_be_written_to_replacement_directory(case):
    with pytest.raises(ValueError):
        compatibility.record_resume_source(case['root'] / 'replacement', case['authorization'],
            case['expected_inner']['code'], root=case['root'])
    assert not (case['root'] / 'replacement').exists()
