"""Synthetic sealed-boundary tests; never read the official dataset or use APIs."""
from copy import deepcopy
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import socket

import pytest

from baseline import final_protocol as final
from baseline.data import ROOT, json_bytes, sha256
from baseline.general_protocol import digest


def forbidden(*args, **kwargs):
    pytest.fail('Synthetic boundary test reached real dataset, credentials or network')


@pytest.fixture(autouse=True)
def sealed_external_access(monkeypatch):
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if any(path.is_relative_to(ROOT / name) for name in ('data/raw', 'data/processed')):
            forbidden()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    original_getitem = type(os.environ).__getitem__
    def guarded_environment(mapping, key):
        if key in ('OPENAI_API_KEY', 'OPENAI_PROJECT_ID', 'OPENAI_ORG_ID'):
            forbidden()
        return original_getitem(mapping, key)
    monkeypatch.setattr(type(os.environ), '__getitem__', guarded_environment)


@pytest.fixture
def frozen():
    return json.loads((ROOT / final.BUNDLE / 'protocol.json').read_bytes())


def csv_bytes(labels, *, count=3080, label_for=None):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['text', 'category'])
    for index in range(count):
        writer.writerow([f'SYNTHETIC independent message {index}',
                         label_for(index) if label_for else labels[index % len(labels)]])
    return stream.getvalue().encode('utf-8')


@pytest.fixture
def synthetic(tmp_path, monkeypatch, frozen):
    labels = frozen['population']['labels']
    data = csv_bytes(labels)
    source = {'revision': frozen['population']['revision'], 'files': {'test.csv': {'sha256': sha256(data)}}}
    protocol = deepcopy(frozen)
    protocol['population']['sealed_file_sha256_from_existing_metadata'] = sha256(data)
    approved = digest(protocol)
    monkeypatch.setattr(final, 'FROZEN_PROTOCOL_SHA256', approved)
    monkeypatch.setattr(final, 'verify_preparation', lambda root, protocol: source)
    authorization = final.Authorization(approved, True, True, '100',
                                        datetime.now(timezone.utc).date().isoformat())
    path = tmp_path / final.TEST_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return {'root': tmp_path, 'protocol': protocol, 'authorization': authorization,
            'path': path, 'source': source, 'data': data, 'labels': labels}


def auth_changed(authorization, **changes):
    return final.Authorization(**{**authorization.__dict__, **changes})


@pytest.mark.parametrize('changes', [
    {'approved_protocol_sha256': ''}, {'approved_protocol_sha256': 'f' * 64},
    {'authorize_test_access': False}, {'authorize_test_access': 'true'},
    {'authorize_live': False}, {'authorize_live': 'true'},
    {'spending_cap_usd': None}, {'spending_cap_usd': '0'}, {'spending_cap_usd': '-1'},
    {'spending_cap_usd': 'NaN'}, {'spending_cap_usd': 'Infinity'},
    {'spending_cap_usd': True}, {'spending_cap_usd': 'not a number'},
    {'pricing_date': ''}, {'pricing_date': '2026-01-01'}, {'pricing_date': '2099-12-31'},
])
def test_all_authorization_controls_precede_test_stat_open_and_hash(synthetic, monkeypatch, changes):
    monkeypatch.setattr(final, 'verify_preparation', forbidden)
    original_symlink = Path.is_symlink
    original_read = Path.read_bytes
    def no_test_stat(path):
        if path == synthetic['path']:
            forbidden()
        return original_symlink(path)
    def no_test_read(path):
        if path == synthetic['path']:
            forbidden()
        return original_read(path)
    monkeypatch.setattr(Path, 'is_symlink', no_test_stat)
    monkeypatch.setattr(Path, 'read_bytes', no_test_read)
    with pytest.raises(ValueError):
        final.unseal_inputs(auth_changed(synthetic['authorization'], **changes),
                            synthetic['protocol'], synthetic['root'])


def test_checksum_and_parse_share_one_read_and_return_only_all_ordered_inputs(synthetic, monkeypatch):
    read_bytes = Path.read_bytes
    reads = []
    def counted(path):
        if path == synthetic['path']:
            reads.append(path)
        return read_bytes(path)
    monkeypatch.setattr(Path, 'read_bytes', counted)
    rows = final.unseal_inputs(synthetic['authorization'], synthetic['protocol'], synthetic['root'])
    assert reads == [synthetic['path']]
    assert len(rows) == len({r['id'] for r in rows}) == 3080
    assert [r['id'] for r in rows] == [f'test:{i:05d}' for i in range(3080)]
    assert all(set(row) == {'id', 'text'} for row in rows)
    assert rows[0]['text'] == 'SYNTHETIC independent message 0'


def test_csv_checksum_drift_fails_before_rows_are_returned(synthetic):
    synthetic['path'].write_bytes(synthetic['data'] + b'\n')
    with pytest.raises(ValueError, match='checksum'):
        final.unseal_inputs(synthetic['authorization'], synthetic['protocol'], synthetic['root'])


@pytest.mark.parametrize('change', ['missing_row', 'extra_row', 'missing_label', 'unknown_label', 'extra_column', 'empty_text'])
def test_count_label_schema_and_text_fail_closed(synthetic, change):
    labels = synthetic['labels']
    if change == 'missing_row':
        data = csv_bytes(labels, count=3079)
    elif change == 'extra_row':
        data = csv_bytes(labels, count=3081)
    elif change == 'missing_label':
        data = csv_bytes(labels[:-1])
    elif change == 'unknown_label':
        data = csv_bytes(labels, label_for=lambda i: 'SYNTHETIC_UNKNOWN' if i == 0 else labels[i % 77])
    elif change == 'extra_column':
        data = synthetic['data'].replace(b'text,category', b'text,category,extra', 1)
    else:
        data = synthetic['data'].replace(b'SYNTHETIC independent message 0,', b',', 1)
    synthetic['path'].write_bytes(data)
    synthetic['source']['files']['test.csv']['sha256'] = sha256(data)
    with pytest.raises(ValueError):
        final.unseal_inputs(synthetic['authorization'], synthetic['protocol'], synthetic['root'])


@pytest.mark.parametrize('change', ['labels', 'extra_key', 'duplicate_id', 'missing_id', 'out_of_order', 'empty_text'])
def test_inference_contract_rejects_labels_or_population_changes(synthetic, change):
    rows = final.unseal_inputs(synthetic['authorization'], synthetic['protocol'], synthetic['root'])
    if change == 'labels':
        rows[0]['true_label'] = synthetic['labels'][0]
    elif change == 'extra_key':
        rows[0]['confidence'] = .99
    elif change == 'duplicate_id':
        rows[0]['id'] = rows[1]['id']
    elif change == 'missing_id':
        rows.pop()
    elif change == 'out_of_order':
        rows.reverse()
    else:
        rows[0]['text'] = ' '
    with pytest.raises(ValueError):
        final.validate_rows(rows, synthetic['protocol'])


@pytest.mark.parametrize('verified', [False, None, 1, 'true'])
def test_scoring_truth_cannot_read_until_prediction_freeze(synthetic, monkeypatch, verified):
    monkeypatch.setattr(final, '_authorized_csv', forbidden)
    with pytest.raises(ValueError, match='frozen'):
        final.scoring_truth(synthetic['authorization'], synthetic['protocol'], synthetic['root'], verified)


def test_truth_scoring_boundary_preserves_id_alignment(synthetic):
    truth = final.scoring_truth(synthetic['authorization'], synthetic['protocol'], synthetic['root'], True)
    assert len(truth) == 3080
    assert list(truth) == [f'test:{i:05d}' for i in range(3080)]
    assert truth == {f'test:{i:05d}': synthetic['labels'][i % 77] for i in range(3080)}


@pytest.mark.parametrize('change', ['threshold', 'model', 'retry', 'labels', 'checksum', 'settings'])
def test_exact_protocol_drift_rejected_without_test_access(frozen, change):
    protocol = deepcopy(frozen)
    if change == 'threshold':
        protocol['thresholds']['11']['value'] += .00001
    elif change == 'model':
        protocol['specialists']['embedding']['revision'] = 'f' * 40
    elif change == 'retry':
        protocol['fallback']['retry_policy']['max_attempts_per_request'] += 1
    elif change == 'labels':
        protocol['population']['labels'].reverse()
    elif change == 'checksum':
        protocol['population']['sealed_file_sha256_from_existing_metadata'] = '0' * 64
    else:
        protocol['fallback']['settings']['reasoning']['effort'] = 'low'
    with pytest.raises(ValueError, match='SHA-256'):
        final.validate_protocol(protocol)


def test_frozen_protocol_and_preparation_verification_need_no_test_access(frozen):
    protocol, prompt, schema = final.load_frozen()
    assert protocol == frozen
    assert prompt and schema['properties']['intent']['enum'] == frozen['population']['labels']
    source = final.verify_preparation(ROOT, protocol)
    assert source['revision'] == frozen['population']['revision']
    assert source['files']['test.csv']['sha256'] == frozen['population']['sealed_file_sha256_from_existing_metadata']


def test_altered_protocol_file_bytes_rejected_even_if_json_semantics_match(tmp_path, frozen):
    path = tmp_path / final.BUNDLE / 'protocol.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(json_bytes(frozen) + b' ')
    with pytest.raises(ValueError, match='file bytes'):
        final.load_frozen(tmp_path)


@pytest.mark.parametrize('relation', ['root_link', 'nested_link', 'different_root'])
def test_fixed_output_rejects_redirection(tmp_path, frozen, relation):
    if relation == 'different_root':
        changed = deepcopy(frozen)
        changed['outputs']['root'] = 'artifacts/another-test-run'
        with pytest.raises(ValueError, match='output changed'):
            final.safe_output(tmp_path, changed)
        return
    output = tmp_path / frozen['outputs']['root']
    target = tmp_path / 'other'
    target.mkdir()
    output.parent.mkdir(parents=True)
    if relation == 'root_link':
        output.symlink_to(target, target_is_directory=True)
    else:
        output.mkdir()
        (output / 'predictions.json').symlink_to(target / 'data.json')
    with pytest.raises(ValueError, match='Symlink'):
        final.safe_output(tmp_path, frozen)


def test_fixed_output_is_canonical_and_does_not_create_files(tmp_path, frozen):
    output = final.safe_output(tmp_path, frozen)
    assert output == tmp_path / 'artifacts/exp007-fixed-threshold-test-v1'
    assert not output.exists()
