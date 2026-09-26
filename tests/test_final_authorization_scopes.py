"""Synthetic permission-scope regressions; no real dataset, model or API access."""
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import socket
import urllib.request

import pytest

from baseline import final_protocol as protocol_module
from baseline import final_test as runner
from baseline.data import ROOT
from baseline.final_protocol import Authorization, FROZEN_PROTOCOL_SHA256, load_frozen


def forbidden(*args, **kwargs):
    pytest.fail('Authorization-scope test reached external data, credentials, models or API')


@pytest.fixture(autouse=True)
def sealed_boundaries(monkeypatch):
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if any(path.is_relative_to(ROOT / name) for name in ('data/raw', 'data/processed', '.cache', 'artifacts')):
            forbidden()
        return original_open(path, *args, **kwargs)

    original_getitem = type(os.environ).__getitem__

    def guarded_environment(mapping, key):
        if key in ('OPENAI_API_KEY', 'OPENAI_PROJECT_ID', 'OPENAI_ORG_ID', 'OPENAI_BASE_URL'):
            forbidden()
        return original_getitem(mapping, key)

    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(type(os.environ), '__getitem__', guarded_environment)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, 'open', forbidden)
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    for name in ('collect', 'run_specialists', 'scoring_truth', 'evaluate_predictions'):
        monkeypatch.setattr(runner, name, forbidden)


@pytest.fixture
def frozen():
    return load_frozen()


@pytest.fixture
def offline_authorization():
    return Authorization(approved_protocol_sha256=FROZEN_PROTOCOL_SHA256,
                         authorize_test_access=True)


@pytest.mark.parametrize('unused_live_fields', [
    {}, {'authorize_live': False}, {'spending_cap_usd': ''},
    {'pricing_date': ''}, {'pricing_date': '2000-01-01'},
])
def test_test_access_validation_does_not_require_live_fields(frozen, offline_authorization, unused_live_fields):
    authorization = replace(offline_authorization, **unused_live_fields)
    assert authorization.validate(frozen[0], live=False) is None


@pytest.mark.parametrize('changes', [
    {'approved_protocol_sha256': ''}, {'approved_protocol_sha256': 'f' * 64},
    {'authorize_test_access': False}, {'authorize_test_access': None},
    {'authorize_test_access': 1}, {'authorize_test_access': 'true'},
])
def test_offline_scope_still_requires_exact_hash_and_explicit_boolean_test_authorization(
        frozen, offline_authorization, changes):
    with pytest.raises(ValueError):
        replace(offline_authorization, **changes).validate(frozen[0], live=False)


def test_offline_unseal_scope_returns_only_ids_and_text(frozen, offline_authorization, monkeypatch):
    protocol = frozen[0]
    labels = protocol['population']['labels']
    calls = []

    def synthetic_csv(authorization, incoming_protocol, root, *, live=True):
        calls.append(live)
        authorization.validate(incoming_protocol, live=live)
        return [{'text': f'SYNTHETIC request {i}', 'category': labels[i % 77]} for i in range(3080)]

    monkeypatch.setattr(protocol_module, '_authorized_csv', synthetic_csv)
    rows = protocol_module.unseal_inputs(offline_authorization, protocol, live=False)
    assert calls == [False]
    assert len(rows) == 3080
    assert [row['id'] for row in rows] == [f'test:{i:05d}' for i in range(3080)]
    assert all(set(row) == {'id', 'text'} for row in rows)


def test_default_unseal_does_not_inherit_offline_permission(frozen, offline_authorization, monkeypatch):
    protocol = frozen[0]
    assert offline_authorization.validate(protocol, live=False) is None
    monkeypatch.setattr(protocol_module, 'verify_preparation', forbidden)
    with pytest.raises(ValueError, match='authorize-live'):
        protocol_module.unseal_inputs(offline_authorization, protocol)


@pytest.fixture
def synthetic_preflight(monkeypatch, frozen):
    protocol, prompt, schema = frozen
    events = []
    rows = [{'id': f'test:{i:05d}', 'text': f'SYNTHETIC permission-only request {i}'} for i in range(3080)]
    monkeypatch.setattr(runner, 'load_frozen', lambda *args: (protocol, prompt, schema))
    monkeypatch.setattr(runner, 'verify_preparation', lambda *args: events.append('metadata'))
    monkeypatch.setattr(runner, 'verify_specialist_files', lambda *args, **kwargs: events.append('model_hashes'))

    def unseal(authorization, incoming_protocol, root, *, live=True):
        authorization.validate(incoming_protocol, live=live)
        events.append(('synthetic_unseal', live))
        return rows

    monkeypatch.setattr(runner, 'unseal_inputs', unseal)
    return events


def test_preflight_computes_complete_reservation_without_live_authorization(
        offline_authorization, synthetic_preflight):
    result = runner.preflight(offline_authorization)
    plan = result['plan']
    assert result['api_calls'] == result['model_executions'] == 0
    assert result['test_rows'] == plan['planned_case_count'] == len(plan['aliases']) == 3080
    assert plan['planned_unique_requests'] == len(plan['requests']) == 3080
    assert plan['maximum_attempts'] == 3080 * 4
    assert result['required_cap_usd'] == plan['full_retry_reservation_usd']
    assert Decimal(result['required_cap_usd']) > 0
    assert result['provided_cap_usd'] is None and result['cap_sufficient'] is None
    assert synthetic_preflight == ['metadata', 'model_hashes', ('synthetic_unseal', False)]


@pytest.mark.parametrize('cap,sufficient', [('0.001', False), ('100', True)])
def test_offline_preflight_reports_cap_sufficiency_without_authorizing_calls(
        offline_authorization, synthetic_preflight, cap, sufficient):
    result = runner.preflight(replace(offline_authorization, spending_cap_usd=cap))
    assert Decimal(result['provided_cap_usd']) == Decimal(cap)
    assert result['cap_sufficient'] is sufficient
    assert result['api_calls'] == result['model_executions'] == 0


@pytest.mark.parametrize('cap', ['0', '-1', 'NaN', 'Infinity', 'abc', True])
def test_invalid_optional_preflight_cap_stops_before_unseal(
        offline_authorization, synthetic_preflight, cap):
    with pytest.raises(ValueError):
        runner.preflight(replace(offline_authorization, spending_cap_usd=cap))
    assert not any(isinstance(event, tuple) for event in synthetic_preflight)


@pytest.mark.parametrize('changes', [
    {'approved_protocol_sha256': ''}, {'authorize_test_access': False},
    {'authorize_live': False}, {'spending_cap_usd': ''},
    {'spending_cap_usd': '0'}, {'pricing_date': ''}, {'pricing_date': '2000-01-01'},
])
def test_successful_offline_preflight_never_grants_live_permission(
        offline_authorization, synthetic_preflight, monkeypatch, changes):
    runner.preflight(offline_authorization)
    synthetic_preflight.clear()
    monkeypatch.setattr(runner, 'unseal_inputs', forbidden)
    live = replace(offline_authorization, authorize_live=True, spending_cap_usd='100',
                   pricing_date=datetime.now(timezone.utc).date().isoformat())
    with pytest.raises(ValueError):
        runner.execute(replace(live, **changes))
    assert synthetic_preflight == []


def test_live_cap_still_fails_closed_after_offline_report(
        offline_authorization, synthetic_preflight):
    low = replace(offline_authorization, spending_cap_usd='0.001')
    assert runner.preflight(low)['cap_sufficient'] is False
    live = replace(low, authorize_live=True,
                   pricing_date=datetime.now(timezone.utc).date().isoformat())
    with pytest.raises(ValueError, match='reservation exceeds'):
        runner.execute(live)


def test_cli_preflight_accepts_only_protocol_and_test_authorization(
        offline_authorization, synthetic_preflight, capsys):
    runner.main(['preflight', '--approved-protocol-sha256', FROZEN_PROTOCOL_SHA256,
                 '--authorize-test-access'])
    result = json.loads(capsys.readouterr().out)
    assert result['authorization']['authorize_live'] is False
    assert result['required_cap_usd'] == result['plan']['full_retry_reservation_usd']
    assert result['provided_cap_usd'] is None
    assert result['api_calls'] == 0


def test_cli_preflight_missing_test_authorization_cannot_reach_loader(monkeypatch):
    monkeypatch.setattr(runner, 'verify_preparation', forbidden)
    monkeypatch.setattr(runner, 'unseal_inputs', forbidden)
    with pytest.raises(SystemExit) as stopped:
        runner.main(['preflight', '--approved-protocol-sha256', FROZEN_PROTOCOL_SHA256])
    assert stopped.value.code == 2


def test_cli_dry_run_does_not_validate_or_use_any_approval(monkeypatch, capsys):
    monkeypatch.setattr(Authorization, 'validate', forbidden)
    monkeypatch.setattr(runner, 'prepare_authorized', forbidden)
    monkeypatch.setattr(runner, 'verify_specialist_files', forbidden)
    monkeypatch.setattr(runner, 'unseal_inputs', forbidden)
    runner.main(['dry-run', '--approved-protocol-sha256', 'SYNTHETIC INVALID',
                 '--spending-cap-usd', 'invalid', '--acknowledge-current-pricing-date', 'invalid'])
    result = json.loads(capsys.readouterr().out)
    assert result['official_test_opened'] is False
    assert result['api_calls'] == result['model_executions'] == 0
    assert result['protocol_sha256'] == FROZEN_PROTOCOL_SHA256


@pytest.mark.parametrize('entrypoint', ['collect', 'verify_completed'])
def test_direct_collection_entrypoints_reject_test_only_authorization(
        frozen, offline_authorization, monkeypatch, tmp_path, entrypoint):
    from baseline import final_collection
    protocol, prompt, schema = frozen
    monkeypatch.setattr(final_collection, '_requests', forbidden)
    with pytest.raises(ValueError, match='authorize-live'):
        getattr(final_collection, entrypoint)([], protocol, prompt, schema,
                                             tmp_path / 'synthetic', offline_authorization)


def test_scoring_cannot_use_test_only_authorization_even_with_freeze_flag(
        frozen, offline_authorization, monkeypatch, tmp_path):
    monkeypatch.setattr(protocol_module, 'verify_preparation', forbidden)
    with pytest.raises(ValueError, match='authorize-live'):
        protocol_module.scoring_truth(offline_authorization, frozen[0], tmp_path,
                                      frozen_predictions_verified=True)


@pytest.mark.parametrize('ambiguous_scope', [None, 0, 1, '', 'false', 'true'])
def test_scope_must_be_boolean(frozen, offline_authorization, ambiguous_scope):
    with pytest.raises(ValueError, match='scope'):
        offline_authorization.validate(frozen[0], live=ambiguous_scope)
