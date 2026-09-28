"""Exact, explicitly approved source transition for the interrupted EXP-009 test run."""
from copy import deepcopy
from pathlib import Path

from .data import ROOT, read_json, sha256
from .final_protocol import runtime_code
from .general import atomic_json, sync_reservation_ledger
from .general_protocol import digest
from .selective import require

RECEIPT = Path('experiments/exp009-resume-compatibility/receipt.json')
RUN = Path('artifacts/exp009-retrieved-luna-v1/test')
PROTOCOL_SHA256 = 'b78f3d32a9bd86ea130ce1f51e2feded9d52b5f5ff796507c49d6f53fccc3334'
CHANGED = {'baseline/retrieved_collection.py', 'baseline/retrieved_run.py',
           'baseline/retrieved_compatibility.py'}
PREPARED = {'prepared.json', 'requests.json', 'retrieval_evidence.json', 'specialist.json', 'overhead.json'}


def _bound_files(source):
    files = dict(source['reused_code']['files_sha256'])
    for field in ('code_sha256', 'reused_helper_sha256'):
        for name, value in source[field].items():
            key = 'baseline/' + name
            require(key not in files or files[key] == value, 'Conflicting source binding')
            files[key] = value
    return files


def validate_receipt(root, approval):
    """Read-only checks; no dataset/model loading, ledger repair or new reservations."""
    root = Path(root)
    approved = getattr(approval, 'resume_compatibility_sha256', '')
    require(isinstance(approved, str) and len(approved) == 64,
            'Resume source drift requires explicit --approved-resume-compatibility-sha256')
    require(approval.protocol_sha256 == PROTOCOL_SHA256 and approval.authorize_test_access is True,
            'EXP-009 protocol and test-access approval required for compatibility')
    raw = (root / RECEIPT).read_bytes()
    require(sha256(raw) == approved, 'Resume compatibility receipt hash mismatch')
    receipt = read_json(root / RECEIPT)
    require(receipt['study_id'] == 'EXP-009' and receipt['protocol_sha256'] == PROTOCOL_SHA256 and
            receipt['run_directory'] == str(RUN), 'Resume compatibility scope mismatch')
    old, new = receipt['original_runtime_files_sha256'], receipt['patched_runtime_files_sha256']
    require(runtime_code(root) == new, 'Patched runtime differs from approved compatibility receipt')
    changed = {p for p in set(old) | set(new) if old.get(p) != new.get(p)}
    require(changed == set(receipt['changed_source_files']) == CHANGED,
            'Compatibility exceeds the exact cooldown/receipt source transition')
    run = root / RUN
    require(run.is_dir() and not any(p.is_symlink() for p in [run, *run.parents]) and
            not any(p.is_symlink() for p in run.rglob('*')), 'Existing original run without symlinks required')
    mp = run / 'collection/manifest.json'
    require(sha256(mp.read_bytes()) == receipt['original_manifest_sha256'], 'Historical manifest changed')
    saved = read_json(mp)
    require(saved['study_id'] == 'EXP-009' and saved['scope'] == 'test' and
            saved['protocol_sha256'] == PROTOCOL_SHA256 and
            saved['requests_sha256'] == receipt['requests_sha256'] and
            saved['cap_usd'] == receipt['cap_usd'] and
            saved['full_retry_reservation_usd'] == receipt['full_retry_reservation_usd'] and
            saved['reused_code']['git_head'] == receipt['original_git_head'], 'Original manifest identity drift')
    require(all(old.get(p) == h for p, h in _bound_files(saved).items()), 'Old source binding mismatch')
    require(set(receipt['prepared_files_sha256']) == PREPARED, 'Complete prepared evidence binding required')
    for name, expected in receipt['prepared_files_sha256'].items():
        require(sha256((run / name).read_bytes()) == expected, 'Prepared evidence changed: ' + name)
    prepared, requests = read_json(run / 'prepared.json'), read_json(run / 'requests.json')
    ids = [f'test:{i:05d}' for i in range(3080)]
    require(prepared['protocol_sha256'] == PROTOCOL_SHA256 and prepared['scope'] == 'test' and
            prepared['ids'] == ids and set(requests) == set(ids) and
            digest(requests) == prepared['requests_sha256'] == receipt['requests_sha256'] and
            prepared['reservation_usd'] == receipt['full_retry_reservation_usd'], 'Prepared request identity drift')
    original_ledger = receipt['original_ledger']
    require(digest(original_ledger) == receipt['original_ledger_sha256'] and
            original_ledger['manifest_sha256'] == digest(saved), 'Original ledger receipt mismatch')
    original_counts = original_ledger['attempt_counts']
    require(original_counts and set(original_counts) <= set(ids) and
            all(type(v) is int and v == 1 for v in original_counts.values()), 'Invalid original successful counts')
    expected_paths = {'collection/responses/' + rid.replace(':', '_') + '.json' for rid in original_counts}
    require(set(receipt['preserved_response_files_sha256']) == expected_paths, 'Missing original response binding')
    for name, expected in receipt['preserved_response_files_sha256'].items():
        require(sha256((run / name).read_bytes()) == expected, 'Preserved successful response changed: ' + name)
    ledger_path = run / 'collection/reservations.json'
    ledger = read_json(ledger_path)
    require(set(ledger) == {'manifest_sha256', 'attempt_counts'} and ledger['manifest_sha256'] == digest(saved),
            'Historical ledger provenance changed')
    require(set(ledger['attempt_counts']) <= set(ids) and
            all(type(v) is int and 1 <= v <= 4 for v in ledger['attempt_counts'].values()) and
            all(ledger['attempt_counts'].get(rid) == count for rid, count in original_counts.items()),
            'Original reservation removed or successful request retried')
    if ledger == original_ledger:
        require(sha256(ledger_path.read_bytes()) == receipt['original_ledger_sha256'], 'Original ledger bytes changed')
    from .retrieved_collection import _entry, validate_requests
    protocol_path = root / 'experiments/exp009-retrieved-luna/protocol.json'
    require(sha256(protocol_path.read_bytes()) == PROTOCOL_SHA256, 'Frozen protocol bytes changed')
    protocol = read_json(protocol_path)
    validate_requests(requests, protocol)
    entries = {rid: _entry(run / 'collection', rid, requests[rid], protocol, saved['kind'])[1] for rid in ids}
    require({p.name for p in (run / 'collection/responses').glob('*.json')} ==
            {rid.replace(':', '_') + '.json' for rid, e in entries.items() if e['attempts']},
            'Unrecognized response file')
    require(all(len(entries[rid]['attempts']) == 1 and entries[rid]['attempts'][0]['status'] == 'ok' and
                entries[rid]['attempts'][0].get('retryable') is False and not entries[rid]['attempts'][0].get('halt')
                for rid in original_counts), 'Original success no longer terminal')
    sync_reservation_ledger(run / 'collection', saved, entries, write=False)
    return receipt


def accepted_manifest(path, expected, approval, *, root=ROOT):
    """Normalize only the exact approved source transition; return the old manifest."""
    path, root = Path(path), Path(root)
    approved = getattr(approval, 'resume_compatibility_sha256', '')
    if not approved:
        if path.exists():
            require(read_json(path) == expected, 'Resume manifest/code/input/cap drift')
        return expected
    require(path == root / RUN / 'collection/manifest.json' and path.is_file(),
            'Compatibility cannot create a new/replacement run')
    receipt = validate_receipt(root, approval)
    saved = read_json(path)
    new = receipt['patched_runtime_files_sha256']
    require(expected['code_sha256'] == {Path(p).name: h for p, h in new.items()
                                       if p.startswith('baseline/retrieved') and p.endswith('.py')} and
            expected['reused_helper_sha256'] == {name: new['baseline/' + name] for name in
                                                set(saved['reused_helper_sha256']) | {'final_collection.py'}} and
            all(new.get(p) == h for p, h in _bound_files(expected).items()),
            'Expected patched source binding mismatch')
    adjusted = deepcopy(expected)
    adjusted['code_sha256'] = saved['code_sha256']
    adjusted['reused_helper_sha256'] = saved['reused_helper_sha256']
    adjusted['reused_code']['git_head'] = saved['reused_code']['git_head']
    require(adjusted == saved, 'Compatibility cannot change requests, cap, scope or environment')
    return saved


def record_resume_source(output, approval, source, *, root=ROOT, write=True):
    """A new immutable provenance record, written only by a later authorized resume."""
    approved = getattr(approval, 'resume_compatibility_sha256', '')
    if not approved:
        require(not (Path(output) / 'source_compatibility.json').exists(), 'Receipt approval required for patched provenance')
        return
    require(Path(output) == Path(root) / RUN / 'collection', 'Compatibility provenance directory changed')
    validate_receipt(root, approval)
    from .retrieved_collection import source_record
    require(source == source_record(), 'Resume provenance must describe current runtime')
    record = {'receipt_sha256': approved, 'source': source}
    path = Path(output) / 'source_compatibility.json'
    if path.exists():
        require(read_json(path) == record, 'Patched resume source provenance changed')
    else:
        require(write, 'Missing patched resume source provenance')
        atomic_json(path, record)
