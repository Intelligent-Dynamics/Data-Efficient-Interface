"""Explicit, exact-source receipt for the single interrupted EXP-007 cooldown fix.

Historical manifests remain immutable. No general migration or drift exemption.
"""
from copy import deepcopy
from pathlib import Path

from .data import ROOT, read_json, sha256
from .final_protocol import FROZEN_PROTOCOL_SHA256, runtime_code
from .general import atomic_json
from .selective import require

RECEIPT = Path('experiments/exp007-cooldown-compatibility/receipt.json')
RUN = Path('artifacts/exp007-fixed-threshold-test-v1')
CHANGED = {'baseline/final_collection.py', 'baseline/final_protocol.py',
           'baseline/final_test.py', 'baseline/final_compatibility.py'}


def validate_receipt(root, authorization):
    root = Path(root)
    approved = getattr(authorization, 'resume_compatibility_sha256', '')
    require(isinstance(approved, str) and len(approved) == 64,
            'Resume manifest/code drift requires explicit --approved-resume-compatibility-sha256')
    raw = (root / RECEIPT).read_bytes()
    require(sha256(raw) == approved, 'Resume compatibility receipt hash mismatch')
    receipt = read_json(root / RECEIPT)
    require(receipt['study_id'] == 'EXP-007' and
            receipt['protocol_sha256'] == FROZEN_PROTOCOL_SHA256 and
            receipt['run_directory'] == str(RUN), 'Resume compatibility scope mismatch')
    old, new = receipt['original_runtime_files_sha256'], receipt['patched_runtime_files_sha256']
    require(runtime_code(root) == new, 'Patched runtime differs from approved compatibility receipt')
    changed = {p for p in set(old) | set(new) if old.get(p) != new.get(p)}
    require(changed == set(receipt['changed_source_files']) == CHANGED,
            'Compatibility receipt exceeds the cooldown-only source transition')
    output = root / RUN
    require(set(receipt['manifest_files_sha256']) == {'manifest.json', 'luna/manifest.json'},
            'Both historical manifests must be bound')
    for name, expected in receipt['manifest_files_sha256'].items():
        require(sha256((output / name).read_bytes()) == expected, 'Historical manifest changed: ' + name)
    outer, inner = read_json(output / 'manifest.json'), read_json(output / 'luna/manifest.json')
    require(outer['code_files_sha256'] == inner['code']['files_sha256'] == old and
            inner['code']['git_head'] == receipt['original_git_head'] and
            outer['protocol_sha256'] == inner['protocol_sha256'] == FROZEN_PROTOCOL_SHA256,
            'Original source/protocol binding mismatch')
    for name, expected in receipt['preserved_files_sha256'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid preservation path')
        require(sha256((output / relative).read_bytes()) == expected,
                'Preserved EXP-007 evidence changed: ' + name)
    # The journal may grow on later authorized resumes; its old reservations cannot disappear.
    ledger = read_json(output / 'luna/reservations.json')
    from .general_protocol import digest
    require(ledger['manifest_sha256'] == digest(inner), 'Historical ledger binding changed')
    require(all(ledger['attempt_counts'].get(key, 0) == count == 1
                for key, count in receipt['original_attempt_counts'].items()),
            'An original successful request was removed or retried')
    require(all('luna/responses/' + key + '.json' in receipt['preserved_files_sha256']
                for key in receipt['original_attempt_counts']), 'Missing original response binding')
    return receipt


def accepted_manifest(path, expected, authorization, *, root=ROOT):
    """Return the original manifest only after proving this exact approved transition."""
    path, root = Path(path), Path(root)
    if not path.exists():
        require(not getattr(authorization, 'resume_compatibility_sha256', ''),
                'Compatibility approval cannot create a replacement run')
        return expected
    saved = read_json(path)
    if saved == expected and not getattr(authorization, 'resume_compatibility_sha256', ''):
        return saved
    receipt = validate_receipt(root, authorization)
    relative = str(path.relative_to(root / RUN))
    adjusted = deepcopy(expected)
    if relative == 'manifest.json':
        require(adjusted['code_files_sha256'] == receipt['patched_runtime_files_sha256'],
                'Expected outer runtime mismatch')
        adjusted['code_files_sha256'] = receipt['original_runtime_files_sha256']
    elif relative == 'luna/manifest.json':
        require(adjusted['code']['files_sha256'] == receipt['patched_runtime_files_sha256'],
                'Expected collection runtime mismatch')
        adjusted['code']['files_sha256'] = receipt['original_runtime_files_sha256']
        adjusted['code']['git_head'] = receipt['original_git_head']
    else:
        raise ValueError('Compatibility cannot modify another artifact binding')
    require(saved == adjusted, 'Resume compatibility cannot change configuration, cap, rows or environment')
    return saved


def record_resume_source(output, authorization, code, *, root=ROOT, write=True):
    """New provenance, written only by a later authorized resume; never edit old records."""
    approved = getattr(authorization, 'resume_compatibility_sha256', '')
    if not approved:
        return
    validate_receipt(root, authorization)
    require(Path(output) == Path(root) / RUN / 'luna', 'Compatibility provenance directory changed')
    record = {'receipt_sha256': approved, 'code': code}
    path = Path(output) / 'source_compatibility.json'
    if path.exists():
        require(read_json(path) == record, 'Patched resume source provenance changed')
    else:
        require(write, 'Missing patched resume source provenance')
        atomic_json(path, record)
