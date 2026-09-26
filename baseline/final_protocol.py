"""EXP-007 authorization, immutable protocol checks and guarded dataset boundary.

Import and dry-run do not access the dataset, credentials, models or network.
Only authorized preflight opens the official CSV; inference receives ID/text only.
"""
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path

from .data import ROOT, read_json, sha256
from .general import spending_cap
from .general_protocol import digest, PRICING, SETTINGS, ENDPOINT
from .recovery import POLICY
from .selective import require

FROZEN_PROTOCOL_SHA256 = '0ff095f523ebc8f05d295dad4545f6a730dd07c6c9bc7ac194bbbb4a759cfa00'
BUNDLE = Path('experiments/exp007-fixed-threshold-preparation')
PREPARATION_MANIFEST_SHA256 = 'cede66f369a58f30be2a6e969b8f679475406e879da9b667207a7085b4af403a'
TEST_RELATIVE_PATH = Path('data/raw/banking77/test.csv')


def validate_protocol(protocol):
    require(digest(protocol) == FROZEN_PROTOCOL_SHA256, 'Frozen EXP-007 protocol SHA-256 mismatch')
    require(protocol['fallback']['settings'] == SETTINGS and protocol['fallback']['endpoint'] == ENDPOINT,
            'Frozen request settings changed')
    require(protocol['fallback']['retry_policy'] == POLICY, 'Frozen retry policy changed')
    require(protocol['population']['expected_count_from_existing_metadata'] == 3080 and
            len(protocol['population']['labels']) == len(set(protocol['population']['labels'])) == 77,
            'Frozen test population changed')
    require(protocol['specialists']['seeds'] == [11, 22, 33, 44, 55], 'Frozen seeds changed')
    for seed in protocol['specialists']['seeds']:
        threshold = protocol['thresholds'][str(seed)]
        require(type(threshold['value']) is float and threshold['value'] == float(threshold['decimal']) ==
                float.fromhex(threshold['binary64_hex']), 'Frozen threshold precision changed')
    return protocol


def load_frozen(root=ROOT):
    root = Path(root)
    raw = (root / BUNDLE / 'protocol.json').read_bytes()
    require(sha256(raw) == FROZEN_PROTOCOL_SHA256, 'Frozen protocol file bytes changed')
    protocol = validate_protocol(json.loads(raw))
    prompt = (root / protocol['fallback']['prompt_path']).read_text()
    schema = read_json(root / protocol['fallback']['schema_path'])
    require(sha256(prompt.encode()) == protocol['fallback']['prompt_sha256'], 'Frozen prompt changed')
    require(digest(schema) == protocol['fallback']['schema_sha256'], 'Frozen schema changed')
    costs = read_json(root / BUNDLE / 'cost_estimate.json')
    require(digest(costs) == protocol['accounting']['cost_estimate_sha256'], 'Frozen cost assumptions changed')
    require(costs['pricing'] == PRICING, 'Frozen pricing changed')
    return protocol, prompt, schema


@dataclass(frozen=True)
class Authorization:
    approved_protocol_sha256: str = ''
    authorize_test_access: bool = False
    authorize_live: bool = False
    spending_cap_usd: str = ''
    pricing_date: str = ''

    def validate(self, protocol, *, live=True):
        validate_protocol(protocol)
        require(self.approved_protocol_sha256 == FROZEN_PROTOCOL_SHA256,
                'Explicit approval must match the frozen EXP-007 protocol SHA-256')
        require(self.authorize_test_access is True, 'Explicit --authorize-test-access required')
        require(type(live) is bool, 'Authorization scope must be explicit')
        if not live:
            return None  # Test inspection grants no API permission or spending approval.
        require(self.authorize_live is True, 'Explicit --authorize-live required')
        require(not isinstance(self.spending_cap_usd, bool), 'Positive numeric USD spending cap required')
        cap = spending_cap(self.spending_cap_usd)
        # This flag is a human acknowledgement that the frozen rates were checked
        # against the official source today; never masquerade as an automated quote.
        require(self.pricing_date == datetime.now(timezone.utc).date().isoformat(),
                'Acknowledge unchanged official prices verified on the current UTC date')
        return cap


def verify_preparation(root, protocol):
    """Check frozen code/source metadata before opening test data; never scan raw/."""
    validate_protocol(protocol)
    root = Path(root)
    raw = (root / BUNDLE / 'manifest.json').read_bytes()
    require(sha256(raw) == PREPARATION_MANIFEST_SHA256, 'Frozen preparation manifest changed')
    manifest = json.loads(raw)
    for name, expected in manifest['code_files_sha256'].items():
        require(sha256((root / name).read_bytes()) == expected, 'Frozen preparation code/package changed: ' + name)
    source_path = 'data/banking77-source.json'
    source_bytes = (root / source_path).read_bytes()
    require(sha256(source_bytes) == protocol['input_files_sha256'][source_path], 'Pinned dataset metadata changed')
    source = json.loads(source_bytes)
    require(source['revision'] == protocol['population']['revision'] and
            source['expected_counts'] == {'train': 10003, 'test': 3080, 'classes': 77} and
            source['files']['test.csv']['sha256'] == protocol['population']['sealed_file_sha256_from_existing_metadata'],
            'Dataset revision/count/checksum drift')
    return source


def validate_rows(rows, protocol):
    """This is the entire input contract exposed to inference and API collection."""
    require(isinstance(rows, list) and len(rows) == 3080, 'Exactly 3080 test rows required')
    require(all(isinstance(row, dict) and set(row) == {'id', 'text'} for row in rows),
            'Inference rows must contain only id and text, never labels')
    require([row['id'] for row in rows] == [f'test:{i:05d}' for i in range(3080)],
            'Every official test ID must occur exactly once in source order')
    require(all(isinstance(row['text'], str) and row['text'].strip() for row in rows), 'Empty request text')
    return rows


def _authorized_csv(authorization, protocol, root, *, live=True):
    # Test access always precedes stat/read/hash. Live/scoring retain all five controls.
    authorization.validate(protocol, live=live)
    source = verify_preparation(root, protocol)
    path = Path(root) / TEST_RELATIVE_PATH
    require(not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents),
            'Test path symlinks are not allowed')
    data = path.read_bytes()
    require(sha256(data) == source['files']['test.csv']['sha256'], 'Official test checksum mismatch')
    reader = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
    require(reader.fieldnames == ['text', 'category'], 'Official test CSV schema mismatch')
    records = list(reader)
    require(len(records) == 3080 and all(set(r) == {'text', 'category'} for r in records), 'Official test count/schema mismatch')
    require(set(r['category'] for r in records) == set(protocol['population']['labels']), 'Official test must contain all 77 labels')
    require(all(isinstance(r['text'], str) and r['text'].strip() for r in records), 'Empty official test text')
    return records


def unseal_inputs(authorization, protocol, root=ROOT, *, live=True):
    records = _authorized_csv(authorization, protocol, root, live=live)
    # Truth is discarded at this boundary; callers get no label side channel.
    return validate_rows([{'id': f'test:{i:05d}', 'text': r['text']} for i, r in enumerate(records)], protocol)


def scoring_truth(authorization, protocol, root, frozen_predictions_verified):
    require(frozen_predictions_verified is True, 'All prediction artifacts must be frozen before joining truth')
    return {f'test:{i:05d}': r['category'] for i, r in enumerate(_authorized_csv(authorization, protocol, root))}


def runtime_code(root=ROOT):
    root = Path(root)
    paths = sorted((root / 'baseline').glob('*.py')) + [root / 'pyproject.toml', root / 'uv.lock']
    return {str(p.relative_to(root)): sha256(p.read_bytes()) for p in paths}


def safe_output(root, protocol):
    """One canonical run directory, without an alternate-output escape hatch."""
    output = Path(root) / protocol['outputs']['root']
    require(output == Path(root) / 'artifacts/exp007-fixed-threshold-test-v1', 'Frozen output changed')
    require(not output.is_symlink() and not any(p.is_symlink() for p in output.parents), 'Symlink output forbidden')
    if output.exists():
        require(not any(p.is_symlink() for p in output.rglob('*')), 'Symlink artifacts forbidden')
    return output
