"""Run the synthetic suite with real datasets/models/caches and networking blocked.

No official-test file is opened, stat'ed or hashed by this harness. Temporary
synthetic fixtures remain available. Reports go only to a fresh /private/tmp run.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
RUN = Path(tempfile.mkdtemp(prefix='v1-offline-suite-', dir='/private/tmp'))
os.environ['MPLCONFIGDIR'] = str(RUN / 'matplotlib')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
for key in ('OPENAI_API_KEY', 'OPENAI_PROJECT_ID', 'OPENAI_ORG_ID', 'OPENAI_BASE_URL'):
    os.environ.pop(key, None)
blocked = [str(ROOT / p) for p in ('data/raw', 'data/processed', 'artifacts', '.cache/huggingface')]
counters = {'protected_access_attempts': 0, 'network_attempts': 0}

def audit(event, args):
    if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo'):
        counters['network_attempts'] += 1
        raise RuntimeError('Offline verification forbids network access')
    if event in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes)):
        name = os.path.abspath(os.fsdecode(args[0]))
        if any(name == prefix or name.startswith(prefix + os.sep) for prefix in blocked):
            counters['protected_access_attempts'] += 1
            raise RuntimeError('Synthetic verification forbids real data/model/cache access')

sys.addaudithook(audit)
import pytest
start = time.perf_counter()
code = int(pytest.main(['-q', '--basetemp=' + str(RUN / 'fixtures')]))
report = {'command': '.venv/bin/python experiments/v1-strengthening/verify_suite.py',
          'exit_code': code, 'elapsed_seconds': time.perf_counter() - start,
          'guards': counters, 'fixture_directory': str(RUN / 'fixtures'),
          'scope': 'Synthetic test suite; real data, weights, response caches and network denied'}
(RUN / 'guards.json').write_text(json.dumps(report, indent=2) + '\n')
print('VERIFICATION_REPORT=' + str(RUN / 'guards.json'))
if any(counters.values()):
    raise SystemExit('Forbidden access was attempted; review before checkpointing')
raise SystemExit(code)
