#!/usr/bin/env python3
"""Upload runner + config to Molab, then fire the pipeline."""
import sys, base64, requests

URL = sys.argv[1]
TOKEN = sys.argv[2]
RUNNER_PATH = sys.argv[3]
CONFIG_PATH = sys.argv[4] if len(sys.argv) > 4 else None

# Get session
sessions = requests.get(f'{URL}/api/sessions', headers={
    'Authorization': f'Bearer {TOKEN}',
    'Content-Type': 'application/json',
}).json()
session_id = list(sessions.keys())[0]
print(f'Session: {session_id}')

def exec_code(code):
    r = requests.post(f'{URL}/api/kernel/execute', headers={
        'Authorization': f'Bearer {TOKEN}',
        'Content-Type': 'application/json',
        'Marimo-Session-Id': session_id,
    }, json={'code': code}, stream=True, timeout=300)
    text = ''
    for line in r.iter_lines():
        if line:
            decoded = line.decode()
            if decoded.startswith('data: '):
                text += decoded[6:]
    return text

# Upload runner
with open(RUNNER_PATH, 'rb') as f:
    runner_b64 = base64.b64encode(f.read()).decode()

code = f"import base64; open('/tmp/absolver-direct-runner.py','wb').write(base64.b64decode('''{runner_b64}'''))"
r = exec_code(code)
print(f'Runner upload: {r.strip()[:60] if r.strip() else "ok"}')

# Upload config
if CONFIG_PATH:
    with open(CONFIG_PATH) as f:
        cfg = f.read()
    escaped = cfg.replace("'", "'\\''")
    code = f"open('/tmp/absolver-config.yaml','w').write('{escaped}')"
    r = exec_code(code)
    print(f'Config upload: {r.strip()[:60] if r.strip() else "ok"}')

# Fire the runner
print('Firing pipeline...')
r = exec_code("exec(open('/tmp/absolver-direct-runner.py').read())")
print(r[-3000:])
