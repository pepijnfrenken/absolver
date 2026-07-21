"""Execute Python code on a Molab Marimo kernel."""
import json, sys, requests

URL = sys.argv[1].rstrip("/")
TOKEN = sys.argv[2]
CODE = sys.argv[3] if len(sys.argv) > 3 else sys.stdin.read()

try:
    # Get session ID
    resp = requests.get(f"{URL}/api/sessions", headers={
        "Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"
    }, timeout=15)
    if not resp.ok:
        print(f"FAILED: GET /api/sessions returned {resp.status_code}")
        print(resp.text[:500])
        sys.exit(1)
    sess = resp.json()
    sid = list(sess.keys())[0]
    print(f"Session: {sid}", file=sys.stderr)

    # Execute
    hdrs = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
            "Marimo-Session-Id": sid}
    r = requests.post(f"{URL}/api/kernel/execute", headers=hdrs,
                      json={"code": CODE}, timeout=600, stream=True)

    # Parse SSE output
    output = []
    for line in r.text.split("\n"):
        if line.startswith("data: ") and '"data":' in line:
            try:
                d = json.loads(line[6:])
                if d.get("data"):
                    output.append(str(d["data"]))
            except json.JSONDecodeError:
                pass

    success = '"success": true' in r.text
    print("SUCCESS" if success else "FAILED")
    if output:
        print("\n".join(output))
    elif not success:
        print("No output and response was:", r.text[:500])
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
