#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${SIGNALDESK_PYTHON:-.venv/bin/python}"
api_port="${SIGNALDESK_API_PORT:-8000}"
ui_port="${SIGNALDESK_UI_PORT:-8501}"
if [[ ! -x "$python_bin" ]]; then
  echo "Prepare the environment with 'make install' first." >&2
  exit 1
fi
"$python_bin" - "$api_port" "$ui_port" <<'PY'
import socket
import sys
ports = [int(value) for value in sys.argv[1:]]
if len(set(ports)) != len(ports) or any(not 1 <= port <= 65535 for port in ports):
    raise SystemExit('Choose two distinct ports between 1 and 65535.')
for port in ports:
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            raise SystemExit(f'Port {port} is already in use. Stop that service or set SIGNALDESK_API_PORT and SIGNALDESK_UI_PORT.')
PY
cleanup() { kill "${api_pid:-}" "${ui_pid:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
"$python_bin" -m uvicorn signaldesk.api:app --host 127.0.0.1 --port "$api_port" &
api_pid=$!
"$python_bin" - "$api_port" <<'PY'
import sys
import time
import urllib.request
for _ in range(100):
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/readyz', timeout=1)
        break
    except OSError:
        time.sleep(0.2)
else:
    raise SystemExit(f'API did not become ready. Check artifacts and port {sys.argv[1]}.')
PY
export SIGNALDESK_API_URL="http://127.0.0.1:$api_port"
"$python_bin" -m streamlit run app/dashboard.py --server.port "$ui_port" &
ui_pid=$!
echo "SignalDesk: http://127.0.0.1:$ui_port — API docs: http://127.0.0.1:$api_port/docs"
wait -n "$api_pid" "$ui_pid"
