"""The local launcher must fail before starting processes on conflicting ports."""

import os
from pathlib import Path
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_launcher_rejects_an_occupied_port_without_disturbing_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        env = {**os.environ, "SIGNALDESK_PYTHON": sys.executable,
               "SIGNALDESK_API_PORT": str(port), "SIGNALDESK_UI_PORT": "18599"}
        result = subprocess.run(["bash", "scripts/serve.sh"], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=5)
        assert result.returncode != 0
        assert "already in use" in result.stdout + result.stderr
        assert listener.getsockname()[1] == port


def test_launcher_rejects_equal_api_and_dashboard_ports():
    env = {**os.environ, "SIGNALDESK_PYTHON": sys.executable,
           "SIGNALDESK_API_PORT": "18598", "SIGNALDESK_UI_PORT": "18598"}
    result = subprocess.run(["bash", "scripts/serve.sh"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert "distinct ports" in result.stdout + result.stderr
