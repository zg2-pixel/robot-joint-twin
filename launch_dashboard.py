"""Local Windows-friendly launcher: start the server, then open the browser."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent
BASE = "robot-joint-" + hashlib.sha256(str(ROOT).encode()).hexdigest()[:10]
STATE = ROOT / ".dashboard" / "server.json"


def healthy(url):
    try:
        with urlopen(url + "/_stcore/health", timeout=0.4) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true", help="start without opening a browser")
    args = parser.parse_args()
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit("Streamlit is missing. Install requirements.txt into .venv first.")
    if STATE.is_file():
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
            port = int(state["port"])
            url = f"http://127.0.0.1:{port}/{BASE}"
            if 1024 <= port <= 65535 and healthy(url):
                print("Dashboard is already running: " + url, flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
                return
        except (ValueError, KeyError, OSError):
            pass
    port = None
    for candidate in range(8501, 8521):
        try:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", candidate))
            port = candidate
            break
        except OSError:
            continue
    if port is None:
        raise SystemExit("No available local port between 8501 and 8520.")
    url = f"http://127.0.0.1:{port}/{BASE}"
    print("Starting Dashboard. Keep this window open; close it to stop the local server.", flush=True)
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
           "--server.address", "127.0.0.1", "--server.port", str(port),
           "--server.baseUrlPath", BASE, "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, cwd=ROOT)
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError("Dashboard server exited before it was ready.")
            if healthy(url):
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("Dashboard did not become ready within 30 seconds.")
        STATE.parent.mkdir(exist_ok=True)
        STATE.write_text(json.dumps({"port": port}), encoding="utf-8")
        print("Open " + url, flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
