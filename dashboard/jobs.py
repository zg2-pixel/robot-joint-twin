"""Run the fixed analysis pipeline without blocking Streamlit or replacing saved results."""
from collections import deque
import atexit
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "dashboard_runs"
PROFILES = {
    "Detection and decision replay": ("03_detect", "05_replay"),
    "Full analysis (steps 1-5)": ("01_explore", "02_uncertainty", "03_detect", "04_forecast", "05_replay"),
}


def build_commands(profile, data, output, runs):
    if profile not in PROFILES:
        raise ValueError("Choose a supported analysis.")
    if not 1 <= runs <= 50:
        raise ValueError("Simulation runs must be between 1 and 50.")
    commands = []
    for step in PROFILES[profile]:
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / (step + ".py")),
               "--data", str(data), "--out", str(output / step)]
        if step in ("04_forecast", "05_replay"):
            cmd += ["--step3", str(output / "03_detect"), "--runs", str(runs)]
        commands.append((step, cmd))
    return commands


def completed_runs():
    """Only publish runs whose entire requested pipeline finished successfully."""
    found = {}
    for manifest in sorted(RUNS.glob("*/completed.json"), reverse=True):
        replay = manifest.parent / "05_replay" / "replay_log.csv"
        if replay.is_file():
            found[manifest.parent.name] = replay
    return found


class JobManager:
    def __init__(self, run_root=RUNS):
        self.run_root = Path(run_root)
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.process = None
        self.thread = None
        self.lines = deque(maxlen=150)
        self.state = {"status": "idle", "step": "", "done": 0, "total": 0,
                      "output": "", "error": ""}
        atexit.register(self.stop)

    def snapshot(self):
        with self.lock:
            return dict(self.state, log="".join(self.lines))

    def start(self, profile, data, runs=10):
        data = Path(data).expanduser().resolve()
        if not data.is_file():
            raise ValueError("Dataset not found. Select the downloaded 100 Hz Parquet file.")
        output = self.run_root / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        commands = build_commands(profile, data, output, int(runs))
        with self.lock:
            if self.state["status"] in ("running", "stopping") or (self.thread and self.thread.is_alive()):
                raise ValueError("An analysis is already running.")
            output.mkdir(parents=True)
            (output / "request.json").write_text(json.dumps(
                {"profile": profile, "data": str(data), "runs": int(runs),
                 "commands": commands}, indent=2), encoding="utf-8")
            self.lines.clear()
            self.cancel.clear()
            self.state = {"status": "running", "step": "Starting", "done": 0,
                          "total": len(commands), "output": str(output), "error": ""}
            self.thread = threading.Thread(target=self._run, args=(commands, output), daemon=True)
            self.thread.start()
        return output

    def stop(self):
        with self.lock:
            if self.state["status"] != "running":
                return
            self.cancel.set()
            self.state["status"] = "stopping"
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()

    def _run(self, commands, output):
        try:
            with (output / "run.log").open("w", encoding="utf-8") as logfile:
                for step, cmd in commands:
                    with self.lock:
                        if self.cancel.is_set():
                            break
                        self.state["step"] = step
                        self.lines.append("\nStarting " + step + "\n")
                        self.process = subprocess.Popen(
                            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
                        proc = self.process
                    for line in proc.stdout:
                        logfile.write(line)
                        logfile.flush()
                        with self.lock:
                            self.lines.append(line)
                    code = proc.wait()
                    proc.stdout.close()
                    with self.lock:
                        self.process = None
                        if self.cancel.is_set():
                            break
                        if code:
                            raise RuntimeError(f"{step} failed (exit {code}). See the run log.")
                        self.state["done"] += 1
                with self.lock:
                    if self.cancel.is_set():
                        self.state["status"] = "cancelled"
                    else:
                        replay = output / "05_replay" / "replay_log.csv"
                        if not replay.is_file():
                            raise RuntimeError("Analysis finished without producing a replay log.")
                        (output / "completed.json").write_text(
                            json.dumps({"completed": datetime.now().isoformat()}), encoding="utf-8")
                        self.state["status"] = "completed"
        except Exception as exc:
            with self.lock:
                self.state["status"] = "failed"
                self.state["error"] = str(exc)
                self.lines.append("\n" + str(exc) + "\n")
        finally:
            with self.lock:
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait(timeout=10)
                self.process = None
