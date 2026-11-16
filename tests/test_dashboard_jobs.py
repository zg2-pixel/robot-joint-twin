import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from dashboard.jobs import JobManager, build_commands, completed_runs
import dashboard.jobs as jobs


class JobsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "input.parquet"
        self.data.touch()
        self.manager = JobManager(self.root / "runs")

    def tearDown(self):
        self.manager.stop()
        if self.manager.thread:
            self.manager.thread.join(5)
        self.tmp.cleanup()

    def wait(self):
        self.manager.thread.join(10)
        self.assertFalse(self.manager.thread.is_alive())
        return self.manager.snapshot()

    def test_pipeline_dependencies_are_isolated(self):
        commands = build_commands("Full analysis (steps 1-5)", self.data, self.root, 2)
        self.assertEqual(len(commands), 5)
        for step, command in commands[-2:]:
            self.assertEqual(command[command.index("--step3") + 1], str(self.root / "03_detect"))
            self.assertEqual(command[command.index("--out") + 1], str(self.root / step))

    def test_missing_data_starts_nothing(self):
        with self.assertRaises(ValueError):
            self.manager.start("Detection and decision replay", self.root / "missing")
        self.assertEqual(self.manager.snapshot()["status"], "idle")
        self.assertFalse((self.root / "runs").exists())

    def test_success_publishes_only_completed_result(self):
        def fake_commands(profile, data, out, runs):
            code = ("from pathlib import Path; p=Path(" + repr(str(out / "05_replay")) +
                    "); p.mkdir(parents=True); (p/'replay_log.csv').write_text('scenario,cycle\\nS1,0\\n'); print('done')")
            return [("test", [sys.executable, "-u", "-c", code])]
        with patch.object(jobs, "build_commands", fake_commands):
            output = self.manager.start("Detection and decision replay", self.data)
        state = self.wait()
        self.assertEqual(state["status"], "completed", state)
        self.assertIn("done", state["log"])
        self.assertTrue((output / "completed.json").is_file())
        with patch.object(jobs, "RUNS", self.root / "runs"):
            self.assertEqual(list(completed_runs().values()), [output / "05_replay" / "replay_log.csv"])

    def test_failure_does_not_publish_or_continue(self):
        commands = [("bad", [sys.executable, "-u", "-c", "print('failure'); raise SystemExit(3)"]),
                    ("must not run", [sys.executable, "-c", "raise SystemExit(99)"])]
        with patch.object(jobs, "build_commands", return_value=commands):
            output = self.manager.start("Detection and decision replay", self.data)
        state = self.wait()
        self.assertEqual(state["status"], "failed")
        self.assertIn("exit 3", state["error"])
        self.assertEqual(state["done"], 0)
        self.assertFalse((output / "completed.json").exists())

    def test_cancel_and_duplicate_start(self):
        commands = [("slow", [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(30)"])]
        with patch.object(jobs, "build_commands", return_value=commands):
            output = self.manager.start("Detection and decision replay", self.data)
            with self.assertRaises(ValueError):
                self.manager.start("Detection and decision replay", self.data)
        deadline = time.monotonic() + 5
        while "started" not in self.manager.snapshot()["log"] and time.monotonic() < deadline:
            time.sleep(.02)
        self.manager.stop()
        self.assertEqual(self.wait()["status"], "cancelled")
        self.assertFalse((output / "completed.json").exists())


if __name__ == "__main__":
    unittest.main()
