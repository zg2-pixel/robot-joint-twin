from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from dashboard import jobs, run_panel, twin_view

APP = str(jobs.ROOT / "dashboard" / "app.py")


class DashboardTest(unittest.TestCase):
    def test_no_results_still_exposes_analysis_page(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(twin_view, "LOG_PATH", root / "missing.csv"), patch.object(jobs, "RUNS", root):
                app = AppTest.from_file(APP, default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertTrue(any("No replay results" in item.value for item in app.info))
                app.sidebar.radio(key="workspace_page").set_value("Run analysis").run()
                self.assertFalse(app.exception)
                self.assertTrue(next(b for b in app.button if b.label == "Run analysis").disabled)

    def test_completed_background_run_can_be_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "fixture.parquet"
            data.touch()
            manager = jobs.JobManager(root / "runs")
            def commands(profile, dataset, output, count):
                code = ("from pathlib import Path; import shutil; p=Path(" +
                        repr(str(output / "05_replay")) +
                        "); p.mkdir(parents=True); shutil.copyfile(" +
                        repr(str(twin_view.LOG_PATH)) + ",p/'replay_log.csv')")
                return [("fixture", [sys.executable, "-u", "-c", code])]
            with patch.object(jobs, "RUNS", root / "runs"), patch.object(run_panel, "get_manager", return_value=manager), patch.object(jobs, "build_commands", commands):
                app = AppTest.from_file(APP, default_timeout=30).run()
                self.assertFalse(app.exception)
                app.sidebar.radio(key="workspace_page").set_value("Run analysis").run()
                app.text_input(key="dataset_path").set_value(str(data)).run()
                next(b for b in app.button if b.label == "Run analysis").click().run()
                manager.thread.join(10)
                self.assertEqual(manager.snapshot()["status"], "completed", manager.snapshot())
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(app.success)
                app.sidebar.radio(key="workspace_page").set_value("Dashboard").run()
                selected = next(s for s in app.selectbox if s.label == "Results")
                run_name = Path(manager.snapshot()["output"]).name
                self.assertIn(run_name, selected.options)
                selected.set_value(run_name).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.metric), 5)


if __name__ == "__main__":
    unittest.main()
