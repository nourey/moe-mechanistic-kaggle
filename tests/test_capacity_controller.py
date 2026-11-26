"""CPU-only controller/lifecycle checks; no native runtime or GPU is launched."""
import contextlib
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

import capacity_run
from capacity_contract import freeze


class ControllerLifecycleTests(unittest.TestCase):
    def test_terminated_leader_cannot_leave_its_owned_process_group_running(self):
        """A launcher must clean its session even after its direct child exits."""
        with tempfile.TemporaryDirectory() as directory:
            child_pid = Path(directory) / "child.pid"
            leader = (
                "import pathlib, subprocess, sys; "
                "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                "pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
            )
            proc = subprocess.Popen([sys.executable, "-c", leader, str(child_pid)],
                                    start_new_session=True)
            self.assertEqual(proc.wait(timeout=5), 0)
            for _ in range(50):
                if child_pid.exists():
                    break
                time.sleep(.01)
            self.assertTrue(child_pid.exists())
            self.assertTrue(capacity_run.owned_group_exists(proc.pid))
            self.assertTrue(capacity_run.terminate_owned_group(proc.pid))
            deadline = time.monotonic() + 2
            while capacity_run.owned_group_exists(proc.pid) and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertFalse(capacity_run.owned_group_exists(proc.pid))

    def test_launch_specs_normalize_child_paths_without_resolving_model_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / "model-target"; target.mkdir()
            model_link = base / "model-link"; model_link.symlink_to(target, target_is_directory=True)
            old_cwd = Path.cwd(); os.chdir(base)
            Path("relative-output").mkdir()
            expected_out = os.path.abspath("relative-output/D1")
            expected_control = os.path.abspath("relative-output/control")
            expected_model = os.path.abspath("model-link")
            calls = []
            proc = Mock(pid=999); proc.poll.return_value = 0
            def spawn(argv, **kwargs):
                calls.append((argv, kwargs)); return proc
            try:
                with patch("capacity_run.reserve_check"), \
                     patch("capacity_run.wait_files", side_effect=RuntimeError("stop after launch")), \
                     patch("capacity_run.terminate_owned_group", return_value=False):
                    with self.assertRaisesRegex(RuntimeError, "stop after launch"):
                        capacity_run.launch_trial("relative-output", "B", freeze(smoke=True), "diagnostic",
                                                  sys.executable, "model-link", [
                                                      {"uuid": "GPU-0"}, {"uuid": "GPU-1"}],
                                                  {"ttft_seconds": None, "mean_tpot_seconds": None,
                                                   "end_to_end_seconds": None}, spawn=spawn)
            finally:
                os.chdir(old_cwd)
            spec = capacity_run.read(base / "relative-output" / "D1" / "spec.json")
            self.assertEqual(spec["output"], expected_out)
            self.assertEqual(spec["control"], expected_control)
            self.assertEqual(spec["model"], expected_model)
            self.assertTrue(Path(spec["model"]).is_symlink())
            self.assertTrue(calls[0][1]["start_new_session"])

    def test_relative_path_entry_becomes_absolute_but_preserves_venv_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); bin_dir = base / "bin"; bin_dir.mkdir()
            runtime_link = bin_dir / "runtime-python"; runtime_link.symlink_to(sys.executable)
            old_cwd = Path.cwd(); os.chdir(base)
            expected_runtime = os.path.abspath("bin/runtime-python")
            args = types.SimpleNamespace(python="runtime-python", model="model", output="out",
                                         workload="work", install_report="install", slo=None,
                                         arms="NOT_AN_ARM", repetitions=1, mode="diagnostic",
                                         devices="0", max_wall_seconds=1)
            config = {"slo": {}, "topology": {"B": []}}
            try:
                with patch.dict(os.environ, {"PATH": "bin"}, clear=False), \
                     patch("capacity_run.read", side_effect=[config, {"smoke": True}]), \
                     patch("capacity_run.validate_workload"), patch("capacity_run.validate_slo"), \
                     patch("capacity_run.preflight") as preflight:
                    with self.assertRaisesRegex(ValueError, "supported arms"):
                        capacity_run.execute(args)
            finally:
                os.chdir(old_cwd)
            self.assertEqual(args.python, expected_runtime)
            self.assertTrue(Path(args.python).is_symlink())
            preflight.assert_not_called()

    def test_source_drift_between_trials_blocks_before_launching_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = types.SimpleNamespace(python=sys.executable, model=str(base / "model"),
                                         output=str(base / "out"), workload=str(base / "work"),
                                         install_report=str(base / "install"), slo=None, arms="B",
                                         repetitions=1, mode="diagnostic", devices="0,1",
                                         max_wall_seconds=99_999)
            config = {"slo": {}, "topology": {"B": [{"group": "D1", "dp": 2, "devices": [0, 1]}]},
                      "startup_timeout_seconds": 1, "drain_timeout_seconds": 1,
                      "cleanup_timeout_seconds": 1, "scope": "synthetic"}
            preflight = Mock(return_value=[{"uuid": "GPU-0"}, {"uuid": "GPU-1"}])
            with patch("capacity_run.read", side_effect=[config, {"smoke": True}]), \
                 patch("capacity_run.validate_workload"), patch("capacity_run.validate_slo"), \
                 patch("capacity_run.source_hashes", side_effect=[{"a": "old"}, {"a": "changed"}]) as hashes, \
                 patch("capacity_run.SOURCE", []), patch("capacity_run.atomic"), \
                 patch("capacity_run.preflight", preflight), patch("capacity_run.runtime.seal", return_value={}), \
                 patch("capacity_run.subprocess.check_output", side_effect=["", "commit"]), \
                 patch("capacity_run.launch_trial") as launch, \
                 contextlib.redirect_stderr(io.StringIO()):
                result = capacity_run.execute(args)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(hashes.call_count, 2)
            preflight.assert_called_once()
            launch.assert_not_called()

    def test_readonly_duplicate_port_validation_has_no_side_effects(self):
        with self.assertRaisesRegex(ValueError, "ports must differ"):
            capacity_run.reserve_check([30101, 30101])


if __name__ == "__main__":
    unittest.main()
