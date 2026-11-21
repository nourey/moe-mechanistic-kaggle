import contextlib
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import diagnose


class FakeCommands:
    """Explicit synthetic host responses; never actual GPU evidence."""
    def __init__(self, missing_gpu=False, broken_vllm=False, timeout_torch=False, secret=""):
        self.calls = []
        self.missing_gpu, self.broken_vllm = missing_gpu, broken_vllm
        self.timeout_torch, self.secret = timeout_torch, secret

    def __call__(self, label, argv, timeout):
        self.calls.append((label, argv, timeout))
        code, out, err, value = 0, "", "", None
        if label == "nvidia-smi":
            code, out, err = (127, "", "nvidia-smi missing") if self.missing_gpu else (0, "SYNTHETIC CUDA Version: 12.9", "")
        elif label.startswith("gpu-query"):
            if self.missing_gpu: code, err = 127, "driver unavailable"
            else: out = "0, Tesla T4, GPU-synthetic-0, 15360, 14900, 580.178.04, 7.5\n1, Tesla T4, GPU-synthetic-1, 15360, 14900, 580.178.04, 7.5\n"
        elif label == "torch":
            if self.timeout_torch: code, err = 124, "synthetic import timeout"
            else: value = {"ok": True, "version": "2.13.0+cu129", "cuda_available": not self.missing_gpu,
                           "gpu_count": 0 if self.missing_gpu else 2, "nccl_available": True,
                           "devices": [{"name": "Tesla T4", "compute_capability": [7, 5]}] * (0 if self.missing_gpu else 2),
                           "peer_access": [{"from": 0, "to": 1, "driver_reports_access": False}]}
        elif label in ("vllm", "triton"):
            value = {"ok": not (label == "vllm" and self.broken_vllm), "version": "0.31.0" if label == "vllm" else "3.7.1"}
            if not value["ok"]: code, err = 1, "synthetic vLLM ImportError: dependency not available"
            if self.secret: err += " Bearer " + self.secret
        elif label == "package-versions": value = {"ok": True, "versions": {"vllm": "0.31.0"}}
        elif label == "git-head": out = "f" * 40 + "\n"
        elif label == "git-origin": out = "https://github.com/nourey/moe-mechanistic-kaggle.git\n"
        elif label == "nvcc-version": code, err = 127, "optional nvcc absent"
        if value is not None: out = diagnose.MARKER + json.dumps(value) + "\n"
        return {"label": label, "argv": argv, "exit_code": code, "timed_out": code == 124,
                "stdout": out, "stderr": err}


class DiagnosticTests(unittest.TestCase):
    def collect(self, root, fake):
        with contextlib.redirect_stdout(io.StringIO()):
            return diagnose.collect(root, execute=fake)

    def config(self):
        return json.loads((diagnose.ROOT / "experiment.json").read_text())

    def test_gpu_csv_supports_capability_and_fallback(self):
        self.assertEqual(diagnose.parse_gpu_csv("0, Tesla T4, GPU-a, 15360, 12000, 580, 7.5\n")[0]["compute_capability"], "7.5")
        self.assertIsNone(diagnose.parse_gpu_csv("0, Tesla T4, GPU-a, 15360, 12000, 580\n")[0]["compute_capability"])
        self.assertEqual(diagnose.parse_gpu_csv("not valid GPU CSV"), [])

    def test_failed_import_still_has_complete_archive_and_raw_error(self):
        with tempfile.TemporaryDirectory() as root:
            receipt = self.collect(root, FakeCommands(broken_vllm=True))
            with zipfile.ZipFile(receipt["archive"]) as z:
                self.assertIsNone(z.testzip())
                status = json.loads(z.read("status.json"))
                self.assertEqual(status["collection_status"], "PASS")
                self.assertEqual(status["qualification"]["A_environment"]["status"], "FAIL")
                self.assertTrue(all(v["status"] == "UNKNOWN" for k, v in status["qualification"].items() if k != "A_environment"))
                self.assertIn("ImportError", z.read("logs/vllm.stderr").decode())
                for line in z.read("SHA256SUMS").decode().splitlines():
                    digest, path = line.split("  ", 1)
                    self.assertEqual(hashlib.sha256(z.read(path)).hexdigest(), digest)
            self.assertEqual(hashlib.sha256(Path(receipt["archive"]).read_bytes()).hexdigest(), receipt["sha256"])

    def test_missing_driver_does_not_prevent_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            receipt = self.collect(root, FakeCommands(missing_gpu=True))
            self.assertTrue(Path(receipt["archive"]).is_file())
            self.assertEqual(receipt["qualification"]["A_environment"]["status"], "FAIL")
            self.assertEqual(receipt["qualification"]["C_TWO_GPU_COMMUNICATION"]["status"], "UNKNOWN")

    def test_false_peer_access_does_not_claim_collective_failure(self):
        with tempfile.TemporaryDirectory() as root:
            receipt = self.collect(root, FakeCommands())
            self.assertEqual(receipt["qualification"]["A_environment"]["status"], "PASS")
            self.assertEqual(receipt["qualification"]["C_TWO_GPU_COMMUNICATION"]["status"], "UNKNOWN")
            self.assertTrue(receipt["qualification"]["A_environment"]["warnings"])

    def test_unknown_installed_version_requires_review_not_auto_upgrade(self):
        env = {"gpu_inventory": [{"name": "Tesla T4", "uuid": str(n), "total_mib": 15360, "free_mib": 10000} for n in range(2)],
               "torch": {"ok": True, "cuda_available": True, "gpu_count": 2, "nccl_available": True},
               "imports": {"vllm": {"ok": True, "version": "0.11.0"}, "triton": {"ok": True, "version": "3.7.1"}}}
        self.assertEqual(diagnose.classify(env, [], self.config())["qualification"]["A_environment"]["status"], "UNKNOWN")
        env["imports"]["vllm"]["version"] = "0.31.0.dev9"
        self.assertEqual(diagnose.classify(env, [], self.config())["qualification"]["A_environment"]["status"], "UNKNOWN")

    def test_secret_values_not_in_archive(self):
        secret = "hf_" + "S" * 32
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"HF_TOKEN": secret, "KAGGLE_KEY": "never-read-private-key"}):
            receipt = self.collect(root, FakeCommands(secret=secret))
            with zipfile.ZipFile(receipt["archive"]) as z:
                for name in z.namelist():
                    self.assertNotIn(secret, z.read(name).decode())
                    self.assertNotIn("never-read-private-key", z.read(name).decode())
                self.assertIn("[REDACTED]", z.read("logs/vllm.stderr").decode())
                self.assertNotIn("HF_TOKEN", json.loads(z.read("environment.json"))["selected_environment"])

    def test_repeated_runs_preserve_previous_evidence_and_unrelated_file(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / "unrelated.txt"; marker.write_text("keep")
            first = self.collect(root, FakeCommands())
            digest = hashlib.sha256(Path(first["archive"]).read_bytes()).hexdigest()
            second = self.collect(root, FakeCommands())
            self.assertNotEqual(first["archive"], second["archive"])
            self.assertEqual(hashlib.sha256(Path(first["archive"]).read_bytes()).hexdigest(), digest)
            self.assertEqual(marker.read_text(), "keep")

    def test_timeout_is_retained_and_later_stages_not_run(self):
        with tempfile.TemporaryDirectory() as root:
            receipt = self.collect(root, FakeCommands(timeout_torch=True))
            self.assertEqual(receipt["qualification"]["A_environment"]["status"], "FAIL")
            with zipfile.ZipFile(receipt["archive"]) as z:
                self.assertIn("timeout", z.read("logs/torch.stderr").decode())

    def test_diagnostic_does_not_install_load_models_or_run_collectives(self):
        fake = FakeCommands()
        with tempfile.TemporaryDirectory() as root:
            self.collect(root, fake)
        for _, argv, _ in fake.calls:
            self.assertNotIn("install", argv)
            self.assertNotIn("serve", argv)
            code = " ".join(argv)
            for forbidden in ("init_process_group(", "torch.empty(", "torch.zeros(", "torch.ones(", "from_pretrained(", "LLM("):
                self.assertNotIn(forbidden, code)

    def test_timeout_runner_keeps_partial_output_and_redacts_url_credentials(self):
        error = subprocess.TimeoutExpired(["probe"], timeout=1, output=b"partial output", stderr=b"partial error")
        with patch("diagnose.subprocess.run", side_effect=error):
            result = diagnose.run_command("synthetic-timeout", ["probe"], 1)
        self.assertEqual(result["exit_code"], 124)
        self.assertEqual(result["stdout"], "partial output")
        self.assertEqual(result["stderr"], "partial error")
        self.assertNotIn("secret", diagnose.redact("https://user:secret@example.com/repo"))


if __name__ == "__main__":
    unittest.main()
