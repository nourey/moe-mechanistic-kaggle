#!/usr/bin/env python3
"""Read-only environment discovery; no installations, tensors, models or collectives.

Only the standard library is required. Each optional import is isolated and bounded.
Collection success is distinct from qualification status. Always return the ZIP.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MARKER = "PROBE_JSON:"
SAFE_ENV = ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
            "TORCH_HOME", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "NCCL_P2P_DISABLE",
            "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME",
            "VLLM_WORKER_MULTIPROC_METHOD")
PACKAGES = ("torch", "vllm", "triton", "flashinfer-python", "transformers",
            "huggingface-hub", "safetensors", "numpy", "pip", "nvidia-nccl-cu12",
            "nvidia-cuda-runtime-cu12", "tokenspeed-triton")


def redact(value):
    value = str(value)
    value = re.sub(r"\b(?:hf_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{16,}|KGAT_[A-Za-z0-9_]{12,})\b", "[REDACTED]", value)
    value = re.sub(r"(?i)(Bearer\s+)[^\s\"']+", r"\1[REDACTED]", value)
    value = re.sub(r"(https?://)[^\s/:@]+:[^\s/@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)((?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*)[^\s,;\"']+", r"\1[REDACTED]", value)
    return value


def run_command(label, argv, timeout):
    started, begin = dt.datetime.now(dt.timezone.utc).isoformat(), time.monotonic()
    env = os.environ.copy()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "KAGGLE_KEY", "GITHUB_TOKEN", "GH_TOKEN",
                "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                env=env, shell=False)
        code, out, err, timed_out = result.returncode, result.stdout, result.stderr, False
    except FileNotFoundError as e:
        code, out, err, timed_out = 127, "", str(e), False
    except subprocess.TimeoutExpired as e:
        def text(x): return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
        code, out, err, timed_out = 124, text(e.stdout), text(e.stderr), True
    except Exception:
        code, out, err, timed_out = 125, "", traceback.format_exc(), False
    return {"label": label, "argv": [redact(x) for x in argv], "started_utc": started,
            "duration_seconds": round(time.monotonic() - begin, 6), "exit_code": code,
            "timed_out": timed_out, "stdout": redact(out), "stderr": redact(err)}


def probe_value(command):
    for line in reversed(command.get("stdout", "").splitlines()):
        if line.startswith(MARKER):
            try: return json.loads(line[len(MARKER):])
            except json.JSONDecodeError: return {"ok": False, "error": "Malformed probe JSON; raw output retained."}
    return {"ok": False, "error": "No structured result; raw stdout/stderr retained."}


def import_code(name):
    return """import importlib, importlib.metadata, json, traceback
result = {"module": %r, "ok": False}
try:
    module = importlib.import_module(%r)
    result.update(ok=True, version=str(getattr(module, "__version__", "unknown")), file=str(getattr(module, "__file__", "unknown")))
except Exception as e:
    result["error"] = type(e).__name__ + ": " + str(e)
    traceback.print_exc()
print("PROBE_JSON:" + json.dumps(result))
raise SystemExit(0 if result["ok"] else 1)
""" % (name, name)


TORCH_CODE = r"""import ctypes, json, traceback
result = {"ok": False}
try:
    import torch
    result.update(ok=True, version=str(torch.__version__), build_cuda=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), gpu_count=torch.cuda.device_count(),
                  devices=[], peer_access=[], torch_arch_list=[])
    if result["cuda_available"]:
        result["torch_arch_list"] = torch.cuda.get_arch_list()
        for i in range(result["gpu_count"]):
            p = torch.cuda.get_device_properties(i)
            result["devices"].append({"index": i, "name": p.name, "compute_capability": [p.major, p.minor], "total_memory_bytes": p.total_memory})
        for i in range(result["gpu_count"]):
            for j in range(result["gpu_count"]):
                if i != j:
                    try: result["peer_access"].append({"from": i, "to": j, "driver_reports_access": torch.cuda.can_device_access_peer(i, j)})
                    except Exception as e: result["peer_access"].append({"from": i, "to": j, "error": str(e)})
    import torch.distributed as dist
    result["distributed_available"] = dist.is_available()
    result["nccl_available"] = dist.is_available() and dist.is_nccl_available()
    try: result["nccl_version"] = torch.cuda.nccl.version() if result["nccl_available"] else None
    except Exception as e: result["nccl_version_error"] = str(e)
    candidates = []
    if torch.version.cuda: candidates.append("libcudart.so." + torch.version.cuda.split(".")[0])
    candidates.append("libcudart.so")
    runtime_errors = []
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            version = ctypes.c_int()
            code = lib.cudaRuntimeGetVersion(ctypes.byref(version))
            result.update(loaded_cudart_version=version.value if code == 0 else None,
                          cuda_runtime_query_return=code, cudart_soname=name)
            break
        except Exception as e: runtime_errors.append(name + ": " + str(e))
    if "loaded_cudart_version" not in result: result["cuda_runtime_query_errors"] = runtime_errors
except Exception as e:
    result["error"] = type(e).__name__ + ": " + str(e)
    traceback.print_exc()
print("PROBE_JSON:" + json.dumps(result))
raise SystemExit(0 if result["ok"] else 1)
"""

PACKAGE_CODE = """import importlib.metadata as m, json
names = %r
versions = {}
for name in names:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
result = {"ok": True, "versions": versions}
try:
    from huggingface_hub import constants
    result["model_cache"] = {"hf_home": constants.HF_HOME, "hf_hub_cache": constants.HF_HUB_CACHE}
except Exception as e: result["model_cache_query_error"] = type(e).__name__ + ": " + str(e)
print("PROBE_JSON:" + json.dumps(result))
""" % (PACKAGES,)


def parse_gpu_csv(text):
    devices = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) not in (6, 7): continue
        try:
            devices.append({"index": int(row[0].strip()), "name": row[1].strip(), "uuid": row[2].strip(),
                            "total_mib": float(row[3].strip()), "free_mib": float(row[4].strip()),
                            "driver_version": row[5].strip(), "compute_capability": row[6].strip() if len(row) == 7 else None})
        except ValueError: continue
    return devices


def classify(environment, commands, config):
    reasons, warnings = [], []
    gpu = environment.get("gpu_inventory", [])
    torch = environment.get("torch", {})
    expected = config["candidate_experiment"]["expected_gpu_count"]
    if len(gpu) != expected: reasons.append("Expected two GPUs in NVIDIA inventory; actual count=" + str(len(gpu)))
    if gpu and any(not re.search(r"\bT4\b", x["name"]) for x in gpu): reasons.append("GPU model differs from the requested T4 target; review before changing the experiment domain.")
    if len({x["uuid"] for x in gpu}) != len(gpu): reasons.append("NVIDIA inventory contains duplicate GPU UUIDs.")
    if any(x["total_mib"] <= 0 or x["free_mib"] < 0 or x["free_mib"] > x["total_mib"] for x in gpu): reasons.append("GPU memory inventory is inconsistent.")
    if not torch.get("ok"): reasons.append("PyTorch import failed or is unavailable; inspect torch.stderr.")
    elif not torch.get("cuda_available") or torch.get("gpu_count") != expected: reasons.append("PyTorch does not expose the expected two CUDA devices.")
    if torch.get("ok") and not torch.get("nccl_available"): reasons.append("PyTorch NCCL backend is unavailable; no collective was attempted.")
    for name in ("vllm", "triton"):
        if not environment.get("imports", {}).get(name, {}).get("ok"):
            reasons.append(name + " import failed or is missing; inspect " + name + ".stderr.")
    if any(d.get("compute_capability") == [7, 5] for d in torch.get("devices", [])):
        warnings.append("SM75 observed. FP16 MoE kernel compatibility remains UNKNOWN; dense inference, imports and CUDA enumeration do not prove it.")
    if any(not p.get("driver_reports_access", False) for p in torch.get("peer_access", [])):
        warnings.append("Direct CUDA peer access absent/unverified. NCCL may use host/shared-memory paths; transport and communication remain UNKNOWN.")
    current = environment.get("imports", {}).get("vllm", {}).get("version", "")
    candidate = config["candidate_vllm"]["release"]
    version_review = current and current != "unknown" and current.split("+", 1)[0] != candidate
    if version_review: warnings.append("Installed vLLM differs from reviewed candidate " + candidate + "; source/feature review required, no automatic upgrade.")
    stage_a = "FAIL" if reasons else ("UNKNOWN" if version_review else "PASS")
    stages = {"A_environment": {"status": stage_a, "scope": "Inventory/import preconditions only; no kernel or serving qualification.", "reasons": reasons, "warnings": warnings}}
    for stage in config["later_stages"]:
        stages[stage] = {"status": "UNKNOWN", "reason": "NOT_EXECUTED: review this environment evidence first."}
    return {"collection_status": "PASS", "qualification": stages,
            "next_action": "Return this evidence ZIP for review. Do not install dependencies or advance to model serving.",
            "scientific_boundary": config["scientific_boundary"]}


def collect(output_root, execute=run_command, repo_root=ROOT):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = "moe-env-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = output_root / run_id
    out.mkdir(exist_ok=False)
    logs = out / "logs"; logs.mkdir()
    commands = []
    def capture(label, argv, timeout=30):
        print("Collecting " + label + " ...", flush=True)
        try: result = execute(label, argv, timeout)
        except Exception: result = {"label": label, "argv": argv, "exit_code": 125, "timed_out": False, "stdout": "", "stderr": traceback.format_exc()}
        result["stdout"], result["stderr"] = redact(result.get("stdout", "")), redact(result.get("stderr", ""))
        (logs / (label + ".stdout")).write_text(result["stdout"])
        (logs / (label + ".stderr")).write_text(result["stderr"])
        commands.append({k: v for k, v in result.items() if k not in ("stdout", "stderr")})
        commands[-1].update(stdout="logs/" + label + ".stdout", stderr="logs/" + label + ".stderr")
        return result
    config = json.loads((repo_root / "experiment.json").read_text())
    environment = {"run_id": run_id, "utc_started": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
                   "python_executable": sys.executable, "glibc": list(platform.libc_ver()),
                   "kaggle_path_present": Path("/kaggle/working").is_dir(),
                   "selected_environment": {k: redact(os.environ[k]) for k in SAFE_ENV if k in os.environ},
                   "disk": {}, "imports": {}, "gpu_inventory": []}
    for path in dict.fromkeys([str(output_root), "/kaggle/working", "/dev/shm", str(Path.home() / ".cache"), os.environ.get("HF_HOME", ""), os.environ.get("HF_HUB_CACHE", "")]):
        if path and Path(path).exists():
            try:
                usage = shutil.disk_usage(path)
                environment["disk"][redact(path)] = {"total_bytes": usage.total, "free_bytes": usage.free}
            except OSError as e: environment["disk"][redact(path)] = {"error": redact(e)}
    environment["model_cache_note"] = "Only configured paths/free space collected; no cache contents, tokens or model files inspected."
    banner = capture("nvidia-smi", ["nvidia-smi"], 15)
    (out / "nvidia-smi.log").write_text(banner["stdout"] + banner["stderr"])
    fields = "index,name,uuid,memory.total,memory.free,driver_version"
    query = capture("gpu-query", ["nvidia-smi", "--query-gpu=" + fields + ",compute_cap", "--format=csv,noheader,nounits"], 15)
    if query["exit_code"] != 0:
        query = capture("gpu-query-fallback", ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"], 15)
    environment["gpu_inventory"] = parse_gpu_csv(query["stdout"]) if query["exit_code"] == 0 else []
    match = re.search(r"CUDA Version:\s*([0-9.]+)", banner["stdout"])
    environment["driver_advertised_cuda_max"] = match.group(1) if match else None
    environment["cuda_version_semantics"] = "nvidia-smi reports driver capability; torch build_cuda and queried loaded_cudart are separate facts. nvcc is optional."
    capture("topology", ["nvidia-smi", "topo", "-m"], 15)
    capture("nvcc-version", ["nvcc", "--version"], 10)
    capture("pip-version", [sys.executable, "-m", "pip", "--version"], 15)
    environment["packages"] = probe_value(capture("package-versions", [sys.executable, "-c", PACKAGE_CODE], 20))
    for cache_path in environment["packages"].get("model_cache", {}).values():
        if Path(cache_path).exists():
            try:
                usage = shutil.disk_usage(cache_path)
                environment["disk"][redact(cache_path)] = {"total_bytes": usage.total, "free_bytes": usage.free}
            except OSError as e: environment["disk"][redact(cache_path)] = {"error": redact(e)}
    environment["torch"] = probe_value(capture("torch", [sys.executable, "-c", TORCH_CODE], 45))
    for name in ("vllm", "triton"):
        environment["imports"][name] = probe_value(capture(name, [sys.executable, "-c", import_code(name)], 40))
    source = {"repository_url": "https://github.com/nourey/moe-mechanistic-kaggle", "run_id": run_id, "source_sha256": {}}
    for label, argv in [("git-head", ["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
                        ("git-status", ["git", "-C", str(repo_root), "status", "--porcelain"]),
                        ("git-origin", ["git", "-C", str(repo_root), "remote", "get-url", "origin"])]:
        result = capture(label, argv, 10)
        source[label] = result["stdout"].strip() if result["exit_code"] == 0 else None
    for name in ("diagnose.py", "experiment.json", "requirements.txt", "README.md", "Kaggle_Environment_Diagnostic.ipynb", "tests/test_diagnose.py"):
        file = repo_root / name
        if file.is_file(): source["source_sha256"][name] = hashlib.sha256(file.read_bytes()).hexdigest()
    source["revision_note"] = "Git HEAD plus source hashes identify this runner, not the installed vLLM source/build. No model revision is qualified yet."
    status = classify(environment, commands, config)
    for name, value in [("environment.json", environment), ("status.json", status), ("commands.json", commands), ("source.json", source), ("experiment-config.json", config)]:
        (out / name).write_text(redact(json.dumps(value, indent=2, allow_nan=False)) + "\n")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {"scope": "Stage A environment discovery only; B/C/D/E/M not executed.", "files": [{"path": str(p.relative_to(out)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in files]}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files.append(out / "MANIFEST.json")
    (out / "SHA256SUMS").write_text("\n".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(out)) for p in files) + "\n")
    archive = output_root / (run_id + ".zip")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.rglob("*")):
            if file.is_file(): z.write(file, str(file.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise RuntimeError("Evidence ZIP integrity failure")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(archive_sha + "  " + archive.name + "\n")
    return {"evidence_directory": str(out), "archive": str(archive), "sha256": archive_sha,
            "collection_status": status["collection_status"], "qualification": status["qualification"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/kaggle/working", help="New unique evidence directory/ZIP will be created here; existing runs are preserved.")
    args = parser.parse_args()
    result = collect(args.output_root)
    print(json.dumps(result, indent=2))
    print("EVIDENCE_JSON:" + json.dumps(result, separators=(",", ":")))
    # Exit 0 means collection/archive completed, not qualification PASS.
