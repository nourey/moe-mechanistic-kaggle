#!/usr/bin/env python3
"""One isolated binary installation attempt; no model, kernel or collective test."""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import traceback
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import diagnose

ROOT = Path(__file__).resolve().parent
VLLM_WHEEL = "vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_SHA256 = "39132b433967f0962d03d8a5f633b9bee6b32708f566c7d5637b79f8d0ef3433"
VERSIONS = """import importlib.metadata as m, json, sys
names = ['torch', 'vllm', 'triton', 'transformers', 'pip']
versions = {}
for name in names:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python': sys.executable, 'prefix': sys.prefix, 'versions': versions}))
"""
NATIVE_IMPORT = """import json, sys, torch, vllm, vllm._C, triton, transformers
if sys.prefix == sys.base_prefix: raise RuntimeError('Not inside isolated venv')
if torch.__version__.split('+')[0] != '2.13.0': raise RuntimeError(torch.__version__)
if vllm.__version__ != '0.31.0': raise RuntimeError(vllm.__version__)
print(json.dumps({'torch': torch.__version__, 'vllm': vllm.__version__,
                 'native_extension': vllm._C.__file__, 'triton': triton.__version__,
                 'transformers': transformers.__version__}))
"""


def run(label, argv, timeout):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('PIP_') or key in ('PYTHONPATH', 'PYTHONHOME', 'HF_TOKEN',
                'HUGGING_FACE_HUB_TOKEN', 'KAGGLE_KEY', 'GH_TOKEN', 'GITHUB_TOKEN',
                'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN'):
            env.pop(key)
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    # A timed-out pip must not leave a child installation running.
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env, start_new_session=True)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
    return dict(label=label, argv=argv, started_utc=started, exit_code=124 if timed_out else proc.returncode,
                timed_out=timed_out, stdout=diagnose.redact(out), stderr=diagnose.redact(err))


def verify_install_report(report):
    installs = report.get('install', [])
    matches = [p for p in installs if p.get('metadata', {}).get('name', '').lower() == 'vllm']
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous vLLM wheel provenance')
    for package in installs:
        url = package['download_info']['url']
        if urlparse(url).hostname != 'files.pythonhosted.org' or not urlparse(url).path.endswith('.whl'):
            raise ValueError('Unexpected non-PyPI wheel source')
    package = matches[0]
    info = package['download_info']
    if (package['metadata']['version'] != '0.31.0'
            or urlparse(info['url']).path.rsplit('/', 1)[-1] != VLLM_WHEEL
            or info['archive_info']['hashes']['sha256'] != VLLM_SHA256):
        raise ValueError('vLLM wheel differs from reviewed release/hash')
    return dict(vllm_wheel=VLLM_WHEEL, sha256=VLLM_SHA256,
                dependency_selection='See full pip-install-report.json; transitive versions recorded, not a frozen lock.')


def seal(out):
    # Environment directories and model files are never included.
    files = sorted(p for p in out.rglob('*') if p.is_file())
    sums = [(str(p.relative_to(out)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    (out / 'SHA256SUMS').write_text(''.join(sha + '  ' + name + '\n' for name, sha in sums))
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p, str(p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise RuntimeError('Evidence ZIP integrity failure')
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(sha + '  ' + archive.name + '\n')
    return dict(archive=str(archive), sha256=sha)


def prepare(output_root, execute=run, inventory=diagnose.collect):
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = 'moe-runtime-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    out = root / run_id
    out.mkdir(exist_ok=False)
    (out / 'logs').mkdir()
    venv = root / (run_id + '-venv')  # Outside archive and outside the Git checkout.
    commands, status = [], dict(runtime_preparation='FAIL', failed_step='preflight',
        scope='Stage A isolated binary installation/native imports only. B/C/D/E/M NOT_EXECUTED.',
        source_note='PyPI release wheel/hash, not a proof of the exact reviewed source commit or producer patch.')
    before = None

    def capture(label, argv, timeout=90):
        status['failed_step'] = label
        print('Running ' + label + ' ...', flush=True)
        result = execute(label, [str(a) for a in argv], timeout)
        for stream in ('stdout', 'stderr'):
            (out / 'logs' / (label + '.' + stream)).write_text(diagnose.redact(result[stream]))
        commands.append({k: v for k, v in result.items() if k not in ('stdout', 'stderr')})
        if result['exit_code'] != 0:
            raise RuntimeError(label + ' failed; inspect logs/' + label + '.stderr')
        return result['stdout']

    try:
        if platform.system() != 'Linux' or platform.machine() != 'x86_64' or not Path('/kaggle/working').is_dir():
            raise RuntimeError('This installer requires the reviewed Linux x86_64 Kaggle environment')
        # Do not run twice against an existing environment; preserve previous attempts.
        if venv.exists(): raise RuntimeError('Refusing to reuse an existing venv')
        (out / 'runtime-requirements.txt').write_bytes((ROOT / 'runtime-requirements.txt').read_bytes())
        capture('git-head', ['git', '-C', ROOT, 'rev-parse', 'HEAD'])
        if capture('git-status', ['git', '-C', ROOT, 'status', '--porcelain']).strip():
            raise RuntimeError('Runner checkout has changes; refusing installation')
        tracked = capture('source-files', ['git', '-C', ROOT, 'ls-files']).splitlines()
        (out / 'SOURCE_SHA256SUMS').write_text(''.join(
            hashlib.sha256((ROOT / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in tracked))
        before = json.loads(capture('base-before', [sys.executable, '-c', VERSIONS]))
        initial = inventory(out / 'before')
        status['failed_step'] = 'gpu-preflight'
        env = json.loads((Path(initial['evidence_directory']) / 'environment.json').read_text())
        gpus = env.get('gpu_inventory', [])
        if (len(gpus) != 2 or len({g.get('uuid') for g in gpus}) != 2
                or any('T4' not in g.get('name', '') or g.get('compute_capability') != '7.5' for g in gpus)):
            raise RuntimeError('Expected exactly two identifiable SM75 T4 GPUs; no install attempted')
        capture('create-venv', [sys.executable, '-m', 'venv', '--without-pip', venv])
        py = venv / 'bin/python'
        # Avoid ensurepip; the Kaggle base pip already supports --python.
        capture('seed-pip', [sys.executable, '-m', 'pip', '--python', py, '--no-cache-dir',
                'install', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', 'pip==26.2.2'], 300)
        capture('install', [py, '-m', 'pip', '--no-cache-dir', 'install', '--only-binary=:all:',
                '--index-url', 'https://pypi.org/simple', '--report', out / 'pip-install-report.json',
                '-r', ROOT / 'runtime-requirements.txt'], 1800)
        status['failed_step'] = 'verify-wheel'
        receipt = verify_install_report(json.loads((out / 'pip-install-report.json').read_text()))
        (out / 'wheel-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
        capture('pip-check', [py, '-m', 'pip', 'check'])
        capture('native-import', [py, '-c', NATIVE_IMPORT], 120)
        # Use sanitized subprocess environment here too, rather than the base notebook's PYTHONPATH.
        text = capture('isolated-inventory', [py, '-B', ROOT / 'diagnose.py', '--output-root', out / 'after'], 300)
        result = json.loads(next(line.split(':', 1)[1] for line in reversed(text.splitlines()) if line.startswith('EVIDENCE_JSON:')))
        (out / 'isolated-inventory-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        if result['qualification']['A_environment']['status'] != 'PASS':
            raise RuntimeError('Isolated Stage A preconditions did not pass; no later stage executed')
        status.update(runtime_preparation='PASS', failed_step=None, isolated_python=str(py))
    except Exception:
        status['error'] = diagnose.redact(traceback.format_exc())
    finally:
        if before is not None:
            previous_step = status['failed_step']
            try:
                after = json.loads(capture('base-after', [sys.executable, '-c', VERSIONS]))
                status['base_selected_versions_unchanged'] = before == after
                if before != after:
                    status.update(runtime_preparation='FAIL', error='Selected Kaggle base versions changed')
            except Exception:
                status.update(runtime_preparation='FAIL', base_verification_error=diagnose.redact(traceback.format_exc()))
            status['failed_step'] = previous_step
        status['venv_preserved_at'] = str(venv)
        status['working_disk_free_bytes_after'] = shutil.disk_usage(root).free
        (out / 'commands.json').write_text(diagnose.redact(json.dumps(commands, indent=2)) + '\n')
        (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    return dict(**seal(out), **status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default='/kaggle/working')
    args = parser.parse_args()
    result = prepare(args.output_root)
    print(json.dumps(result, indent=2))
    print('EVIDENCE_JSON:' + json.dumps(result, separators=(',', ':')))
    # Packaging success does not imply runtime_preparation PASS; inspect status.json.
