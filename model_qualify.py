#!/usr/bin/env python3
"""Bounded fresh-session runtime restore, pinned OLMoE load and native EP2 smoke."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
import uuid

import prepare_runtime as runtime

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def json_line(text):
    return json.loads(next(s for s in reversed(text.splitlines()) if s.startswith('{')))


def lock_entries(text):
    rows = {}
    for line in text.splitlines():
        if not line or line.startswith('#'): continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s]+) --hash=sha256:([0-9a-f]{64})', line)
        if not match: raise ValueError('Invalid frozen wheel lock entry')
        name, version, sha = match.groups()
        key = re.sub(r'[-_.]+', '-', name).lower()
        if key in rows: raise ValueError('Duplicate frozen package')
        rows[key] = (version, sha)
    return rows


def verify_report(report, locked):
    runtime.verify_install_report(report)
    seen = {}
    for item in report['install']:
        name = re.sub(r'[-_.]+', '-', item['metadata']['name']).lower()
        seen[name] = (item['metadata']['version'], item['download_info']['archive_info']['hashes']['sha256'])
    if seen != locked: raise ValueError('Installed artifact report differs from the 199 reviewed wheels')
    return True


def verify_model_file(path, spec):
    path = Path(path)
    if path.stat().st_size != spec['bytes']: raise ValueError('Wrong model file size: ' + spec['path'])
    sha = hashlib.sha256()
    blob = hashlib.sha1(b'blob ' + str(spec['bytes']).encode() + b'\0')
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            sha.update(block); blob.update(block)
    expected = spec.get('lfs_sha256', spec.get('sha256'))
    # LFS Git object ID identifies its pointer, not the downloaded weight tensor.
    if expected and sha.hexdigest() != expected: raise ValueError('Wrong model SHA-256: ' + spec['path'])
    if not spec.get('lfs_sha256') and blob.hexdigest() != spec['git_blob_sha1']:
        raise ValueError('Wrong model Git blob: ' + spec['path'])
    return dict(path=spec['path'], bytes=spec['bytes'], sha256=sha.hexdigest(),
                verified_against='LFS SHA-256' if spec.get('lfs_sha256') else 'Pinned Git blob and available SHA-256')


def acquire(cache, output):
    # run() intentionally defaults to offline. Only this exact public checkpoint acquisition goes online.
    os.environ.update(HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0',
                      HF_HOME=str(Path(cache) / 'hf'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      HF_HUB_DISABLE_TELEMETRY='1', HF_XET_CHUNK_CACHE_SIZE_BYTES='0')
    from huggingface_hub import snapshot_download
    pin = json.loads((ROOT / 'model-pin.json').read_text())
    snapshot = Path(snapshot_download(repo_id=pin['model_id'], revision=pin['revision'],
        cache_dir=str(Path(cache) / 'hf' / 'hub'), allow_patterns=[p['path'] for p in pin['files']],
        token=False, max_workers=1))
    if snapshot.name != pin['revision']: raise ValueError('Snapshot revision mismatch')
    receipts = [verify_model_file(snapshot / row['path'], row) for row in pin['files']]
    dump(Path(output) / 'model-acquisition.json', dict(model_id=pin['model_id'], revision=pin['revision'],
         model_path=str(snapshot), files=receipts, quota_guarantee=False,
         scope='Hashes verified; BF16 checkpoint is explicitly converted to FP16 by native loader'))


def safe_log(text):
    text = runtime.diagnose.redact(text)
    # Signed download URL query strings are credentials, not necessary evidence.
    return re.sub(r'(https?://[^\s?\"<>]+)\?[^\s\"<>]+', r'\1?[QUERY_REDACTED]', text)


GPU_CODE = """import json, torch
rows=[]
for i in range(torch.cuda.device_count()):
 p=torch.cuda.get_device_properties(i)
 rows.append(dict(ordinal=i, uuid=str(getattr(p,'uuid','UNAVAILABLE')), name=p.name,
  capability=list(torch.cuda.get_device_capability(i)), memory_bytes=p.total_memory))
print(json.dumps({'devices':rows}))
"""


def parse_gpu_inventory(text):
    rows = [[v.strip() for v in line.split(',')] for line in text.splitlines() if line.strip()]
    if (len(rows) != 2 or any(len(r) != 5 for r in rows) or len({r[0] for r in rows}) != 2
            or not all('T4' in r[1] and int(r[3]) >= 12000 for r in rows)):
        raise ValueError('Require two distinct T4 GPUs with at least 12000 MiB initially free')
    return rows


def validate_cuda_identity(devices, nvrows):
    ids = [d['uuid'] for d in devices]
    if (len(ids) != 2 or len(set(ids)) != 2
            or {s.removeprefix('GPU-') for s in ids} != {s[0].removeprefix('GPU-') for s in nvrows}
            or {d['ordinal'] for d in devices} != {0, 1}
            or any(d['capability'] != [7, 5] for d in devices)):
        raise ValueError('CUDA and physical GPU identity/capability mismatch')
    return ids  # CUDA order is authoritative, not nvidia-smi row order.


def qualify(output_root, execute=runtime.run, work_root=None):
    config = json.loads((ROOT / 'model-qualification.json').read_text())
    lock = (ROOT / 'runtime-lock.txt').read_bytes()
    now = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(output_root).resolve() / ('moe-model-' + now + '-' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True); (out / 'logs').mkdir()
    status = dict(D_MODEL_RESIDENCY='BLOCKED', E_FUNCTIONAL_GENERATION='BLOCKED',
                  A_B_C_repeated=False, production_readiness='NOT_ESTABLISHED', commands=[])
    deadline = time.monotonic() + config['total_timeout_seconds']
    work = Path('/kaggle/working') if work_root is None else Path(work_root)
    before = None

    def capture(label, argv, limit=120):
        status['last_step'] = label
        print('Running ' + label + ' …', flush=True)
        remaining = int(deadline - time.monotonic())
        if remaining <= 0: raise RuntimeError('Total 40-minute bound reached')
        receipt = execute(label, [str(s) for s in argv], min(limit, remaining))
        for stream in ('stdout', 'stderr'):
            receipt[stream] = safe_log(receipt[stream])
            (out / 'logs' / (label + '.' + stream)).write_text(receipt[stream])
        status['commands'].append({k: v for k, v in receipt.items() if k not in ('stdout', 'stderr')})
        dump(out / 'status.json', status)
        if receipt['exit_code']: raise RuntimeError(label + ' exited ' + str(receipt['exit_code']) + '; see retained stdout/stderr')
        return receipt['stdout']

    try:
        source_names = ('model_qualify.py', 'native_model.py', 'model_observer.py', 'model-pin.json',
                        'model-qualification.json', 'runtime-lock.txt', 'prepare_runtime.py', 'diagnose.py')
        (out / 'SOURCE_SHA256SUMS').write_text(''.join(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in source_names))
        for name in ('model-pin.json', 'model-qualification.json', 'runtime-lock.txt'): shutil.copyfile(ROOT / name, out / name)
        capture('source-commit', ['git', '-C', ROOT, 'rev-parse', 'HEAD'])
        dirty = capture('source-status', ['git', '-C', ROOT, 'status', '--porcelain'])
        if dirty.strip(): raise ValueError('Runner source checkout is not clean')
        if hashlib.sha256(lock).hexdigest() != config['runtime_lock_sha256']: raise ValueError('Runtime lock hash mismatch')
        locked = lock_entries(lock.decode())
        if len(locked) != config['runtime_packages']: raise ValueError('Wrong cohort size')
        if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64') or sys.version_info[:2] != (3, 13):
            raise RuntimeError('This exact wheel cohort requires Kaggle Linux x86_64 / Python 3.13')
        if not work.is_dir(): raise RuntimeError('Kaggle working directory missing')
        before = json_line(capture('base-before', [sys.executable, '-c', runtime.VERSIONS]))
        nvml = capture('gpu-preflight', ['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits'])
        nvrows = parse_gpu_inventory(nvml)
        venv = work / ('moe-model-' + config['runtime_lock_sha256'][:12] + '-venv')
        py = venv / 'bin' / 'python'
        marker = venv / 'COHORT.json'
        if venv.exists():
            if not marker.is_file() or json.loads(marker.read_text()).get('lock_sha256') != config['runtime_lock_sha256']:
                raise RuntimeError('Existing environment is unverified or incomplete; not overwritten or silently reinstalled')
            prior = venv / 'pip-install-report.json'
            verify_report(json.loads(prior.read_text()), locked)
            shutil.copyfile(prior, out / 'pip-install-report.json')
            status['runtime_reused'] = True
        else:
            if shutil.disk_usage(work).free < 11 * 1024 ** 3: raise RuntimeError('Less than 11 GiB working space for isolated runtime')
            capture('create-venv', [sys.executable, '-m', 'venv', '--without-pip', venv])
            seed = out / 'seed-pip.txt'
            seed.write_text('pip==' + config['seed_pip_version'] + ' --hash=sha256:' + config['seed_pip_sha256'] + '\n')
            capture('seed-pip', [sys.executable, '-m', 'pip', '--python', py, 'install', '--only-binary=:all:',
                    '--require-hashes', '--no-cache-dir', '--index-url', 'https://pypi.org/simple', '-r', seed], 300)
            capture('install', [py, '-m', 'pip', 'install', '--only-binary=:all:', '--require-hashes', '--no-cache-dir',
                    '--index-url', 'https://pypi.org/simple', '--report', out / 'pip-install-report.json',
                    '-r', ROOT / 'runtime-lock.txt'], config['install_timeout_seconds'])
            verify_report(json.loads((out / 'pip-install-report.json').read_text()), locked)
            shutil.copyfile(out / 'pip-install-report.json', venv / 'pip-install-report.json')
        capture('pip-check', [py, '-m', 'pip', 'check'])
        packages = json.loads(capture('pip-list', [py, '-m', 'pip', 'list', '--format=json']))
        installed = {re.sub(r'[-_.]+', '-', p['name']).lower(): p['version'] for p in packages}
        for name, (version, _) in locked.items():
            if installed.get(name) != version: raise ValueError('Runtime cohort changed: ' + name)
        capture('native-import', [py, '-c', runtime.NATIVE_IMPORT])
        dump(marker, dict(lock_sha256=config['runtime_lock_sha256'], source='Exact previous successful wheel cohort'))
        devices = json_line(capture('cuda-device-identity', [py, '-c', GPU_CODE]))['devices']
        ids = validate_cuda_identity(devices, nvrows)
        dump(out / 'cuda-devices.json', devices)
        cache = Path(config['model_cache_root'])
        cache.mkdir(parents=True, exist_ok=True)
        owner = cache / 'OWNER.json'
        if not owner.exists() and any(cache.iterdir()): raise RuntimeError('Model cache contains unowned files; not modified')
        if owner.exists() and json.loads(owner.read_text()).get('experiment') != 'moe-mechanistic-1525': raise RuntimeError('Wrong cache owner')
        dump(owner, dict(experiment='moe-mechanistic-1525', deletion_policy='Never deleted by runner'))
        if cache.stat().st_dev == work.stat().st_dev: raise RuntimeError('Model cache and 20 GB working directory share storage; refusing oversized combined download')
        with tempfile.TemporaryFile(dir=cache) as f: f.write(b'probe'); f.seek(0); assert f.read() == b'probe'
        usage = shutil.disk_usage(cache)
        dump(out / 'storage.json', dict(cache=str(cache), device_id=cache.stat().st_dev, free_bytes=usage.free,
             working_device_id=work.stat().st_dev, writable_probe=True, quota_verified=False))
        # Reported statvfs capacity is a necessary check, never a user quota guarantee.
        if usage.free < 15 * 1024 ** 3: raise RuntimeError('Less than 15 GiB model-cache space reported')
        capture('model-acquisition', [py, ROOT / 'model_qualify.py', '--acquire', '--cache', cache, '--output', out], config['download_timeout_seconds'])
        model_path = json.loads((out / 'model-acquisition.json').read_text())['model_path']
        capture('native-model', [py, ROOT / 'native_model.py', '--output', out, '--model-path', model_path,
                '--gpu-ids', json.dumps(ids)], config['model_timeout_seconds'])
    except Exception:
        status['error'] = safe_log(traceback.format_exc())
    finally:
        native = out / 'native-result.json'
        if native.exists(): status.update(json.loads(native.read_text()))
        try:
            after = json_line(capture('base-after', [sys.executable, '-c', runtime.VERSIONS], 30))
            status['base_selected_versions_unchanged'] = None if before is None else after == before
            if before is not None and after != before: status['error'] = 'Base selected package versions changed'
        except Exception: status['base_verification_error'] = safe_log(traceback.format_exc())
        dump(out / 'status.json', status)
        sealed = runtime.seal(out)
    return dict(**{k: v for k, v in status.items() if k != 'commands'}, **sealed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/kaggle/working')
    parser.add_argument('--acquire', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--cache', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.acquire:
        acquire(args.cache, args.output); return
    result = qualify(args.output)
    print('EVIDENCE_JSON:' + json.dumps(result))


if __name__ == '__main__':
    main()
