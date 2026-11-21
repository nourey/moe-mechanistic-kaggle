#!/usr/bin/env python3
"""Storage question before model acquisition; no torch, GPU or weight download."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from prepare_runtime import seal

ROOT = Path(__file__).resolve().parent


def probe(path, required_bytes):
    target = Path(path).expanduser()
    ancestor = target
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    ancestor = ancestor.resolve()
    row = dict(requested_path=str(target), target_exists=target.exists(), probed_existing_directory=str(ancestor),
               quota_verified=False, note='statvfs free bytes and a tiny write do not prove a container/user quota')
    try:
        usage = shutil.disk_usage(ancestor)
        row.update(total_bytes=usage.total, free_bytes=usage.free, device_id=ancestor.stat().st_dev,
                   checkpoint_fits_reported_free_bytes=usage.free >= required_bytes)
        with tempfile.TemporaryFile(prefix='.moe-storage-probe-', dir=str(ancestor)) as f:
            data = b'moe-storage-probe\n' * 128
            f.write(data); f.flush(); os.fsync(f.fileno()); f.seek(0)
            row['tiny_write_read_passed'] = f.read() == data
            row['probe_bytes'] = len(data)
    except OSError as error:
        row.update(tiny_write_read_passed=False, error=str(error))
    return row


def collect(output_root):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Run this information collector inside the existing Kaggle session')
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = root / ('moe-storage-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    out.mkdir()
    model = json.loads((ROOT / 'model-pin.json').read_text())
    candidates = ['/kaggle/working', '/kaggle/temp', str(Path.home() / '.cache/huggingface')]
    candidates += [os.environ[k] for k in ('HF_HOME', 'HF_HUB_CACHE') if os.environ.get(k)]
    rows = [probe(p, model['all_selected_artifact_bytes']) for p in dict.fromkeys(candidates)]
    memory = {}
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current',
                 '/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
        try:
            if Path(path).is_file(): memory[path] = Path(path).read_text().strip()[:128]
        except OSError as error: memory[path] = 'UNREADABLE: ' + str(error)
    result = dict(status='STORAGE_INFORMATION_COLLECTED', D_MODEL_LOADING='NOT_EXECUTED',
                  model_id=model['model_id'], model_revision=model['revision'],
                  required_checkpoint_artifact_bytes=model['all_selected_artifact_bytes'],
                  storage=rows, cgroup_memory_raw=memory,
                  earlier_venv_present=Path('/kaggle/working/moe-runtime-20261121T210750Z-1d519346-venv/bin/python').is_file(),
                  model_download_attempted=False, GPU_work_executed=False,
                  question='Choose an explicit model-cache path after reviewing these observations; no quota or model-residency PASS is implied.')
    result['git_head'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    result['git_status'] = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
    (out / 'storage.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'model-pin.json').write_bytes((ROOT / 'model-pin.json').read_bytes())
    names = ('model_storage.py', 'model-pin.json', 'prepare_runtime.py', 'diagnose.py')
    (out / 'SOURCE_SHA256SUMS').write_text(''.join(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in names))
    return dict(**seal(out), **result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default='/kaggle/working')
    args = parser.parse_args()
    result = collect(args.output_root)
    print(json.dumps(result, indent=2))
    print('EVIDENCE_JSON:' + json.dumps(result, separators=(',', ':')))
