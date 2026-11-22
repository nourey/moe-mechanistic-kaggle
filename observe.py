#!/usr/bin/env python3
"""Restore/reuse only the pinned runtime/checkpoint; run the observation once."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid
import zipfile

import model_qualify
import prepare_runtime as runtime

ROOT = Path(__file__).resolve().parent


def main():
    if not Path('/kaggle/working').is_dir(): raise RuntimeError('Run manually in Kaggle T4 x2, Internet On')
    print('Checking exact runtime/checkpoint; completed A/B/C and model qualification are not rerun.', flush=True)
    prep = model_qualify.qualify('/kaggle/working', prepare_only=True)
    if prep.get('runtime_and_checkpoint') != 'READY':
        print('EVIDENCE_JSON:' + json.dumps(dict(status='BLOCKED', preparation=prep, archive=prep['archive']))); return
    out = Path('/kaggle/working') / ('moe-observation-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    out.mkdir(); (out / 'logs').mkdir()
    shutil.copyfile(prep['archive'], out / 'runtime-preparation.zip')
    with zipfile.ZipFile(prep['archive']) as z:
        (out / 'SOURCE_COMMIT').write_bytes(z.read('logs/source-commit.stdout'))
    files = ['observe.py', 'owner_reversal.py', 'observation_contract.py', 'observation_scheduler.py',
             'observation_worker.py', 'observation_timing.py', 'observation.json', 'model-qualification.json', 'model-pin.json',
             'model_qualify.py', 'model_observer.py', 'native_model.py', 'runtime-lock.txt', 'prepare_runtime.py', 'diagnose.py']
    (out / 'SOURCE_SHA256SUMS').write_text(''.join(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in files))
    (out / 'source').mkdir()
    for n in files: shutil.copyfile(ROOT / n, out / 'source' / n)
    for n in ['observation.json', 'model-qualification.json', 'model-pin.json']: shutil.copyfile(ROOT / n, out / n)
    command = [prep['python'], '-u', '-B', str(ROOT / 'owner_reversal.py'), '--output', str(out),
               '--model', prep['model_path'], '--gpu-ids', json.dumps(prep['gpu_ids'])]
    print('Running native A-B-B-A observation …', flush=True)
    receipt = runtime.run('owner-reversal', command, 900)
    for stream in ('stdout', 'stderr'): (out / 'logs' / ('owner-reversal.' + stream)).write_text(model_qualify.safe_log(receipt.pop(stream)))
    (out / 'command.json').write_text(json.dumps(receipt, indent=2) + '\n')
    result = json.loads((out / 'observation-result.json').read_text()) if (out / 'observation-result.json').is_file() else dict(status='BLOCKED', reason='Native process failed before result; inspect logs')
    if receipt['exit_code'] != 0: result.update(status='BLOCKED', process_exit_code=receipt['exit_code'])
    (out / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
    print('EVIDENCE_JSON:' + json.dumps(dict(status=result['status'], **runtime.seal(out))))


if __name__ == '__main__': main()
