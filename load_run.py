#!/usr/bin/env python3
"""Restore/reuse the frozen runtime, run one bounded load protocol, preserve all evidence."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import zipfile

import model_qualify
import prepare_runtime as runtime
from load_experiment import CODE

ROOT=Path(__file__).resolve().parent

def retained_result(out,receipt):
    try:result=json.loads((out/'load-result.json').read_text())
    except (OSError,ValueError) as error:
        result=dict(status='BLOCKED',reason='Native result missing/incomplete; original raw file and logs retained',read_error=repr(error))
    if receipt['exit_code']!=0:result.update(status='BLOCKED',process_exit_code=receipt['exit_code'])
    return result

def main():
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Manual Kaggle T4 x2 / Internet On only')
    print('Reusing/restoring exact runtime and model; previous qualifications are not repeated.',flush=True)
    prep=model_qualify.qualify('/kaggle/working',prepare_only=True)
    if prep.get('runtime_and_checkpoint')!='READY':
        print('EVIDENCE_JSON:'+json.dumps(dict(status='BLOCKED',archive=prep['archive'])));return
    out=Path('/kaggle/working')/('moe-load-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    out.mkdir();(out/'logs').mkdir();(out/'source').mkdir()
    shutil.copyfile(prep['archive'],out/'runtime-preparation.zip')
    with zipfile.ZipFile(prep['archive']) as z:(out/'SOURCE_COMMIT').write_bytes(z.read('logs/source-commit.stdout'))
    for n in CODE:shutil.copyfile(ROOT/n,out/'source'/n)
    (out/'SOURCE_SHA256SUMS').write_text(''.join(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in CODE))
    print('Running mapping → bounded range discovery → one load/recovery trajectory …',flush=True)
    receipt=runtime.run('load',[prep['python'],'-u','-B',str(ROOT/'load_experiment.py'),'--output',str(out),
        '--model',prep['model_path'],'--gpu-ids',json.dumps(prep['gpu_ids'])],900)
    for stream in ('stdout','stderr'):(out/'logs'/('load.'+stream)).write_text(model_qualify.safe_log(receipt.pop(stream)))
    (out/'command.json').write_text(json.dumps(receipt,indent=2)+'\n')
    result=retained_result(out,receipt)
    (out/'status.json').write_text(json.dumps(result,indent=2)+'\n')
    print('EVIDENCE_JSON:'+json.dumps(dict(status=result['status'],**runtime.seal(out))))

if __name__=='__main__':main()
