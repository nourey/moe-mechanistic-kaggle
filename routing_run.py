#!/usr/bin/env python3
"""Manual Part A or freeze-fenced Part B; never automatically chains stages."""
import argparse
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

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['discover','compare'])
    p.add_argument('--freeze');p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.stage=='compare':
        if not a.freeze or not a.freeze_sha256 or not Path(a.freeze).is_file():p.error('Part B requires the actual Part A freeze file and exact SHA-256')
        if hashlib.sha256(Path(a.freeze).read_bytes()).hexdigest()!=a.freeze_sha256:p.error('Freeze hash mismatch; no runtime started')
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Run manually in Kaggle T4 x2, Internet On')
    print('Reusing/restoring exact runtime/checkpoint only; A/B/C/model qualification not repeated.',flush=True)
    prep=model_qualify.qualify('/kaggle/working',prepare_only=True)
    if prep.get('runtime_and_checkpoint')!='READY':
        print('EVIDENCE_JSON:'+json.dumps(dict(status='BLOCKED',archive=prep['archive'])));return
    out=Path('/kaggle/working')/('moe-routing-'+a.stage+'-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    out.mkdir();(out/'logs').mkdir();(out/'source').mkdir()
    shutil.copyfile(prep['archive'],out/'runtime-preparation.zip')
    with zipfile.ZipFile(prep['archive']) as z:
        (out/'SOURCE_COMMIT').write_bytes(z.read('logs/source-commit.stdout'))
        drivers=[r.split(',')[-1].strip() for r in z.read('logs/gpu-preflight.stdout').decode().splitlines() if r.strip()]
    (out/'context.json').write_text(json.dumps(dict(driver_versions=drivers,gpu_ids=prep['gpu_ids'],
        hardware='T4 x2 / SM75; GPU UUIDs observed independently in each stage',
        model_revision=json.loads((ROOT/'model-pin.json').read_text())['revision']),indent=2)+'\n')
    files=['routing_run.py','routing_experiment.py','routing_analysis.py','routing_inputs.json','observation_worker.py',
           'observation_timing.py','observation_contract.py','observation_scheduler.py','observation.json',
           'model_qualify.py','model_observer.py','native_model.py','prepare_runtime.py','diagnose.py',
           'model-qualification.json','model-pin.json','runtime-lock.txt']
    (out/'SOURCE_SHA256SUMS').write_text(''.join(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in files))
    for n in files:shutil.copyfile(ROOT/n,out/'source'/n)
    command=[prep['python'],'-u','-B',str(ROOT/'routing_experiment.py'),'--output',str(out),'--model',prep['model_path'],
             '--gpu-ids',json.dumps(prep['gpu_ids']),'--stage',a.stage]
    if a.stage=='compare':command+=['--freeze',str(Path(a.freeze).resolve()),'--freeze-sha256',a.freeze_sha256]
    print('Running fixed routing '+a.stage+' stage …',flush=True)
    receipt=runtime.run('routing-'+a.stage,command,900)
    for stream in ('stdout','stderr'):(out/'logs'/('routing.'+stream)).write_text(model_qualify.safe_log(receipt.pop(stream)))
    (out/'command.json').write_text(json.dumps(receipt,indent=2)+'\n')
    result=json.loads((out/'routing-result.json').read_text()) if (out/'routing-result.json').is_file() else dict(status='BLOCKED',reason='Native stage produced no result; inspect full logs')
    if receipt['exit_code']!=0:result.update(status='BLOCKED',process_exit_code=receipt['exit_code'])
    (out/'status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    report=dict(status=result['status'],stage=a.stage,**runtime.seal(out))
    for key in ('freeze_path','freeze_sha256'):
        if key in result:report[key]=result[key]
    print('EVIDENCE_JSON:'+json.dumps(report))


if __name__=='__main__':main()
