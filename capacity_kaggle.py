#!/usr/bin/env python3
"""Manual two-T4 B smoke only. Does not emulate or launch four-GPU C."""
import datetime as dt
import json
from pathlib import Path
import shutil
import uuid
import zipfile

from capacity_contract import ROOT,atomic,freeze
import model_qualify
import prepare_runtime as runtime


def main():
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Manual Kaggle session only')
    prep=model_qualify.qualify('/kaggle/working',prepare_only=True)
    if prep.get('runtime_and_checkpoint')!='READY':
        print('EVIDENCE_JSON:'+json.dumps(dict(status='BLOCKED',archive=prep['archive'])));return
    out=Path('/kaggle/working')/('moe-capacity-smoke-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    out.mkdir();shutil.copyfile(prep['archive'],out/'runtime-preparation.zip')
    with zipfile.ZipFile(prep['archive']) as z:(out/'pip-install-report.json').write_bytes(z.read('pip-install-report.json'))
    atomic(out/'workload.json',freeze(smoke=True))
    receipt=runtime.run('capacity-smoke',[prep['python'],'-u','-B',str(ROOT/'capacity_run.py'),'run',
        '--workload',str(out/'workload.json'),'--python',prep['python'],'--model',prep['model_path'],
        '--install-report',str(out/'pip-install-report.json'),'--devices','0,1','--arms','B','--mode','diagnostic',
        '--repetitions','1','--output',str(out),'--max-wall-seconds','1500'],1500)
    atomic(out/'command.json',receipt)
    values=[json.loads(s.split(':',1)[1]) for s in receipt['stdout'].splitlines() if s.startswith('EVIDENCE_JSON:')]
    result=values[-1] if values else dict(status='BLOCKED',reason='No final archive receipt; inspect command stdout/stderr')
    atomic(out/'status.json',result)
    print('EVIDENCE_JSON:'+json.dumps(dict(status=result['status'],**runtime.seal(out))))


if __name__=='__main__':main()
