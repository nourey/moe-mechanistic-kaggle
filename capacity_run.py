#!/usr/bin/env python3
"""Portable single-host physical capacity experiment; no cluster or GPU allocation API."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid

from capacity_contract import ROOT, SOURCE, read, atomic, digest, source_hashes, freeze, validate_workload, topology, route_plan, check_devices, validate_slo, memory_floor,physical_gpu_binding
from capacity_analysis import merge_group_records, summarize,contrasts
from observation_contract import clock_receipt
import prepare_runtime as runtime
import model_qualify


def runtime_info():
    import importlib.metadata as m
    import platform,re
    packages={re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in m.distributions() if d.metadata.get('Name')}
    memory={}
    for p in ('/proc/meminfo','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'):
        if Path(p).is_file():memory[p]=Path(p).read_text()
    return dict(python=list(sys.version_info[:3]),platform=sys.platform,machine=platform.machine(),packages=packages,
                host_memory=memory,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))


def cuda_info():
    import torch
    if not torch.cuda.is_available():raise ValueError('No native CUDA runtime; CPU fallback forbidden')
    rows=[]
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        free,total=torch.cuda.mem_get_info(i)
        rows.append(dict(ordinal=i,uuid=str(p.uuid),name=p.name,total_memory=p.total_memory,free_memory=free,compute_capability=[p.major,p.minor]))
    return dict(torch=torch.__version__,cuda_build=torch.version.cuda,devices=rows)


def checked(out,label,argv,timeout=60):
    receipt=runtime.run(label,[str(x) for x in argv],timeout)
    atomic(out/(label+'-command.json'),receipt)
    if receipt['exit_code']!=0:raise RuntimeError(label+' failed; complete stdout/stderr retained')
    return receipt['stdout']


def preflight(out,python,model,install_report,device_indices,arms):
    config=read(ROOT/'capacity_protocol.json')
    report=read(install_report);locked=model_qualify.lock_entries((ROOT/'runtime-lock.txt').read_text())
    model_qualify.verify_report(report,locked);atomic(out/'pip-install-report.json',report)
    info=json.loads(checked(out,'runtime-info',[python,ROOT/'capacity_run.py','runtime-info']))
    if info['platform']!='linux' or info['machine'] not in ('x86_64','AMD64') or info['python'][:2]!=[3,13]:
        raise ValueError('Pinned cohort requires Linux x86_64 CPython 3.13; no substitute runtime')
    for name,(version,_) in locked.items():
        if info['packages'].get(name)!=version:raise ValueError('Runtime package drift: '+name)
    cuda=json.loads(checked(out,'cuda-info',[python,ROOT/'capacity_run.py','cuda-info']))
    if len(set(device_indices))!=len(device_indices) or any(i<0 or i>=len(cuda['devices']) for i in device_indices):
        raise ValueError('Invalid/duplicate CUDA-visible device selection')
    devices=[cuda['devices'][i] for i in device_indices]
    nvml=checked(out,'physical-gpus',['nvidia-smi','--query-gpu=uuid,pci.bus_id,mig.mode.current','--format=csv,noheader,nounits'],20)
    devices=physical_gpu_binding(devices,nvml)
    checks={arm:check_devices(arm,devices) for arm in arms}
    selected=','.join(d['uuid'] if d['uuid'].startswith('GPU-') else 'GPU-'+d['uuid'] for d in devices)
    apps=checked(out,'gpu-compute-apps',['nvidia-smi','-i',selected,'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],20)
    if any(line.strip() for line in apps.splitlines()):raise ValueError('Selected GPUs have existing compute processes; not modified or killed')
    try:checked(out,'nvidia-smi',['nvidia-smi','-i',selected,'-q'],20)
    except Exception as error:atomic(out/'nvidia-smi-unavailable.json',dict(reason=str(error),note='CUDA UUID/memory observations remain mandatory'))
    pin=read(ROOT/'model-pin.json')
    model_receipts=[model_qualify.verify_model_file(Path(model)/s['path'],s) for s in pin['files']]
    atomic(out/'model-receipts.json',dict(model_id=pin['model_id'],revision=pin['revision'],files=model_receipts))
    value=dict(runtime=info,cuda=cuda,selected_devices=devices,checks=checks,
               memory_note='Necessary FP16 tensor floor only; native load/workspace/KV fit is not certified',
               image_digest='Not inferred: native venv route uses locked wheel report; cluster image must be pinned by its owner')
    atomic(out/'PREFLIGHT.json',value)
    return devices


def reserve_check(ports):
    if len(ports)!=len(set(ports)):raise ValueError('Group RPC ports must differ')
    held=[]
    try:
        for port in ports:
            s=socket.socket();s.bind(('127.0.0.1',port));held.append(s)
    finally:
        for s in held:s.close()


def clean_env(gpu_ids):
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('PIP_') or key in ('PYTHONPATH','PYTHONHOME','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','KAGGLE_KEY','GH_TOKEN','GITHUB_TOKEN','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_SESSION_TOKEN'):
            env.pop(key)
    # Only UUIDs discovered inside the inherited CUDA-visible allocation may be selected.
    env.update(CUDA_VISIBLE_DEVICES=','.join(x if x.startswith('GPU-') else 'GPU-'+x for x in gpu_ids),
               PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return env


def wait_files(handles,filename,deadline,control):
    while not all((p/filename).exists() for _,p,_ in handles):
        failed=[str(p) for proc,p,_ in handles if proc.poll() is not None and not (p/filename).exists()]
        if failed:raise RuntimeError('Group exited before '+filename+': '+','.join(failed))
        if time.monotonic()>deadline:raise TimeoutError(filename+' deadline')
        time.sleep(.1)


def owned_group_exists(pgid):
    try:os.killpg(pgid,0);return True
    except ProcessLookupError:return False


def terminate_owned_group(pgid):
    """Only PGIDs created by this launcher's start_new_session=True children."""
    if not owned_group_exists(pgid):return False
    try:os.killpg(pgid,signal.SIGTERM)
    except ProcessLookupError:return False
    deadline=time.monotonic()+5
    while owned_group_exists(pgid) and time.monotonic()<deadline:time.sleep(.1)
    if owned_group_exists(pgid):
        try:os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError:pass
    return True


def launch_trial(out,arm,workload,mode,python,model,devices,slo,spawn=subprocess.Popen):
    out=Path(out).absolute();model=Path(model).absolute()
    config=read(ROOT/'capacity_protocol.json');groups=topology(arm);handles=[];initial=source_hashes()
    control=out/'control';control.mkdir();deadline=time.monotonic()+config['startup_timeout_seconds']
    atomic(out/'ROUTING_PLAN.json',route_plan(arm,workload))
    try:
        reserve_check(config['group_rpc_ports'][:len(groups)])
        for i,g in enumerate(groups):
            p=out/g['group'];p.mkdir()
            gpu_ids=[devices[d]['uuid'] for d in g['devices']]
            spec=dict(arm=arm,group=g['group'],dp=g['dp'],mode=mode,gpu_ids=gpu_ids,group_incarnation=uuid.uuid4().hex,
                      rpc_port=config['group_rpc_ports'][i],model=str(model),output=str(p),control=str(control),workload=workload,slo=slo)
            atomic(p/'spec.json',spec)
            log=(p/'native.log').open('w')
            proc=spawn([str(python),'-u','-B',str(ROOT/'capacity_group.py'),'--spec',str(p/'spec.json')],
                       env=clean_env(gpu_ids),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(ROOT))
            handles.append((proc,p,log))
        wait_files(handles,'READY.json',deadline,control)
        clocks=[read(p/'READY.json')['clock_identity'] for _,p,_ in handles]
        from load_analysis import common_clock
        common_clock([clock_receipt()]+clocks)
        ready=[read(p/'READY.json') for _,p,_ in handles]
        for r,(_,p,_) in zip(ready,handles):
            s=read(p/'spec.json')
            if any(r[k]!=s[k] for k in ('group','group_incarnation','gpu_ids')):raise ValueError('READY identity differs from planned group')
        uuids=[u for r in ready for u in r['gpu_ids']]
        if len(uuids)!=len(set(uuids)):raise ValueError('Group GPU overlap at native readiness')
        starts={}
        for stage in ('primer','main'):
            t0=time.monotonic_ns()+2_000_000_000;starts[stage]=t0
            atomic(control/f'GO_{stage}.json',dict(t0_ns=t0,clock_identity=clock_receipt(),stage=stage))
            wait_files(handles,stage.upper()+'_DONE.json',time.monotonic()+2+workload['duration_ns']/1e9+config['drain_timeout_seconds']+45,control)
        wait_files(handles,'DONE.json',time.monotonic()+config['cleanup_timeout_seconds'],control)
        results=[]
        for proc,p,_ in handles:
            code=proc.wait(timeout=10);value=read(p/'DONE.json')
            if code!=0 or value['status']!='PASS_GROUP_COLLECTION' or not value.get('shutdown_returned'):
                raise ValueError('Native group/cleanup failed: '+p.name)
            results.append(read(p/'main-analysis.json'))
        plan=route_plan(arm,workload);records=merge_group_records(plan,results)
        if source_hashes()!=initial:raise ValueError('Source drift during trial')
        result=dict(status='PASS_PHYSICAL_COLLECTION',arm=arm,mode=mode,workload_sha256=workload['sha256'],
                    records=records,summary=summarize(records,workload['duration_ns'],starts['main'],slo),
                    groups={g['group']:r for g,r in zip(groups,results)},
                    same_resident_control='C0 and C1 keep both EP2 groups loaded throughout; only frozen new-request assignments change',
                    primary_performance='reference only' if mode=='diagnostic' else 'one independent reference repetition',
                    limitations='Direct AsyncLLM harness, combined prefill+decode. No Real EPP control/authority, no pure EP-only or steady-state claim.')
        atomic(out/'TRIAL.json',result);return result
    except Exception:
        atomic(out/'TRIAL_FAILURE.json',dict(error=traceback.format_exc()));raise
    finally:
        atomic(control/'STOP.json',dict(reason='Trial finished or failed',publication_ns=time.monotonic_ns()))
        forced=False
        for proc,p,log in handles:
            group_forced=False
            if proc.poll() is None:
                try:proc.wait(timeout=config['cleanup_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    terminate_owned_group(proc.pid);group_forced=True
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
            # A failed frontend may have exited while its native workers remain.
            if terminate_owned_group(proc.pid):group_forced=True
            if group_forced:atomic(p/'watchdog.json',dict(status='FORCED_PROCESS_GROUP_TERMINATION',pid=proc.pid,cleanup='UNVERIFIED; do not run another trial'))
            forced |= group_forced
            log.close()
        if forced:raise RuntimeError('Forced cleanup of owned process session; trial invalid and no later trial allowed')


def execute(args):
    # Preserve venv interpreter symlinks: resolving them would silently run base Python.
    candidate=shutil.which(args.python) if os.sep not in args.python else os.path.abspath(os.path.expanduser(args.python))
    if not candidate:raise ValueError('Explicit runtime Python executable not found')
    candidate=os.path.abspath(candidate)
    args.python=candidate;args.model=str(Path(args.model).expanduser().absolute())
    args.output=str(Path(args.output).expanduser().absolute())
    args.workload=str(Path(args.workload).expanduser().absolute());args.install_report=str(Path(args.install_report).expanduser().absolute())
    cfg=read(ROOT/'capacity_protocol.json');workload=read(args.workload);validate_workload(workload)
    slo=read(args.slo) if args.slo else cfg['slo'];validate_slo(slo)
    arms=args.arms.split(',');repetitions=args.repetitions
    if len(set(arms))!=len(arms) or not arms or any(a not in cfg['topology'] for a in arms) or not 1<=repetitions<=5:
        raise ValueError('Explicit supported arms and 1..5 repetitions required')
    if args.mode=='reference' and not workload['smoke'] and repetitions<2:raise ValueError('Physical performance comparison requires independent repetitions >=2')
    out=Path(args.output)/('moe-capacity-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True);(out/'source').mkdir()
    result=dict(status='BLOCKED',trials=[],scope=cfg['scope']);started=time.monotonic()
    try:
        dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True)
        if dirty.strip():raise ValueError('Research source must be a clean reviewed commit')
        commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        initial_source=source_hashes()
        atomic(out/'SOURCE.json',dict(commit=commit,hashes=initial_source))
        for name in SOURCE:
            dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
        atomic(out/'WORKLOAD.json',workload);atomic(out/'SLO.json',slo)
        devices=preflight(out,args.python,args.model,args.install_report,[int(x) for x in args.devices.split(',')],arms)
        for repeat in range(repetitions):
            # Reverse arm order on alternate repetitions; keep all results, no favorable selection.
            for arm in (arms if repeat%2==0 else list(reversed(arms))):
                if source_hashes()!=initial_source:raise ValueError('Source/protocol changed between trials')
                remaining=args.max_wall_seconds-(time.monotonic()-started)
                needed=cfg['startup_timeout_seconds']+2*(workload['duration_ns']/1e9+cfg['drain_timeout_seconds']+50)+cfg['cleanup_timeout_seconds']
                if remaining<needed:raise TimeoutError('Insufficient whole-trial wall budget; no truncated next trial')
                p=out/f'rep{repeat+1}-{arm}-{args.mode}';p.mkdir()
                if shutil.disk_usage(out).free<2*1024**3:raise RuntimeError('Less than 2 GiB evidence storage available; no next trial')
                checked(p,'gpu-before',['nvidia-smi','-i',','.join(devices[d]['uuid'] if devices[d]['uuid'].startswith('GPU-') else 'GPU-'+devices[d]['uuid'] for g in topology(arm) for d in g['devices']),'-q'],20)
                print('Running '+p.name,flush=True)
                trial=launch_trial(p,arm,workload,args.mode,args.python,args.model,devices,slo)
                if source_hashes()!=initial_source:raise ValueError('Source/protocol changed after trial')
                result['trials'].append(dict(path=p.name,arm=arm,repeat=repeat+1,mode=args.mode,status=trial['status'],summary=trial['summary'],workload_sha256=workload['sha256']))
                atomic(out/'status.json',result)
        atomic(out/'COMPARISON.json',contrasts(result['trials']))
        result['status']='PASS_COLLECTION_NOT_CAUSAL_VALIDATION'
    except Exception:
        result.update(status='PARTIAL' if result['trials'] else 'BLOCKED',error=traceback.format_exc());traceback.print_exc()
    atomic(out/'status.json',result)
    return dict(status=result['status'],**runtime.seal(out))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('runtime-info');sub.add_parser('cuda-info')
    f=sub.add_parser('freeze');f.add_argument('--output',required=True);f.add_argument('--intensity',type=float,default=1)
    f.add_argument('--family',choices=('narrative','identifiers','mixed'),default='narrative');f.add_argument('--smoke',action='store_true')
    sub.add_parser('plan')
    p=sub.add_parser('run');p.add_argument('--workload',required=True);p.add_argument('--python',required=True)
    p.add_argument('--model',required=True);p.add_argument('--install-report',required=True);p.add_argument('--devices',required=True)
    p.add_argument('--arms',default='B,C0,C1');p.add_argument('--mode',choices=('reference','diagnostic'),default='reference')
    p.add_argument('--repetitions',type=int,default=2);p.add_argument('--slo');p.add_argument('--output',required=True)
    p.add_argument('--max-wall-seconds',type=int,default=14400)
    a=parser.parse_args()
    if a.command=='runtime-info':print(json.dumps(runtime_info()))
    elif a.command=='cuda-info':print(json.dumps(cuda_info()))
    elif a.command=='freeze':
        path=Path(a.output)
        if path.exists():raise FileExistsError('Do not overwrite a frozen workload')
        path.parent.mkdir(parents=True,exist_ok=True);atomic(path,freeze(a.intensity,a.family,a.smoke));print(path)
    elif a.command=='plan':print(json.dumps(dict(topologies={x:topology(x) for x in ('A','B','C0','C1')},
        weight_floor_bytes_per_gpu=dict(EP1=memory_floor(1),EP2=memory_floor(2)),physical_validation='NOT_EXECUTED'),indent=2))
    else:print('EVIDENCE_JSON:'+json.dumps(execute(a)))


if __name__=='__main__':main()
