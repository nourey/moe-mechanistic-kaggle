#!/usr/bin/env python3
"""One isolated native serving group; controller supplies only frozen workload/barriers."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from capacity_contract import ROOT, read, atomic, source_hashes, engine_arguments, route_plan, translate, validate_residency,validate_membership
from capacity_analysis import request_metrics, summarize, native_accounting, pre_boundary
from load_analysis import common_clock, validate_stage
from load_experiment import owned_process_receipts, scheduled_offers, read_lines
from observation_contract import Writer, clock_receipt


async def pause_until(target, stop):
    while time.monotonic()<target:
        if stop.exists():raise RuntimeError('CONTROLLER_STOP')
        await asyncio.sleep(min(.1,max(0,target-time.monotonic())))


async def barrier(path, stop, timeout):
    deadline=time.monotonic()+timeout
    while not path.exists():
        if stop.exists():raise RuntimeError('CONTROLLER_STOP')
        if time.monotonic()>deadline:raise TimeoutError('Barrier deadline: '+path.name)
        await asyncio.sleep(.1)
    return read(path)


async def run(spec_path):
    spec=read(spec_path);out=Path(spec['output']);control=Path(spec['control']);stop=control/'STOP.json'
    cfg=read(ROOT/'capacity_protocol.json');workload=spec['workload'];group=spec['group'];dp=spec['dp'];mode=spec['mode']
    os.environ.update(MOE_OBSERVATION_ROOT=str(out),MOE_LOAD_MODE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
                      VLLM_ALLOW_INSECURE_SERIALIZATION='0',NCCL_DEBUG='INFO',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    engine=None;ledger=Writer(out,'client',-1);joins={};jobs=[];terminals=set();owned={};active=None
    result=dict(status='BLOCKED',group=group,group_incarnation=spec['group_incarnation'],stages=[],
                scope='Research process namespace, not a Kubernetes or EngineCore attestation nonce')
    initial_source=source_hashes();clock=clock_receipt()
    try:
        if not clock['verified']:raise ValueError('Verified Linux host monotonic clock required')
        import vllm
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        expected={**read(ROOT/'observation.json')['native_source_sha256'],**read(ROOT/'routing_inputs.json')['native_timing_sources'],
                  **read(ROOT/'load_protocol.json')['native_extra_sources']}
        package=Path(vllm.__file__).parent.parent
        actual={n:hashlib.sha256((package/n).read_bytes()).hexdigest() for n in expected}
        atomic(out/'native-sources.json',actual)
        if actual!=expected:raise ValueError('Native source drift')
        kwargs=engine_arguments(dp,mode,spec['model'],spec['rpc_port'])
        baseline=AsyncEngineArgs(**kwargs).create_engine_config();kind=baseline.scheduler_config.get_scheduler_cls().__name__
        if kind not in ('Scheduler','AsyncScheduler'):raise ValueError('Unsupported native scheduler')
        kwargs.update(async_scheduling=baseline.scheduler_config.async_scheduling,
                      scheduler_cls='observation_scheduler.'+('Observed' if mode=='diagnostic' else 'OwnershipOnly')+kind)
        engine=AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs));await engine.check_health()
        pc=engine.vllm_config.parallel_config
        if (pc.tensor_parallel_size,pc.pipeline_parallel_size,pc.data_parallel_size)!=(1,1,dp) or pc.data_parallel_external_lb:
            raise ValueError('Unexpected DP/TP/PP or external LB topology')
        atomic(out/'configuration.json',dict(engine_args=kwargs,clock_identity=clock,source=initial_source,
               parallel=dict(dp=pc.data_parallel_size,local_dp=pc.data_parallel_size_local,tp=pc.tensor_parallel_size,
                             pp=pc.pipeline_parallel_size,rpc_port=pc.data_parallel_rpc_port,
                             dp_master_ip=pc.data_parallel_master_ip,dp_master_port=pc.data_parallel_master_port),
               rendezvous_note='Native single-node DP dynamically chooses init ports; numeric configuration alone is not a port-isolation proof'))
        await engine.collective_rpc('research_rank_state',timeout=30,args=(str(out),'loaded'))
        loaded=[read(out/f'loaded-dp{r}.json') for r in range(dp)];validate_residency(loaded,spec['gpu_ids'],dp)
        await engine.collective_rpc('capacity_identity',timeout=30,args=(str(out),'loaded',spec['group_incarnation']))
        validate_membership([read(out/f'identity-loaded-dp{r}.json') for r in range(dp)],loaded,spec['group_incarnation'])
        core_init=[e for r in range(dp) for e in read_lines(out/f'core-dp{r}.jsonl') if e['event']=='core_initialized']
        workers=owned_process_receipts([w['pid'] for w in loaded])
        owned=owned_process_receipts([e['pid'] for e in core_init]+[w['pid'] for w in loaded]);atomic(out/'owned-before.json',owned)
        for rank,w in enumerate(loaded):
            c=next(e for e in core_init if e['dp_rank']==rank)
            if w['pid']!=c['pid'] and workers[str(w['pid'])].get('parent_pid')!=c['pid']:raise ValueError('Worker/Core incarnation join failed')
        common_clock([clock]+[e['clock_identity'] for e in core_init])
        moe=mode=='diagnostic' and dp==2
        if moe:await engine.collective_rpc('load_install',timeout=30,args=(str(out),))
        atomic(out/'READY.json',dict(group=group,pid=os.getpid(),group_incarnation=spec['group_incarnation'],clock_identity=clock,
                                     gpu_ids=spec['gpu_ids'],loaded_workers=loaded))
        assign=engine.input_processor.assign_request_id
        def identified(request):
            value=assign(request)
            joins[request.external_req_id]=dict(internal_request_id=request.request_id,prompt_token_ids=list(request.prompt_token_ids),
                                                observed_ns=time.monotonic_ns(),source='Native assign_request_id return')
            atomic(out/'request-identity-joins.json',joins);return value
        engine.input_processor.assign_request_id=identified
        inputs=read(ROOT/'load_protocol.json')['inputs'];options=read(ROOT/'load_protocol.json')['observer']
        for stage in ('primer','main'):
            go=await barrier(control/f'GO_{stage}.json',stop,cfg['startup_timeout_seconds']+cfg['pass_timeout_seconds'])
            common_clock([clock,go['clock_identity']]);t0=go['t0_ns']
            plan=route_plan(spec['arm'],workload,stage,group if stage=='primer' else None)
            if stage=='primer':
                for r in plan['arrivals']:r['request_id']=group+'-'+r['request_id']
            rows=translate(plan,t0,group);atomic(out/(stage+'-plan.json'),dict(plan=plan,local_arrivals=rows,t0_ns=t0))
            for row in rows:ledger.emit('planned',**row)
            if moe:await engine.collective_rpc('load_begin',timeout=30,args=(stage,options))
            active=stage;pending_count=0;failed=False;stage_jobs=[]
            async def consume(row):
                nonlocal pending_count,failed
                pending_count+=1;ledger.emit('submitted',**row)
                try:
                    terminal=None
                    async for output in engine.generate(dict(prompt_token_ids=inputs[row['family']]),
                        SamplingParams(temperature=0,max_tokens=8,seed=1525),row['request_id'],data_parallel_rank=row['owner']):
                        if len(output.outputs)!=1:raise ValueError('Unexpected output sequence count')
                        terminal=output
                        ledger.emit('output_chunk',stage=stage,request_id=row['request_id'],group=group,
                                    token_ids=list(output.outputs[0].token_ids),finished=output.finished)
                    if terminal is None or not terminal.finished or terminal.request_id!=row['request_id']:raise ValueError('Missing native terminal')
                    final=terminal.outputs[0]
                    ledger.emit('terminal',stage=stage,request_id=row['request_id'],group=group,
                                token_ids=list(final.token_ids),text=final.text,finish_reason=final.finish_reason)
                    terminals.add(row['request_id'])
                except asyncio.CancelledError:
                    ledger.emit('censored',stage=stage,request_id=row['request_id'],group=group,reason='ABORT/CANCEL');raise
                except Exception:
                    failed=True;ledger.emit('request_error',stage=stage,request_id=row['request_id'],group=group,reason=traceback.format_exc())
                finally:pending_count-=1
            def launch(row):
                task=asyncio.create_task(consume(row),name=row['request_id']);jobs.append(task);stage_jobs.append(task);return task
            end=t0/1e9+workload['duration_ns']/1e9;deadline=end+cfg['drain_timeout_seconds']
            _,halt=await scheduled_offers(rows,launch,lambda:pending_count,ledger.emit,cfg['max_pending_per_group'],end,
                                         lambda:failed or stop.exists())
            if halt is None:await pause_until(end,stop)
            ledger.emit('offers_stopped',stage=stage,group=group,reason=halt or 'END_FROZEN_WINDOW')
            while any(not t.done() for t in stage_jobs) and time.monotonic()<deadline and not stop.exists():await asyncio.sleep(.05)
            unfinished=[t for t in stage_jobs if not t.done()]
            if unfinished:
                ids=[joins[t.get_name()]['internal_request_id'] for t in unfinished if t.get_name() in joins]
                for t in unfinished:ledger.emit('censored',stage=stage,request_id=t.get_name(),group=group,reason='FOLLOWUP_EXPIRED_OR_STOP')
                if ids:await asyncio.wait_for(engine.abort(ids,internal=True),5)
                for t in unfinished:t.cancel()
                await asyncio.wait(unfinished,timeout=3)
            await asyncio.wait_for(engine.wait_for_requests_to_drain(drain_timeout=15),18)
            if moe:await engine.collective_rpc('load_end',timeout=45,args=(stage,))
            active=None
            events=read_lines(out/'client-dp-1.jsonl');core=[e for r in range(dp) for e in read_lines(out/f'core-dp{r}.jsonl')]
            records=request_metrics(rows,events,t0,int(end*1e9),spec['slo'],int(deadline*1e9));complete=all(r['status']=='NORMAL' for r in records)
            accepted={e['request_id']:e for e in core if e['event']=='request_accepted'}
            for r in records:
                r['group_incarnation']=spec['group_incarnation']
                internal=joins.get(r['request_id'],{}).get('internal_request_id');e=accepted.get(internal)
                r.update(internal_request_id=internal,actual_owner=None if e is None else e['dp_rank'],
                         native_core_pid=None if e is None else e['pid'],native_accept_return_ns=None if e is None else e.get('native_accept_return_ns'))
            if complete:
                account=native_accounting(stage,rows,core,joins,dp,inputs,mode=='diagnostic')
                if moe and rows:
                    ws={r:read(out/f'{stage}-load-worker-dp{r}.json') for r in range(dp)}
                    full=validate_stage(stage,core,ws,events,joins,inputs,8,require_multi=workload['smoke'] and len(rows)>=4)
                    atomic(out/(stage+'-native-worker-analysis.json'),full)
            else:account=dict(status='PARTIAL_CENSORED',native_conservation='UNVERIFIED_AFTER_CENSOR')
            drift=[e['request_id'] for e in events if e.get('stage')==stage and e['event']=='submitted'
                   and e['binding_regime']=='pre' and e['monotonic_ns']>=t0+plan['boundary_offset_ns']]
            stage_result=dict(stage=stage,records=records,summary=summarize(records,workload['duration_ns'],t0,spec['slo']),
                  native=account,binding_boundary_drift=drift,pre_boundary=pre_boundary(core,t0+plan['boundary_offset_ns'],dp),
                  MoE_coverage='Existing paired EP2 worker/routing/padding records' if moe else 'UNKNOWN_EP1_OR_REFERENCE; EP1 collectives NOT_APPLICABLE',
                  complete=complete and halt is None and not drift)
            atomic(out/(stage+'-analysis.json'),stage_result);result['stages'].append(stage_result)
            if not stage_result['complete']:raise RuntimeError('Incomplete/censored/boundary-drift cohort; no next pass')
            atomic(out/(stage.upper()+'_DONE.json'),dict(group=group,status='PASS',clock_identity=clock))
        await engine.collective_rpc('research_rank_state',timeout=30,args=(str(out),'after'))
        after=[read(out/f'after-dp{r}.json') for r in range(dp)];validate_residency(after,spec['gpu_ids'],dp)
        await engine.collective_rpc('capacity_identity',timeout=30,args=(str(out),'after',spec['group_incarnation']))
        validate_membership([read(out/f'identity-after-dp{r}.json') for r in range(dp)],after,spec['group_incarnation'])
        for a,b in zip(loaded,after):
            if any(a[k]!=b[k] for k in ('pid','gpu_uuid','expert_layers','kv_tensors','kv_capacity_blocks')):raise ValueError('Rank/resource incarnation changed')
        if source_hashes()!=initial_source:raise ValueError('Source changed during native execution')
        result['status']='PASS_GROUP_COLLECTION'
    except Exception:
        result.update(status='PARTIAL' if result['stages'] else 'BLOCKED',error=traceback.format_exc());traceback.print_exc()
    finally:
        if engine is not None:
            try:
                ids=[v['internal_request_id'] for k,v in joins.items() if k not in terminals]
                if ids:await asyncio.wait_for(engine.abort(ids,internal=True),5)
                await asyncio.wait_for(engine.wait_for_requests_to_drain(drain_timeout=10),12)
                if active and dp==2 and mode=='diagnostic':await asyncio.wait_for(engine.collective_rpc('load_end',timeout=10,args=(active,)),12)
            except Exception:result['failure_export_error']=traceback.format_exc()
            for task in jobs:
                if not task.done():task.cancel()
            try:engine.shutdown(timeout=20);result['shutdown_returned']=True
            except Exception:result.update(status='PARTIAL_CLEANUP',shutdown_error=traceback.format_exc())
            if owned:
                after=owned_process_receipts([int(p) for p in owned]);atomic(out/'owned-after.json',after)
                if any(p.get('present') is not False for p in after.values()):result.update(status='PARTIAL_CLEANUP',cleanup=after)
        ledger.stream.close();atomic(out/'request-identity-joins.json',joins);atomic(out/'DONE.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args()
    value=asyncio.run(run(args.spec));raise SystemExit(0 if value['status']=='PASS_GROUP_COLLECTION' else 1)
