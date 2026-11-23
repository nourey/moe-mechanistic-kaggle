#!/usr/bin/env python3
"""First bounded load trajectory, manually invoked on Kaggle. Never calibrates BLIS."""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

from observation_contract import Writer, clock_receipt
from load_analysis import common_clock, make_plan, validate_stage, phase_report, bounded_trajectory
from routing_experiment import telemetry

ROOT=Path(__file__).resolve().parent
CODE=['load_run.py','load_experiment.py','load_analysis.py','load_protocol.json',
      'observation_worker.py','observation_scheduler.py','observation_contract.py','observation_timing.py',
      'model-qualification.json','model-pin.json','runtime-lock.txt','observation.json','routing_inputs.json',
      'routing_experiment.py','routing_analysis.py','model_qualify.py','prepare_runtime.py','diagnose.py',
      'model_observer.py','native_model.py']

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def hashes():return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in CODE}
def read_lines(path):return [json.loads(s) for s in path.read_text().splitlines()]

def owned_process_receipts(pids):
    """Only previously identified core/worker PIDs; never enumerate/kill other jobs."""
    result={}
    for pid in sorted(set(pids)):
        try:
            raw=Path(f'/proc/{pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
            result[str(pid)]=dict(present=True,state=fields[0],start_ticks=fields[19])
        except FileNotFoundError:result[str(pid)]=dict(present=False)
        except OSError as error:result[str(pid)]=dict(present=None,error=repr(error))
    return result


async def scheduled_offers(plan,launch,pending_count,emit,max_pending,deadline,failed=lambda:False):
    """Arrival calendar never awaits a response. Safety stops are explicit unsent offers."""
    jobs=[];stop=None
    for item in plan:
        if stop is not None:
            emit('unsent',**item,reason=stop)
            continue
        delay=item['due_ns']/1e9-time.monotonic()
        if delay>0:await asyncio.sleep(delay)
        if time.monotonic()>=deadline:stop='WORK_DEADLINE'
        elif failed():stop='REQUEST_ERROR'
        elif pending_count()>=max_pending:stop='PENDING_SAFETY_LIMIT'
        if stop is not None:
            emit('unsent',**item,reason=stop)
            continue
        jobs.append(launch(item))
        # Let newly created tasks actually submit; do not wait for their responses.
        await asyncio.sleep(0)
    return jobs,stop


async def run(out,model,gpu_ids):
    os.environ.update(MOE_OBSERVATION_ROOT=str(out),MOE_LOAD_MODE='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_ALLOW_INSECURE_SERIALIZATION='0',NCCL_DEBUG='INFO')
    protocol=json.loads((ROOT/'load_protocol.json').read_text())
    result=dict(status='BLOCKED',stages=[],claims='One finite-horizon mixed prefill/decode EP2 observation; no SLO/BLIS/D2 claim',
                overhead='UNKNOWN under load; no replicated observer-on/off performance trajectory')
    engine=None;ledger=Writer(out,'client',-1);joins={};jobs_all=[];active_episode=None;owned_before={}
    clock=clock_receipt()
    if not clock['verified']:
        result['reason']='Linux same-host monotonic provenance unavailable';ledger.stream.close();write(out/'load-result.json',result);return result
    try:
        import vllm
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from native_model import validate_residency
        expected={**json.loads((ROOT/'observation.json').read_text())['native_source_sha256'],
                  **json.loads((ROOT/'routing_inputs.json').read_text())['native_timing_sources'],
                  **protocol['native_extra_sources']}
        package=Path(vllm.__file__).parent.parent
        native_hashes={n:hashlib.sha256((package/n).read_bytes()).hexdigest() for n in expected}
        write(out/'native-sources.json',native_hashes)
        if native_hashes!=expected:raise ValueError('Pinned native source identity drift')
        kwargs=dict(json.loads((ROOT/'model-qualification.json').read_text())['engine_args']);kwargs['model']=model
        baseline=AsyncEngineArgs(**kwargs).create_engine_config()
        kind=baseline.scheduler_config.get_scheduler_cls().__name__
        if kind not in ('Scheduler','AsyncScheduler'):raise ValueError('Unsupported baseline scheduler')
        kwargs.update(async_scheduling=baseline.scheduler_config.async_scheduling,
            scheduler_cls='observation_scheduler.Observed'+kind,worker_extension_cls='observation_worker.ObservationWorker')
        engine=AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
        if (engine.vllm_config.scheduler_config.async_scheduling!=baseline.scheduler_config.async_scheduling
            or engine.vllm_config.max_concurrent_batches!=baseline.max_concurrent_batches
            or engine.vllm_config.cache_config.enable_prefix_caching):raise ValueError('Baseline serving policy changed')
        initial_hashes=hashes()
        write(out/'configuration.json',dict(engine_args=kwargs,source=initial_hashes,clock_identity=clock,
            max_concurrent_batches=baseline.max_concurrent_batches,profiler=False,
            timestamp_meaning='Host offer/accept/application instants; CUDA intervals local to device, not global instants'))
        await engine.check_health()
        await engine.collective_rpc('research_rank_state',timeout=30,args=(str(out),'loaded'))
        loaded=[json.loads((out/f'loaded-dp{r}.json').read_text()) for r in (0,1)];validate_residency(loaded,gpu_ids)
        core_initial=[e for rank in (0,1) for e in read_lines(out/f'core-dp{rank}.jsonl') if e['event']=='core_initialized']
        owned_before=owned_process_receipts([e['pid'] for e in core_initial]+[w['pid'] for w in loaded])
        write(out/'owned-processes-before.json',owned_before)
        await engine.collective_rpc('load_install',timeout=30,args=(str(out),))
        for rank in (0,1):
            q=json.loads((out/f'timing-qualification-dp{rank}.json').read_text())
            if not q['usable'] or q['worker_pid']!=loaded[rank]['pid'] or q['gpu_uuid']!=loaded[rank]['gpu_uuid']:
                raise ValueError('Worker/timing sanity not qualified')
        assign=engine.input_processor.assign_request_id
        def observed_assign(request):
            value=assign(request)
            joins[request.external_req_id]=dict(external_request_id=request.external_req_id,
                internal_request_id=request.request_id,prompt_token_ids=list(request.prompt_token_ids),
                observed_ns=time.monotonic_ns(),source='Actual native assign_request_id return')
            write(out/'request-identity-joins.json',joins)
            return value
        engine.input_processor.assign_request_id=observed_assign
        work_start=time.monotonic();deadline=work_start+protocol['work_timeout_seconds']
        write(out/'work-budget.json',dict(start_monotonic_ns=int(work_start*1e9),deadline_monotonic_ns=int(deadline*1e9),
            scope='Includes mapping, range discovery, trajectory and drain; engine/runtime setup recorded separately'))
        actual_offers=0;completed=0;task_error=False;submitted_registry={};normal_terminal_ids=set()

        async def consume(item):
            nonlocal actual_offers,completed,task_error
            actual_offers+=1
            submitted_registry[item['request_id']]=dict(item)
            ledger.emit('submitted',**item)
            try:
                terminal=None
                async for output in engine.generate(dict(prompt_token_ids=protocol['inputs'][item['family']]),
                    SamplingParams(temperature=0,max_tokens=protocol['max_output_tokens'],seed=protocol['seed']),
                    item['request_id'],data_parallel_rank=item['owner']):
                    terminal=output
                    ledger.emit('output_chunk',stage=item['stage'],request_id=item['request_id'],
                        output_tokens=list(output.outputs[0].token_ids),finished=output.finished,
                        scope='Chunk receipt time, not a fabricated timestamp for every token')
                if terminal is None or not terminal.finished or terminal.request_id!=item['request_id'] or len(terminal.outputs)!=1:
                    raise ValueError('Missing/incorrect terminal')
                final=terminal.outputs[0]
                ledger.emit('terminal',stage=item['stage'],request_id=item['request_id'],token_ids=list(final.token_ids),
                    finish_reason=final.finish_reason,text=final.text)
                if final.finish_reason in ('length','stop') and 0<=len(final.token_ids)<=protocol['max_output_tokens']:
                    normal_terminal_ids.add(item['request_id'])
                else:task_error=True
            except asyncio.CancelledError:
                ledger.emit('client_cancelled',stage=item['stage'],request_id=item['request_id']);raise
            except Exception:
                task_error=True
                ledger.emit('request_error',stage=item['stage'],request_id=item['request_id'],traceback=traceback.format_exc())
            finally:completed+=1

        async def abort_tasks(tasks,stage_name):
            unfinished=[t for t in tasks if not t.done()]
            for task in unfinished:ledger.emit('censored_before_cleanup',stage=stage_name,request_id=task.get_name())
            suspects=[rid for rid,item in submitted_registry.items() if item['stage']==stage_name and rid not in normal_terminal_ids]
            for rid in suspects:ledger.emit('cleanup_abort_requested',stage=stage_name,request_id=rid)
            internal=[joins[rid]['internal_request_id'] for rid in suspects if rid in joins]
            if internal:await asyncio.wait_for(engine.abort(internal,internal=True),5)

        async def stage(name,build,options,require_multi=False):
            nonlocal task_error,active_episode
            task_error=False
            if time.monotonic()+protocol['drain_timeout_seconds']>=deadline:raise RuntimeError('Insufficient remaining work/drain budget')
            await engine.wait_for_requests_to_drain(drain_timeout=30)
            await engine.collective_rpc('load_begin',timeout=30,args=(name,options))
            active_episode=name
            start=time.monotonic()+.05
            plan,windows=build(start)
            write(out/(name+'-plan.json'),dict(arrivals=plan,windows=windows,options=options))
            for item in plan:ledger.emit('planned',**item)
            if actual_offers+len(plan)>protocol['max_offers']:
                for item in plan:ledger.emit('unsent',**item,reason='OFFER_PLAN_BUDGET')
                raise RuntimeError('SAFE_STOP_OFFER_PLAN_BUDGET; no truncated trajectory')
            def launch(item):
                task=asyncio.create_task(consume(item),name=item['request_id']);jobs_all.append(task);return task
            tasks,stop=await scheduled_offers(plan,launch,lambda:actual_offers-completed,ledger.emit,
                protocol['max_pending'],deadline-protocol['drain_timeout_seconds'],lambda:task_error)
            planned_end=max((w['end_ns']/1e9 for w in windows),default=start)
            if not tasks:raise RuntimeError('NO_ACTUAL_OFFERS; safety stop, planned ledger retained')
            if stop is None and time.monotonic()<planned_end:await asyncio.sleep(planned_end-time.monotonic())
            drain_start=ledger.emit('offers_stopped',stage=name,reason=stop or 'END_PLAN')['monotonic_ns']
            remaining=min(protocol['drain_timeout_seconds'],max(0,deadline-time.monotonic()))
            _,pending=await asyncio.wait(tasks,timeout=remaining)
            if pending:
                await abort_tasks(pending,name)
                _,pending=await asyncio.wait(pending,timeout=10)
                for task in pending:task.cancel()
                if pending:await asyncio.wait(pending,timeout=3)
                raise RuntimeError('FOLLOWUP_EXPIRED; censored records preserved, no further stage')
            await engine.wait_for_requests_to_drain(drain_timeout=max(1,int(min(30,deadline-time.monotonic()))))
            drain_end=time.monotonic_ns()
            await engine.collective_rpc('load_end',timeout=60,args=(name,))
            active_episode=None
            client=read_lines(out/'client-dp-1.jsonl')
            core=[e for rank in (0,1) for e in read_lines(out/f'core-dp{rank}.jsonl')]
            workers={rank:json.loads((out/f'{name}-load-worker-dp{rank}.json').read_text()) for rank in (0,1)}
            if any(workers[r]['worker_pid']!=loaded[r]['pid'] for r in (0,1)):raise ValueError('Worker incarnation changed')
            assessment=validate_stage(name,core,workers,client,joins,protocol['inputs'],protocol['max_output_tokens'],require_multi)
            write(out/(name+'-assessment.json'),assessment)
            phases=[dict(name=w['name'],**phase_report(core,client,name,w['start_ns'],w['end_ns'])) for w in windows]
            for phase in phases:
                phase['paired_diagnostic_batches']=sum(g['sampled'] and phase['start_ns']<=g['host_submit_ns'][0]<=phase['end_ns'] for g in assessment['groups'])
                phase['sample_phase_scope']='Rank-0 host submission phase, not GPU start or a global atomic snapshot'
            terminal=[c for c in client if c.get('stage')==name and c['event']=='terminal']
            submitted=[c for c in client if c.get('stage')==name and c['event']=='submitted']
            by_id={c['request_id']:c['monotonic_ns'] for c in submitted}
            by_phase={w['name']:w for w in windows}
            phase_drift=[c['request_id'] for c in submitted if c.get('phase') in by_phase and
                c['monotonic_ns']>=by_phase[c['phase']]['end_ns']]
            latencies=[(c['monotonic_ns']-by_id[c['request_id']])/1e9 for c in terminal]
            duration=(max(c['monotonic_ns'] for c in terminal)-min(by_id.values()))/1e9
            request_metrics=[]
            for t in terminal:
                chunks=[c for c in client if c['event']=='output_chunk' and c.get('request_id')==t['request_id'] and c['output_tokens']]
                first=chunks[0]['monotonic_ns'] if chunks else None
                n=len(t['token_ids'])
                request_metrics.append(dict(request_id=t['request_id'],actual_output_tokens=n,
                    latency_ns=t['monotonic_ns']-by_id[t['request_id']],
                    TTFT_ns=first-by_id[t['request_id']] if first is not None else None,
                    mean_TPOT_ns=(chunks[-1]['monotonic_ns']-first)/(n-1) if first is not None and n>1 else None,
                    scope='Frontend chunk-receipt clocks; no invented per-token timestamp or approved SLO'))
            receipt=dict(name=name,status=assessment['status'],requests=len(terminal),stop=stop,
                median_residence_seconds=statistics.median(latencies),cohort_completion_rate=len(terminal)/duration,
                phase_reports=phases,coverage={rank:workers[rank]['coverage'] for rank in (0,1)},
                arrival_phase_mismatches=phase_drift,
                plan_sha256=hashlib.sha256((out/(name+'-plan.json')).read_bytes()).hexdigest(),
                request_metrics=request_metrics,
                drain_report=phase_report(core,client,name,drain_start,drain_end),
                service_scope='Finite-horizon observed cohort rate, not an SLO knee or hardware maximum')
            result['stages'].append(receipt);write(out/'load-result.json',result)
            print('Completed '+name+': '+str(len(terminal))+' normal completions',flush=True)
            if stop is not None or task_error:raise RuntimeError('SAFE_STOP_'+str(stop or 'REQUEST_ERROR'))
            return receipt

        def burst(n,name):
            def build(start):
                rows=make_plan(name,start,1,n)
                for row in rows:row['due_ns']=int(start*1e9);row['phase']='burst'
                return rows,[]
            return build

        telemetry(out,'start')
        mapping=await stage('mapping',burst(4,'mapping'),protocol['mapping_observer'],True)
        result['mapping_gate']='PASS; actual multi-request rows and sampled decode validated'
        light={**protocol['observer'],'diagnostic':False}
        pilots=[]
        for n in (2,4,8):pilots.append(await stage('burst'+str(n),burst(n,'burst'+str(n)),light))
        result['overhead_proxy']=dict(mapping_diagnostic_median_s=mapping['median_residence_seconds'],
            burst4_light_median_s=pilots[1]['median_residence_seconds'],
            limits='Cold mapping versus later burst: different batch/JIT/cache histories; not a valid quantitative observer-overhead estimate')
        result['overhead']='UNKNOWN under load; no fully uninstrumented trajectory or replicated overhead comparison run'
        q0=max(s['cohort_completion_rate'] for s in pilots)
        residence=max(s['median_residence_seconds'] for s in pilots)
        horizon=min(protocol['phase_max_seconds'],max(protocol['phase_min_seconds'],protocol['phase_residence_multiplier']*residence))
        probes=[]
        for i,multiplier in enumerate(protocol['probe_rate_multipliers']):
            rate=q0*multiplier;span=min(horizon,protocol['probe_offer_cap']/rate);name='probe'+str(i+1)
            def build(start,name=name,rate=rate,span=span):
                rows=make_plan(name,start,span,rate)
                for row in rows:row['phase']='probe'
                return rows,[dict(name='probe',start_ns=int(start*1e9),end_ns=int((start+span)*1e9))]
            probe=await stage(name,build,light);probe['tested_rate']=rate;probes.append(probe)
            if probe['phase_reports'][0]['observed_service_deficit']:break
        bounded=bounded_trajectory(probes,horizon,protocol['max_offers']-actual_offers,deadline-time.monotonic(),protocol)
        frozen=dict(reference_burst_rate=q0,reference_residence_seconds=residence,
            probe_receipts=probes,**bounded,condition='narrative',
            source_sha256=initial_hashes,source_scope='Operational excitation derived from retained observations, not calibrated capacity',
            clock_identity=clock)
        horizon=bounded['phase_seconds']
        if hashes()!=initial_hashes:raise ValueError('Source drift before trajectory')
        write(out/'TRAJECTORY_FREEZE.json',frozen)
        write(out/'TRAJECTORY_FREEZE_SHA256.json',dict(sha256=hashlib.sha256((out/'TRAJECTORY_FREEZE.json').read_bytes()).hexdigest()))
        def trajectory(start):
            rows=[];windows=[]
            for i,(phase,rate) in enumerate(zip(protocol['phase_names'],frozen['rates'])):
                phase_start=start+i*horizon
                part=make_plan('trajectory',phase_start,horizon,rate,family='narrative',offset=len(rows))
                for row in part:row['phase']=phase
                rows+=part;windows.append(dict(name=phase,start_ns=int(phase_start*1e9),end_ns=int((phase_start+horizon)*1e9)))
            return rows,windows
        telemetry(out,'before-trajectory')
        trajectory_receipt=await stage('trajectory',trajectory,protocol['observer'])
        result['observed_pressure']=any(p['observed_service_deficit'] for p in trajectory_receipt['phase_reports'])
        result['status']='PASS_BOUNDED_TRAJECTORY' if result['observed_pressure'] else 'PARTIAL_NO_OVERLOAD_DEMONSTRATED'
        if any(p['paired_diagnostic_batches']==0 for p in trajectory_receipt['phase_reports']):
            result['status']='PARTIAL_DIAGNOSTIC_PHASE_COVERAGE'
        if trajectory_receipt['arrival_phase_mismatches']:result['status']='PARTIAL_ARRIVAL_PHASE_DRIFT'
        result['recovery_observations']=dict(high=trajectory_receipt['phase_reports'][2],
            low_recovery=trajectory_receipt['phase_reports'][3],drain=trajectory_receipt['drain_report'],
            final_request_KV_accounting='PASS',scope='Multiple retained trajectories; no scalar recovery verdict or D2 equivalence')
        result['interpretation']='Collection/accounting result; binding physical cause remains a nonexclusive hypothesis'
        telemetry(out,'end')
        await engine.collective_rpc('research_rank_state',timeout=30,args=(str(out),'after'))
        after=[json.loads((out/f'after-dp{r}.json').read_text()) for r in (0,1)];validate_residency(after,gpu_ids)
        if any(any(a[k]!=b[k] for k in ('pid','gpu_uuid','expert_layers','kv_tensors','kv_capacity_blocks')) for a,b in zip(loaded,after)):
            raise ValueError('Rank residency/incarnation changed')
        result['independent_content_comparison']='NOT_RUN'
    except Exception:
        result.update(status='PARTIAL' if result['stages'] else 'BLOCKED',error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if engine is not None and active_episode is not None:
            try:await abort_tasks(jobs_all,active_episode)
            except Exception:result['cleanup_abort_error']=traceback.format_exc()
            try:
                await asyncio.wait_for(engine.wait_for_requests_to_drain(drain_timeout=10),12)
                await asyncio.wait_for(engine.collective_rpc('load_end',timeout=10,args=(active_episode,)),15)
                result['failure_worker_trace_exported']=active_episode
            except Exception:result['failure_worker_export_error']=traceback.format_exc()
        for task in jobs_all:
            if not task.done():task.cancel()
        pending_jobs=[t for t in jobs_all if not t.done()]
        if pending_jobs:await asyncio.wait(pending_jobs,timeout=3)
        if engine is not None:
            try:engine.shutdown(timeout=20);result['shutdown_returned']=True
            except Exception:result.update(status='PARTIAL',shutdown_error=traceback.format_exc())
            telemetry(out,'post-shutdown')
            if owned_before:
                owned_after=owned_process_receipts([int(pid) for pid in owned_before])
                write(out/'owned-processes-after.json',owned_after)
                survivors=[pid for pid,p in owned_after.items() if p.get('present') and
                    p.get('start_ticks')==owned_before[pid].get('start_ticks')]
                result['owned_processes_remaining']=survivors
                unknown=[pid for pid,p in owned_after.items() if p.get('present') is None or owned_before[pid].get('present') is not True]
                result['cleanup_unknown_processes']=unknown
                result['cleanup_identity_scope']='PID plus Linux process start ticks; no unrelated process is signalled'
                if survivors:result['status']='PARTIAL_CLEANUP'
                if unknown:result['status']='PARTIAL_CLEANUP_UNVERIFIED'
            pending_jobs=[t for t in jobs_all if not t.done()]
            if pending_jobs:
                await asyncio.wait(pending_jobs,timeout=3)
                result['client_tasks_remaining']=sum(not t.done() for t in pending_jobs)
                if result['client_tasks_remaining']:result['status']='PARTIAL_CLEANUP'
        ledger.stream.close()
        write(out/'request-identity-joins.json',joins);write(out/'load-result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--model',required=True);p.add_argument('--gpu-ids',required=True)
    a=p.parse_args();result=asyncio.run(run(Path(a.output),a.model,json.loads(a.gpu_ids)))
    print(json.dumps({k:v for k,v in result.items() if k!='stages'},indent=2))
