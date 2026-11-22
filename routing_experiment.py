#!/usr/bin/env python3
"""Experiment 2 only: fixed discovery and separately invoked held-out comparison."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from routing_analysis import digest, serving_summary, routing_summary, select_pair, timing_summary, compare_report
from observation_contract import validate_episode, verify_combine

ROOT=Path(__file__).resolve().parent
CODE=['routing_experiment.py','routing_analysis.py','routing_inputs.json','routing_run.py',
      'observation_worker.py','observation_timing.py','observation_scheduler.py','observation_contract.py',
      'model-qualification.json','model-pin.json','observation.json','runtime-lock.txt',
      'model_qualify.py','prepare_runtime.py','diagnose.py','model_observer.py','native_model.py']


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def code_hashes():return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in CODE}


def frozen_sequence(selected):
    rows=[]
    for block in range(3):
        modes=['control','timed','diagnostic'];modes=modes[block:]+modes[:block]
        families=selected if block!=1 else selected[::-1]
        for mode in modes:
            for family in families:rows.append(dict(block=block+1,mode=mode,family=family))
    return rows


def telemetry(out,label):
    command=['nvidia-smi','--query-gpu=uuid,temperature.gpu,clocks.sm,clocks.mem,pstate,power.draw,utilization.gpu,memory.used','--format=csv,noheader,nounits']
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=10)
        record=dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,
                    host_monotonic_ns=time.monotonic_ns(),scope='Outside request timing; no clock/power settings changed')
    except Exception as error:record=dict(command=command,status='UNOBSERVED',error=repr(error))
    write(out/('gpu-'+label+'.json'),record)


async def run(out,model,gpu_ids,stage,freeze_path=None,freeze_sha=None):
    os.environ.update(MOE_OBSERVATION_ROOT=str(out),VLLM_WORKER_MULTIPROC_METHOD='spawn',
                      VLLM_ALLOW_INSECURE_SERIALIZATION='0',NCCL_DEBUG='INFO')
    result=dict(status='BLOCKED',stage=stage,episodes=[],claims='Post-gate descriptive mechanisms only; no causal timing/BLIS/D2 claim')
    engine=None
    try:
        protocol=json.loads((ROOT/'routing_inputs.json').read_text())
        context=json.loads((out/'context.json').read_text())
        plan=None
        if stage=='compare':
            if hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest()!=freeze_sha:raise ValueError('Freeze digest mismatch')
            plan=json.loads(Path(freeze_path).read_text())
            if plan['code_sha256']!=code_hashes():raise ValueError('Experiment source changed after selection freeze')
            if plan['driver_versions']!=context['driver_versions']:raise ValueError('Driver drift since discovery; no silent adaptation')
            ids={f['id'] for f in protocol['families']}; selected=plan['selected']
            if len(selected)!=2 or len(set(selected))!=2 or not set(selected)<=ids or plan['sequence']!=frozen_sequence(selected):
                raise ValueError('Unknown frozen matrix/conditions')
            write(out/'FROZEN_COMPARISON.json',plan)
        from transformers import AutoTokenizer
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from native_model import validate_residency
        import vllm
        native_spec=json.loads((ROOT/'observation.json').read_text())
        expected={**native_spec['native_source_sha256'],**protocol['native_timing_sources']}
        package=Path(vllm.__file__).parent.parent
        actual_hashes={n:hashlib.sha256((package/n).read_bytes()).hexdigest() for n in expected}
        write(out/'native-sources.json',actual_hashes)
        if actual_hashes!=expected:raise ValueError('Pinned observation/timing native source mismatch')
        kwargs=dict(json.loads((ROOT/'model-qualification.json').read_text())['engine_args']);kwargs['model']=str(model)
        native=AsyncEngineArgs(**kwargs).create_engine_config()
        kind=native.scheduler_config.get_scheduler_cls().__name__
        if kind not in ('Scheduler','AsyncScheduler'):raise ValueError('Unsupported native scheduler')
        kwargs.update(async_scheduling=native.scheduler_config.async_scheduling,
            scheduler_cls='observation_scheduler.Observed'+kind,worker_extension_cls='observation_worker.ObservationWorker')
        engine=AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
        if (engine.vllm_config.scheduler_config.async_scheduling!=native.scheduler_config.async_scheduling
            or engine.vllm_config.max_concurrent_batches!=native.max_concurrent_batches
            or engine.vllm_config.cache_config.enable_prefix_caching):raise ValueError('Baseline policy changed')
        write(out/'configuration.json',dict(engine_args=kwargs,async_scheduling=native.scheduler_config.async_scheduling,
            max_concurrent_batches=native.max_concurrent_batches,profiler=False,APC=False,
            selected_environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','VLLM_USE_V2_MODEL_RUNNER',
                'VLLM_RANDOMIZE_DP_DUMMY_INPUTS','PYTORCH_CUDA_ALLOC_CONF','TORCH_CUDA_ALLOC_CONF']},
            source=code_hashes(),timing='CUDA events; no per-layer sync; no aligned GPU clock'))
        await engine.check_health()
        await engine.collective_rpc('research_rank_state',timeout=30,args=(str(out),'loaded'))
        loaded=[json.loads((out/f'loaded-dp{r}.json').read_text()) for r in (0,1)];validate_residency(loaded,gpu_ids)
        await engine.collective_rpc('observation_install',timeout=30,args=(str(out),True))
        qualification={r:json.loads((out/f'timing-qualification-dp{r}.json').read_text()) for r in (0,1)}
        if any(q['dp_rank']!=r or q['worker_pid']!=loaded[r]['pid'] or q['gpu_uuid']!=loaded[r]['gpu_uuid'] for r,q in qualification.items()):
            raise ValueError('Timing evidence/worker identity mismatch')
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
        def tokens(text):
            ids=tokenizer.encode(text,add_special_tokens=False)[:32]
            if len(ids)!=32:raise ValueError('Prompt shorter than frozen 32-token domain')
            return ids
        material={f['id']:{partition:tokens(f[partition]) for partition in ('screen','evaluate')} for f in protocol['families']}
        write(out/'prompt-material.json',material)
        if len({tuple(ids) for v in material.values() for ids in v.values()})!=12:
            raise ValueError('Screen/evaluation prompts are not twelve distinct token inputs')
        warmup=tokens(protocol['warmup'])
        joins={};assign=engine.input_processor.assign_request_id
        def observed_assign(request):
            value=assign(request)
            joins[request.external_req_id]=dict(internal_request_id=request.request_id,
                external_request_id=request.external_req_id,prompt_token_ids=list(request.prompt_token_ids),
                source='Native assign_request_id result',monotonic_ns=time.monotonic_ns())
            write(out/'request-identity-joins.json',joins);return value
        engine.input_processor.assign_request_id=observed_assign

        async def episode(name,prompt,mode,family=None,block=None,warm=False):
            await engine.wait_for_requests_to_drain(drain_timeout=30)
            await engine.collective_rpc('observation_begin',timeout=30,args=(name,mode))
            start=time.perf_counter_ns();terminal=None
            async for output in engine.generate(dict(prompt_token_ids=prompt),SamplingParams(temperature=0,max_tokens=1,seed=1525),name,data_parallel_rank=0):terminal=output
            end=time.perf_counter_ns()
            if terminal is None or terminal.request_id!=name or not terminal.finished or len(terminal.outputs)!=1:raise ValueError('Missing terminal')
            item=terminal.outputs[0]
            if item.finish_reason not in ('length','stop') or not 0<=len(item.token_ids)<=1:raise ValueError('Unsupported one-output-cap terminal')
            terminal_row=dict(request_id=name,token_ids=list(item.token_ids),finish_reason=item.finish_reason,text=item.text,
                client_start_perf_counter_ns=start,client_terminal_perf_counter_ns=end,
                client_elapsed_ns=end-start,clock='Same frontend process perf_counter_ns; not GPU/network duration')
            write(out/(name+'-terminal.json'),terminal_row)
            await engine.wait_for_requests_to_drain(drain_timeout=30)
            await engine.collective_rpc('observation_end',timeout=60,args=(name,))
            workers={r:json.loads((out/f'{name}-worker-dp{r}.json').read_text()) for r in (0,1)}
            core=[json.loads(line) for r in (0,1) for line in (out/f'core-dp{r}.jsonl').read_text().splitlines()]
            if {e['request_id'] for e in core if e['event']=='request_accepted'}!={j['internal_request_id'] for j in joins.values()}:raise ValueError('Orphan admissions')
            serving,frame=serving_summary(name,joins[name],core,workers,prompt)
            row=dict(episode=name,family=family,block=block,warmup=warm,mode=mode,serving=serving,
                prompt_token_ids=prompt,terminal=terminal_row,client_elapsed_ns=end-start,
                routing=None,timing=None)
            if mode!='control':row['routing']=routing_summary(frame,workers[1])
            if mode=='timed':row['timing']=timing_summary(frame,workers[1],qualification)
            if mode=='diagnostic':
                row['contribution_diagnostic']=validate_episode(name,0,joins[name],core,workers,
                    numerical_check=lambda a,b,s,n:verify_combine(out,a,b,s,n))
            write(out/(name+'-receipt.json'),row);result['episodes'].append(row)
            write(out/'routing-result.json',result)
            print('Completed '+name+' ('+mode+')',flush=True)
            return row

        telemetry(out,'start')
        if stage=='discover':
            await episode('warmup-discovery',warmup,'diagnostic',warm=True)
            screens={}
            for family in protocol['families']:
                row=await episode('screen-'+family['id'],material[family['id']]['screen'],'diagnostic',family=family['id'])
                screens[family['id']]=row['routing']
            best,pairs=select_pair(screens);write(out/'all-screening-pairs.json',pairs)
            result['selection']=best
            result['selection_basis']='Routing only; no timing/effect size significance threshold'
            if best:
                selected=[best['a'],best['b']]
                frozen=dict(protocol='routing-comparison-freeze-v0.8',code_sha256=code_hashes(),
                    model_revision=json.loads((ROOT/'model-pin.json').read_text())['revision'],
                    driver_versions=context['driver_versions'],hardware_domain='Two T4 SM75; UUIDs may differ between A/B, no cross-stage timing comparison',
                    selected=selected,evaluation_tokens={f:material[f]['evaluate'] for f in selected},
                    screening_tokens_sha256={f:digest(material[f]['screen']) for f in selected},
                    screening_distance=best,sequence=frozen_sequence(selected),
                    interpretation='Non-identical screening routing. Held-out contrast/timing not yet measured. No success tuning.')
                write(out/'FROZEN_COMPARISON.json',frozen)
                result.update(status='PASS',freeze_path=str(out/'FROZEN_COMPARISON.json'),
                    freeze_sha256=hashlib.sha256((out/'FROZEN_COMPARISON.json').read_bytes()).hexdigest())
            else:result.update(status='PARTIAL',reason='NO_ROUTING_CONTRAST_IN_FIXED_SET; no comparison freeze or automatic run')
        else:
            selected=plan['selected']
            if len(selected)!=2 or len(set(selected))!=2 or any(f not in material for f in selected):raise ValueError('Unknown frozen conditions')
            if plan['sequence']!=frozen_sequence(selected) or plan['evaluation_tokens']!={f:material[f]['evaluate'] for f in selected}:raise ValueError('Frozen matrix/input tampering')
            for mode in ('diagnostic','timed','control'):await episode('warmup-'+mode,warmup,mode,warm=True)
            last_block=0
            for i,spec in enumerate(plan['sequence'],1):
                if spec['block']!=last_block:telemetry(out,'block-'+str(spec['block']));last_block=spec['block']
                await episode(f'compare-{i:02d}-'+spec['family']+'-'+spec['mode'],plan['evaluation_tokens'][spec['family']],
                    spec['mode'],family=spec['family'],block=spec['block'])
            analysis=compare_report(result['episodes'],selected);write(out/'comparison-analysis.json',analysis)
            result.update(status='PASS' if analysis['timing_complete'] and analysis['logical_communication_bytes_constant'] is True and analysis['serving_shape_constant'] else 'PARTIAL',analysis=analysis,
                status_scope='Collection/qualification only; not proof of a latency effect or essential state variable')
        telemetry(out,'end')
        await engine.collective_rpc('research_rank_state',timeout=30,args=(str(out),'after'))
        after=[json.loads((out/f'after-dp{r}.json').read_text()) for r in (0,1)];validate_residency(after,gpu_ids)
        if any(any(a[k]!=b[k] for k in ['pid','gpu_uuid','expert_layers','kv_tensors','kv_capacity_blocks']) for a,b in zip(loaded,after)):
            raise ValueError('Physical worker identity/placement changed')
    except Exception:
        result.update(status='PARTIAL' if result['episodes'] else 'BLOCKED',error=traceback.format_exc());traceback.print_exc()
    finally:
        if engine is not None:
            try:engine.shutdown(timeout=30);result['shutdown_returned']=True
            except Exception:result.update(status='PARTIAL',shutdown_error=traceback.format_exc())
        write(out/'routing-result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--model',required=True)
    p.add_argument('--gpu-ids',required=True);p.add_argument('--stage',choices=['discover','compare'],required=True)
    p.add_argument('--freeze');p.add_argument('--freeze-sha256');a=p.parse_args()
    result=asyncio.run(run(Path(a.output),a.model,json.loads(a.gpu_ids),a.stage,a.freeze,a.freeze_sha256))
    print(json.dumps({k:v for k,v in result.items() if k not in ('episodes','analysis')},ensure_ascii=False,indent=2))
