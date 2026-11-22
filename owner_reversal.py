#!/usr/bin/env python3
"""Four sequential native A-B-B-A observation episodes, no profiler or fitting."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from observation_contract import validate_episode, verify_combine

ROOT = Path(__file__).resolve().parent


def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n')


async def experiment(out, model, gpu_ids):
    os.environ['MOE_OBSERVATION_ROOT'] = str(out)
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '0'
    os.environ['NCCL_DEBUG'] = 'INFO'
    result = dict(status='BLOCKED', episodes=[], future_routing_prediction='OUT_OF_SCOPE',
                  timing='NOT_QUALIFIED', bottleneck_BLIS_D2_claims='NONE')
    engine = None
    try:
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from transformers import AutoTokenizer
        import vllm
        config = json.loads((ROOT / 'model-qualification.json').read_text())
        expected = json.loads((ROOT / 'observation.json').read_text())
        package = Path(vllm.__file__).parent.parent
        sources = {n: hashlib.sha256((package / n).read_bytes()).hexdigest() for n in expected['native_source_sha256']}
        write(out / 'native-sources.json', sources)
        if sources != expected['native_source_sha256']:
            raise ValueError('Native observation source differs from reviewed pinned source; no compatibility patch attempted')
        kwargs = dict(config['engine_args']); kwargs['model'] = str(model)
        native = AsyncEngineArgs(**kwargs).create_engine_config()
        async_enabled = native.scheduler_config.async_scheduling
        baseline_scheduler = native.scheduler_config.get_scheduler_cls()
        if baseline_scheduler.__name__ not in ('Scheduler', 'AsyncScheduler'):
            raise ValueError('Unexpected baseline native scheduler')
        kwargs.update(async_scheduling=async_enabled,
            scheduler_cls='observation_scheduler.Observed' + baseline_scheduler.__name__,
            worker_extension_cls='observation_worker.ObservationWorker')
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
        actual = engine.vllm_config
        if (actual.scheduler_config.async_scheduling != async_enabled
                or actual.max_concurrent_batches != native.max_concurrent_batches
                or actual.cache_config.enable_prefix_caching):
            raise ValueError('Observation changed baseline scheduling/APC configuration')
        write(out / 'configuration.json', dict(baseline_scheduler=baseline_scheduler.__name__,
            observed_scheduler=actual.scheduler_config.get_scheduler_cls().__name__, async_scheduling=async_enabled,
            max_concurrent_batches=actual.max_concurrent_batches, prefix_caching=False,
            engine_args=kwargs, gpu_ids=gpu_ids, profiler_enabled=False,
            native_source_revision=expected['native_source_revision'], native_source_sha256=sources,
            categories={'persistent_state':'request/KV/pending work', 'configuration':'placement/policy/capacity',
                        'transition_events':'grant/allocation/submission/output application/release',
                        'observations':'post-gate routing and native call/device completion receipts'}))
        await engine.check_health()
        await engine.collective_rpc('research_rank_state', timeout=30, args=(str(out), 'loaded'))
        from native_model import validate_residency
        loaded = [json.loads((out / f'loaded-dp{r}.json').read_text()) for r in (0, 1)]
        validate_residency(loaded, gpu_ids)
        await engine.collective_rpc('observation_install', timeout=30, args=(str(out),))
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        text = ('In the laboratory, a researcher writes careful notes about a small experiment. '
                'Every observation is recorded before any conclusion is drawn. ') * 4
        tokens = tokenizer.encode(text, add_special_tokens=False)[:32]
        if len(tokens) != 32: raise ValueError('Fixed prompt is not exactly 32 tokens')
        write(out / 'workload.json', dict(prompt_token_ids=tokens, decoded=tokenizer.decode(tokens),
            prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(), owners=[0, 1, 1, 0],
            max_tokens=8, temperature=0, seed=1525, ignore_eos=False,
            note='Same known prompt and stop policy. Realized lengths/routing are observations, not prospective inputs.'))
        joins = {}
        assign = engine.input_processor.assign_request_id

        def observed_assign(request):
            value = assign(request)
            joins[request.external_req_id] = dict(external_request_id=request.external_req_id,
                internal_request_id=request.request_id, requested_rank=request.data_parallel_rank,
                prompt_token_ids=list(request.prompt_token_ids), source='Native InputProcessor.assign_request_id output',
                monotonic_ns=time.monotonic_ns())
            write(out / 'request-identity-joins.json', joins)
            return value

        engine.input_processor.assign_request_id = observed_assign
        for i, owner in enumerate([0, 1, 1, 0]):
            name = f'episode-{i+1}-' + ('A' if owner == 0 else 'B')
            await engine.wait_for_requests_to_drain(drain_timeout=30)
            await engine.collective_rpc('observation_begin', timeout=30, args=(name,))
            terminal = None
            params = SamplingParams(temperature=0.0, max_tokens=8, seed=1525)
            async for output in engine.generate(dict(prompt_token_ids=tokens), params, name, data_parallel_rank=owner):
                terminal = output
            if terminal is None or not terminal.finished or len(terminal.outputs) != 1 or terminal.request_id != name:
                raise ValueError('Missing normal terminal')
            completion = terminal.outputs[0]
            terminal_record = dict(request_id=terminal.request_id, finished=terminal.finished,
                token_ids=list(completion.token_ids), text=completion.text, finish_reason=completion.finish_reason,
                receipt_monotonic_ns=time.monotonic_ns(), effective_completion_time='NOT_OBSERVED')
            write(out / (name + '-terminal.json'), terminal_record)
            if completion.finish_reason not in ('stop', 'length') or not 0 < len(completion.token_ids) <= 8:
                raise ValueError('Unsupported terminal outcome')
            await engine.wait_for_requests_to_drain(drain_timeout=30)
            await engine.collective_rpc('observation_end', timeout=60, args=(name,))
            workers = {r: json.loads((out / f'{name}-worker-dp{r}.json').read_text()) for r in (0, 1)}
            core = [json.loads(line) for r in (0, 1) for line in (out / f'core-dp{r}.jsonl').read_text().splitlines()]
            admitted = [e['request_id'] for e in core if e['event']=='request_accepted']
            if len(admitted) != len(joins) or set(admitted) != {j['internal_request_id'] for j in joins.values()}:
                raise ValueError('Duplicate/orphan request admissions outside declared episode cohort')
            proof = validate_episode(name, owner, joins[name], core, workers,
                numerical_check=lambda a,b,start,n:verify_combine(out,a,b,start,n))
            proof['terminal'] = terminal_record
            write(out / (name + '-assessment.json'), proof)
            result['episodes'].append(proof)
            write(out / 'observation-result.json', result)
        result['status'] = 'PASS' if all(e['remote_expert_participation']=='PASS' for e in result['episodes']) else 'PARTIAL'
        await engine.collective_rpc('research_rank_state', timeout=30, args=(str(out), 'after'))
        after = [json.loads((out / f'after-dp{r}.json').read_text()) for r in (0, 1)]
        validate_residency(after, gpu_ids)
        if any(any(a[k]!=b[k] for k in ['pid','gpu_uuid','expert_layers','kv_capacity_blocks','kv_tensors']) for a,b in zip(loaded,after)):
            raise ValueError('Worker incarnation/placement/KV backing changed during the episodes')
        result['worker_identity_placement_and_KV_backing_preserved'] = True
    except Exception:
        result.update(status='BLOCKED' if not result['episodes'] else 'PARTIAL', error=traceback.format_exc())
        traceback.print_exc()
    finally:
        if engine is not None:
            try: engine.shutdown(timeout=30); result['shutdown_returned'] = True
            except Exception: result.update(status='PARTIAL', shutdown_error=traceback.format_exc())
        write(out / 'observation-result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu-ids', required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(experiment(Path(args.output), args.model, json.loads(args.gpu_ids))), indent=2))
