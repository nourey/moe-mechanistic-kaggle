#!/usr/bin/env python3
"""Real AsyncLLM DP2/EP2 model load and conditional two-request functional smoke."""
import argparse
import asyncio
import gzip
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parent


def validate_residency(rows, expected_ids):
    if len(rows) != 2 or {r.get('dp_rank') for r in rows} != {0, 1}:
        raise ValueError('Missing or duplicate DP worker observations')
    by_rank = {r['dp_rank']: r for r in rows}
    if len(set(expected_ids)) != 2 or {r['ep_rank'] for r in rows} != {0, 1}:
        raise ValueError('Distinct GPUs and complete EP rank membership required')
    if len({r['pid'] for r in rows}) != 2: raise ValueError('Two separate worker processes required')
    names = None
    for rank, row in by_rank.items():
        if row['gpu_uuid'] != expected_ids[rank]: raise ValueError('Wrong GPU ownership')
        if (row['tp_world_size'], row['pp_world_size'], row['dp_world_size'], row['ep_world_size']) != (1, 1, 2, 2):
            raise ValueError('Not TP1/PP1/DP2/EP2')
        if row['model_type'] != 'olmoe' or row['model_dtype'] != 'torch.float16' or row['global_num_experts'] != 64:
            raise ValueError('Wrong model or dtype domain')
        if row['all2all_backend'] != 'allgather_reducescatter': raise ValueError('Wrong collective backend')
        if row.get('ep_communication_manager') != 'AgRsAll2AllManager':
            raise ValueError('Native AGRS manager not observed')
        if not row['kv_capacity_blocks'] or not row['kv_tensors'] or not all(t['is_cuda'] for t in row['kv_tensors']):
            raise ValueError('Rank-local GPU KV capacity not independently observed')
        layers = row['expert_layers']
        if len(layers) != 16: raise ValueError('Expected 16 observed MoE layers')
        current = {x['name'] for x in layers}
        if len(current) != 16 or (names is not None and current != names): raise ValueError('Layer identity mismatch')
        names = current
        for layer in layers:
            if not layer['is_cuda'] or layer['dtype'] != 'torch.float16' or layer['w13_shape'][0] != 32 or layer['w2_shape'][0] != 32:
                raise ValueError('Expected 32 resident FP16 experts per layer/rank')
            mapping = layer['global_to_local_map']
            if mapping is None or len(mapping) != 64 or sorted(x for x in mapping if x >= 0) != list(range(32)):
                raise ValueError('Missing or contradictory authoritative expert map')
    for name in names:
        maps = [next(x for x in by_rank[r]['expert_layers'] if x['name'] == name)['global_to_local_map'] for r in (0, 1)]
        if any(sum(m[e] >= 0 for m in maps) != 1 for e in range(64)):
            raise ValueError('Duplicate or orphan global expert ownership')
    return True


def validate_terminals(records, specs):
    if len(records) != len(specs) or len({r['request_id'] for r in records}) != len(specs):
        raise ValueError('Missing/duplicate terminal request')
    by_id = {r['request_id']: r for r in records}
    for spec in specs:
        row = by_id[spec['request_id']]
        if not row['finished'] or row['finish_reason'] not in ('stop', 'length') or not 0 < len(row['token_ids']) <= spec['max_tokens']:
            raise ValueError('Non-normal terminal outcome')
    return True


def inspect_traces(folder, rows):
    receipts = []
    for path in sorted(Path(folder).rglob('*')):
        if not (path.name.endswith('.json') or path.name.endswith('.json.gz')): continue
        try:
            with (gzip.open(path, 'rt') if path.name.endswith('.gz') else path.open()) as f: data = json.load(f)
            events = data.get('traceEvents', [])
            cpu_pids = {e.get('pid') for e in events if e.get('cat') in ('cpu_op', 'python_function', 'user_annotation')}
            kernels = [e for e in events if e.get('cat') == 'kernel']
            for row in rows:
                if row['pid'] not in cpu_pids: continue
                names = [e['name'] for e in kernels if e.get('args', {}).get('device') == row['device_index']]
                receipts.append(dict(file=str(path), dp_rank=row['dp_rank'], worker_pid=row['pid'],
                    device_index=row['device_index'], gpu_uuid=row['gpu_uuid'], kernel_count=len(names),
                    moe_calls=sum('fused_moe_kernel' in n for n in names), nccl_calls=sum('nccl' in n.lower() for n in names),
                    allgather_calls=sum('nccl' in n.lower() and 'allgather' in n.lower().replace('_', '') for n in names),
                    reducescatter_calls=sum('nccl' in n.lower() and 'reducescatter' in n.lower().replace('_', '') for n in names),
                    sendrecv_calls=sum('nccl' in n.lower() and 'sendrecv' in n.lower().replace('_', '') for n in names)))
        except (OSError, ValueError, TypeError):
            receipts.append(dict(file=str(path), error='Unreadable/incomplete trace; no identity guessed'))
    return receipts


async def run(folder, model_path, config, expected_ids):
    out = Path(folder)
    result = dict(D_MODEL_RESIDENCY='FAIL', E_FUNCTIONAL_GENERATION='NOT_EXECUTED',
                  scheduler_request_KV_ledger='NOT_VERIFIED', model_path=str(model_path),
                  scope='One collocated native OLMoE group; not P/D, HTTP/EPP, Real readiness or GPU Q qualification')
    engine = None
    events = []
    def mark(phase):
        events.append(dict(phase=phase, monotonic_ns=time.monotonic_ns(), clock='Harness process monotonic_ns'))
        (out / 'model-events.json').write_text(json.dumps(events, indent=2) + '\n')
    try:
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.config import ProfilerConfig
        from vllm.v1.engine.async_llm import AsyncLLM
        (out / 'worker-state').mkdir()
        profile_dir = out / 'profiles'; profile_dir.mkdir()
        args = dict(config['engine_args'])
        args['model'] = str(model_path)
        args['profiler_config'] = ProfilerConfig(profiler='torch', torch_profiler_dir=str(profile_dir),
            torch_profiler_activities=['CPU', 'CUDA'], torch_profiler_with_stack=False,
            torch_profiler_record_shapes=False, torch_profiler_dump_cuda_time_total=False)
        mark('native_model_construction')
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**args))
        await engine.check_health()
        result['client_type'] = type(engine.engine_core).__name__
        mark('loaded_read_only_worker_observation')
        await engine.collective_rpc('research_rank_state', timeout=30, args=(str(out / 'worker-state'), 'loaded'))
        rows = [json.loads((out / 'worker-state' / ('loaded-dp' + str(r) + '.json')).read_text()) for r in (0, 1)]
        validate_residency(rows, expected_ids)
        result.update(D_MODEL_RESIDENCY='PASS', loaded_workers=rows)
        (out / 'native-result.json').write_text(json.dumps(result, indent=2) + '\n')
        mark('profile_start')
        await engine.start_profile(profile_prefix='two-dp-requests')

        async def request(spec):
            mark('submit_' + spec['request_id'])
            final = None
            params = SamplingParams(temperature=0.0, max_tokens=spec['max_tokens'], seed=1525)
            async for output in engine.generate(spec['prompt'], params, spec['request_id'], data_parallel_rank=spec['data_parallel_rank']):
                final = output
            if final is None or final.request_id != spec['request_id'] or len(final.outputs) != 1:
                raise RuntimeError('Missing or mismatched native request output')
            item = final.outputs[0]
            row = dict(request_id=final.request_id, requested_dp_rank=spec['data_parallel_rank'],
                binding_evidence='Explicit native rank hint; client selector source supports it. No independent runtime binding receipt.',
                finished=final.finished, finish_reason=item.finish_reason, stop_reason=str(item.stop_reason),
                text=item.text, token_ids=list(item.token_ids), prompt_token_ids=list(final.prompt_token_ids or []),
                terminal_receipt_monotonic_ns=time.monotonic_ns(), clock='Harness receipt time, not engine/client-visible effective completion')
            with (out / (spec['request_id'] + '-terminal.json')).open('x') as f: json.dump(row, f, indent=2)
            return row

        terminals = await asyncio.gather(*(request(spec) for spec in config['requests']))
        validate_terminals(terminals, config['requests'])
        result['terminals'] = terminals
        await engine.stop_profile()
        mark('after_read_only_worker_observation')
        await engine.collective_rpc('research_rank_state', timeout=30, args=(str(out / 'worker-state'), 'after'))
        after = [json.loads((out / 'worker-state' / ('after-dp' + str(r) + '.json')).read_text()) for r in (0, 1)]
        validate_residency(after, expected_ids)
        for before, later in zip(rows, after):
            if (before['pid'], before['gpu_uuid'], before['expert_layers']) != (later['pid'], later['gpu_uuid'], later['expert_layers']):
                raise RuntimeError('Worker identity or expert ownership changed during the fixture')
        result['after_workers'] = after
        traces = inspect_traces(profile_dir, rows)
        result['trace_receipts'] = traces
        # Generic NCCL kernels could be coordination rather than MoE payload. Do not count those alone.
        covered = {r.get('dp_rank') for r in traces if r.get('moe_calls', 0) > 0
                   and r.get('allgather_calls', 0) > 0 and r.get('reducescatter_calls', 0) > 0}
        result['E_FUNCTIONAL_GENERATION'] = 'PASS' if covered == {0, 1} else 'PARTIAL'
        result['E_scope_note'] = 'Two normal terminal outputs plus worker-joined MoE/AllGather/ReduceScatter kernels required for bounded functional PASS. Other/variable-size NCCL paths remain PARTIAL, not guessed. Full request-to-engine binding and scheduler/KV page ledger remain NOT VERIFIED.'
    except Exception:
        result['error'] = traceback.format_exc(); traceback.print_exc()
        if result['D_MODEL_RESIDENCY'] == 'PASS': result['E_FUNCTIONAL_GENERATION'] = 'FAIL'
    finally:
        if engine is not None:
            mark('shutdown_begin')
            try: engine.shutdown(timeout=30); result['shutdown_returned'] = True
            except Exception: result['shutdown_error'] = traceback.format_exc()
        (out / 'native-result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    # Only this child process and its descendants; no notebook/system environment change.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '0'
    os.environ['NCCL_DEBUG'] = 'INFO'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--gpu-ids', required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / 'model-qualification.json').read_text())
    result = asyncio.run(run(Path(args.output), args.model_path, config, json.loads(args.gpu_ids)))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['D_MODEL_RESIDENCY'] == 'PASS' and result['E_FUNCTIONAL_GENERATION'] in ('PASS', 'PARTIAL') and result.get('shutdown_returned') else 1)
