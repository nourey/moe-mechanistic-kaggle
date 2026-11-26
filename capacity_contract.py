"""Small experiment contract: physical devices, frozen offers, immutable bindings."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ['capacity_contract.py', 'capacity_analysis.py', 'capacity_group.py', 'capacity_run.py', 'capacity_worker.py',
          'capacity_kaggle.py', 'capacity_protocol.json', 'model-pin.json', 'model-qualification.json',
          'runtime-lock.txt', 'load_protocol.json', 'load_replay_trace.json', 'load_analysis.py',
          'load_sensitivity.py', 'load_sensitivity.json', 'load_experiment.py', 'observation_contract.py',
          'observation_scheduler.py', 'observation_worker.py', 'observation_timing.py', 'model_observer.py',
          'model_qualify.py', 'prepare_runtime.py', 'diagnose.py', 'routing_experiment.py',
          'routing_analysis.py', 'routing_inputs.json', 'observation.json', 'native_model.py']


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def source_hashes():
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in SOURCE}


def freeze(intensity=1.0, family='narrative', smoke=False):
    if not math.isfinite(intensity) or not .25 <= intensity <= 4:
        raise ValueError('Operational intensity must be between .25 and 4')
    if family not in ('narrative', 'identifiers', 'mixed'):
        raise ValueError('Unsupported pinned prompt family')
    if smoke and intensity!=1:raise ValueError('Smoke uses one fixed two-burst calendar')
    from load_sensitivity import frozen_trace
    _, original = frozen_trace()
    t0 = original['windows'][0]['start_ns']
    rows = []
    base = original['arrivals'][:8] if smoke else original['arrivals']
    for i, r in enumerate(base):
        kind = ('identifiers', 'narrative')[((i // 4) + i % 2) % 2] if family == 'mixed' else family
        smoke_offset=i*50_000_000 if i<4 else 2_500_000_000+(i-4)*50_000_000
        rows.append(dict(request_id=f'req-{i:04d}', offset_ns=smoke_offset if smoke else round((r['due_ns']-t0)/intensity), family=kind))
    duration = 4_000_000_000 if smoke else round(100_000_000_000/intensity)
    windows = [dict(name=w['name'], start_offset_ns=round((w['start_ns']-t0)/intensity),
                    end_offset_ns=round((w['end_ns']-t0)/intensity)) for w in original['windows']] if not smoke else []
    value = dict(protocol='olmoe-workload-v1', original_trace_sha256=hashlib.sha256((ROOT/'load_replay_trace.json').read_bytes()).hexdigest(),
                 arrivals=rows, duration_ns=duration, windows=windows, intensity=intensity, family=family,
                 max_output_tokens=8, smoke=smoke,
                 domain='Fixed offers; no output-length/routing oracle, no capacity estimate or SLO knee')
    value['sha256'] = digest(value)
    return value


def validate_workload(workload):
    raw = {k: v for k, v in workload.items() if k != 'sha256'}
    if digest(raw) != workload['sha256']:
        raise ValueError('Workload digest mismatch')
    rows = workload['arrivals']
    if not rows or len(rows) > 128 or len({r['request_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid offered population')
    if not 0 < workload['duration_ns'] <= 400_000_000_000 or workload['max_output_tokens'] != 8:
        raise ValueError('Unsupported workload duration/output policy')
    if any(r['family'] not in ('narrative', 'identifiers') or not 0 <= r['offset_ns'] < workload['duration_ns'] for r in rows):
        raise ValueError('Invalid arrival or prompt family')
    if [r['offset_ns'] for r in rows] != sorted(r['offset_ns'] for r in rows):
        raise ValueError('Arrival order changed')
    return True


def topology(arm):
    config = read(ROOT/'capacity_protocol.json')
    if arm not in config['topology']:
        raise ValueError('Unknown physical arm')
    groups = copy.deepcopy(config['topology'][arm])
    used = [d for g in groups for d in g['devices']]
    if len(used) != len(set(used)) or any(g['dp'] != len(g['devices']) for g in groups):
        raise ValueError('Overlapping GPU allocation or rank count')
    return groups


def route_plan(arm, workload, stage='main', primer_group=None):
    validate_workload(workload)
    groups = topology(arm)
    ranks = {g['group']: 0 for g in groups}
    boundary = round(workload['duration_ns'] * read(ROOT/'capacity_protocol.json')['activation_fraction'])
    rows = []
    cursor = 0
    for offer in workload['arrivals']:
        if primer_group:
            eligible = [primer_group]
        else:
            eligible = ['D1', 'D2'] if arm == 'C1' and offer['offset_ns'] >= boundary else ['D1']
        group = eligible[cursor % len(eligible)]
        cursor += 1
        dp = next(g['dp'] for g in groups if g['group']==group)
        row = dict(offer, request_id=f'{stage}-{offer["request_id"]}', stage=stage,
                   group=group, owner=ranks[group] % dp,
                   binding_regime='post' if offer['offset_ns'] >= boundary else 'pre')
        ranks[group] += 1
        rows.append(row)
    return dict(arm=arm, stage=stage, arrivals=rows, duration_ns=workload['duration_ns'],
                boundary_offset_ns=boundary, workload_sha256=workload['sha256'],
                semantics='Predeclared immutable client routing plan. Not an EPP eligibility command/epoch or production router test.')


def translate(plan, t0_ns, group):
    rows = [dict(r, due_ns=t0_ns+r['offset_ns'], phase=r['binding_regime']) for r in plan['arrivals'] if r['group']==group]
    return rows


def engine_arguments(dp, mode, model, rpc_port):
    if dp not in (1, 2) or mode not in ('reference','diagnostic'):
        raise ValueError('Unsupported group/measurement configuration')
    args = dict(read(ROOT/'model-qualification.json')['engine_args'])
    args.update(model=str(model), data_parallel_size=dp, data_parallel_size_local=dp,
                data_parallel_rpc_port=rpc_port, data_parallel_address='127.0.0.1',
                data_parallel_backend='mp', enable_expert_parallel=True)
    args['worker_extension_cls'] = 'capacity_worker.CapacityWorker'
    # The existing subclasses preserve native sync/async methods; resolved after config construction.
    return args


def memory_floor(dp):
    pin = read(ROOT/'model-pin.json');c=pin['config']
    expert_bytes = c['num_hidden_layers']*c['num_experts']*3*c['hidden_size']*c['intermediate_size']*2
    dense_bytes = pin['index_total_tensor_bytes']-expert_bytes
    return dense_bytes + expert_bytes // dp


def check_devices(arm, devices, utilization=.8):
    groups = topology(arm)
    selected = []
    for g in groups:
        if any(i >= len(devices) for i in g['devices']):
            raise ValueError(f'{arm} requires {sum(x["dp"] for x in groups)} distinct CUDA-visible physical GPUs')
        rows = [devices[i] for i in g['devices']]
        selected += rows
        for d in rows:
            if d['total_memory']*utilization <= memory_floor(g['dp']):
                raise ValueError(f'INFEASIBLE_STATIC {arm}/{g["group"]}: FP16 weight floor exceeds memory target; no offload/quantization/utilization fallback')
            if d.get('free_memory',d['total_memory']) < d['total_memory']*utilization:
                raise ValueError('Selected GPU lacks the unchanged memory target; no interference or automatic utilization reduction')
    if len({d['uuid'] for d in selected}) != len(selected):
        raise ValueError('Distinct physical GPU UUIDs required; no two-GPU emulation of C')
    if len({(d.get('name'),tuple(d.get('compute_capability',[])),d['total_memory']) for d in selected})>1:
        raise ValueError('Heterogeneous GPUs are outside the initial matched hardware domain')
    return dict(status='NECESSARY_MEMORY_AND_IDENTITY_CHECK_PASS',
                sufficient_fit='UNVERIFIED until native load/warmup succeeds', groups=groups)


def physical_gpu_binding(devices, nvml_csv):
    def normal(value):return value.strip().removeprefix('GPU-').lower()
    physical={}
    for line in nvml_csv.splitlines():
        if not line.strip():continue
        parts=[x.strip() for x in line.split(',')]
        if len(parts)!=3:raise ValueError('Malformed physical NVML record')
        gpu,pci,mig=parts
        if normal(gpu) in physical:raise ValueError('Duplicate physical NVML UUID')
        physical[normal(gpu)]=(pci,mig)
    result=[]
    for d in devices:
        key=normal(d['uuid'])
        if key not in physical or d['uuid'].startswith('MIG-'):
            raise ValueError('CUDA device is not independently bound to a full physical NVML GPU; MIG emulation forbidden')
        pci,mig=physical[key]
        if mig.lower()=='enabled':raise ValueError('MIG-enabled GPUs are outside this four-full-GPU domain')
        result.append(dict(d,pci_bus_id=pci,mig_mode=mig))
    if len({d['pci_bus_id'] for d in result})!=len(result):raise ValueError('Duplicate physical PCI GPU; distinct UUIDs alone are insufficient')
    return result


def validate_residency(rows, expected_ids, dp):
    if dp==2:
        from native_model import validate_residency as existing
        return existing(rows, expected_ids)
    if len(rows)!=1 or len(expected_ids)!=1:
        raise ValueError('EP1 requires one observed rank')
    r=rows[0]
    if (r['dp_rank'],r['ep_rank'],r['tp_world_size'],r['pp_world_size'],r['dp_world_size'],r['ep_world_size']) != (0,0,1,1,1,1):
        raise ValueError('Not DP1/EP1')
    if r['gpu_uuid']!=expected_ids[0] or r['model_type']!='olmoe' or r['model_dtype']!='torch.float16' or r['global_num_experts']!=64:
        raise ValueError('Wrong EP1 GPU/model/dtype')
    if not r['kv_capacity_blocks'] or not r['kv_tensors'] or not all(t['is_cuda'] for t in r['kv_tensors']):
        raise ValueError('Missing GPU KV residency')
    if len(r['expert_layers'])!=16 or len({x['name'] for x in r['expert_layers']})!=16:
        raise ValueError('Missing EP1 layers')
    for layer in r['expert_layers']:
        if not layer['is_cuda'] or layer['dtype']!='torch.float16' or layer['w13_shape'][0]!=64 or layer['w2_shape'][0]!=64:
            raise ValueError('All 64 experts must be physically resident on EP1')
        mapping=layer['global_to_local_map']
        if mapping is not None and mapping != list(range(64)):
            raise ValueError('Unexpected singleton expert placement')
    return True


def membership_hash(incarnation, gpu_uuid, rank):
    return list(hashlib.sha256((incarnation+'|'+gpu_uuid+'|'+str(rank)).encode()).digest())


def validate_membership(receipts, loaded, incarnation):
    expected=[membership_hash(incarnation,r['gpu_uuid'],r['dp_rank']) for r in loaded]
    if len(receipts)!=len(loaded):raise ValueError('Missing collective member receipt')
    for rank,(r,w) in enumerate(zip(receipts,loaded)):
        if (r['pid'],r['gpu_uuid'],r['rank'],r['group_incarnation'])!=(w['pid'],w['gpu_uuid'],rank,incarnation):
            raise ValueError('Membership process/GPU/group identity mismatch')
        if r['members']!=expected:raise ValueError('Mixed/missing EP group: collective membership witness mismatch')
    return True


def validate_slo(slo):
    keys={'ttft_seconds','mean_tpot_seconds','end_to_end_seconds'}
    if set(slo)!=keys or any(v is not None and (type(v) not in (int,float) or not math.isfinite(v) or v<=0) for v in slo.values()):
        raise ValueError('Explicit positive finite SLO thresholds or null required')
    return any(v is not None for v in slo.values())
