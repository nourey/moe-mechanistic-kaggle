"""Small experiment-specific receipts and offline structural checks (no CUDA imports)."""
import json
import ast
import math
import os
from pathlib import Path
import threading
import struct
import time
import uuid


class Writer:
    def __init__(self, folder, role, rank):
        self.rank = rank
        self.incarnation = uuid.uuid4().hex
        self.path = Path(folder) / (role + '-dp' + str(rank) + '.jsonl')
        self.stream = self.path.open('x', buffering=1)
        self.seq = 0
        self.lock = threading.Lock()
        self.load_clock = clock_receipt() if os.environ.get('MOE_LOAD_MODE') == '1' else None

    def emit(self, event, **fields):
        with self.lock:
            self.seq += 1
            row = dict(event=event, seq=self.seq, pid=os.getpid(), dp_rank=self.rank,
                       observation_instance=self.incarnation, monotonic_ns=time.monotonic_ns(),
                       clock='Local process monotonic_ns; ordering joins use explicit lineage, not clock equality', **fields)
            if self.load_clock is not None: row['clock_identity'] = self.load_clock
            self.stream.write(json.dumps(row) + '\n')
            return row


def kv_rows(manager, request_id):
    return [[dict(block_id=b.block_id, ref_cnt=b.ref_cnt) for b in group]
            for group in manager.get_blocks(request_id).blocks]


def request_row(req):
    return dict(request_id=req.request_id, status=str(req.status), status_name=getattr(req.status,'name',str(req.status)),
                num_computed_tokens=req.num_computed_tokens,
                num_in_flight_tokens=req.num_in_flight_tokens,
                num_output_tokens=req.num_output_tokens)


def clock_receipt():
    """Same-host comparison is conditional on these actual Linux clock facts."""
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        namespace = os.readlink('/proc/self/ns/time')
        implementation = time.get_clock_info('monotonic').implementation
        verified = bool(boot and namespace and 'CLOCK_MONOTONIC' in implementation)
        return dict(verified=verified, boot_id=boot, time_namespace=namespace,
                    implementation=implementation, units='ns')
    except OSError as error:
        return dict(verified=False, reason=repr(error), units='ns')


def native_batch_record(batch):
    """Read only CPU-native row ordering, never infer it from grant dict order."""
    def plain(value): return value.tolist() if hasattr(value, 'tolist') else list(value)
    ids=list(batch.req_ids); bounds=plain(batch.query_start_loc_np)[:len(ids)+1]
    counts=plain(batch.num_scheduled_tokens)[:len(ids)]
    progress=plain(batch.num_computed_tokens_np)[:len(ids)]
    prefill=plain(batch.is_prefilling_np)[:len(ids)]
    contexts=plain(batch.seq_lens_cpu_upper_bound)[:len(ids)]
    if len(set(ids))!=len(ids) or not bounds or bounds[0]!=0 or len(bounds)!=len(ids)+1:
        raise ValueError('Invalid native batch identity/bounds')
    if any(bounds[i+1]-bounds[i]!=counts[i] or counts[i]<=0 for i in range(len(ids))):
        raise ValueError('Native batch rows do not match positive grants')
    if bounds[-1]!=int(batch.num_tokens): raise ValueError('Native real-token extent mismatch')
    return dict(req_ids=ids, bounds=bounds, counts=counts, computed_cpu=progress,
                is_prefilling=prefill, contexts_cpu_upper_bound=contexts,
                num_tokens=int(batch.num_tokens), padded_tokens=int(batch.num_tokens_after_padding),
                provenance='Native prepare_inputs return; CPU query_start_loc_np and req_ids; context/progress may be optimistic')


def submission_work(native_grants, dummy):
    # Native MRV2 fabricates SchedulerOutput entries for _dummy_run. Their
    # positive token counts are execution/padding work, never request grants.
    return dict(grants={} if dummy else dict(native_grants),
                native_scheduled_tokens=dict(native_grants), dummy=bool(dummy))


def read_half_npy(path):
    with Path(path).open('rb') as stream:
        if stream.read(6) != b'\x93NUMPY': raise ValueError('Not NPY')
        version = tuple(stream.read(2))
        if version not in ((1,0),(2,0)): raise ValueError('Unsupported NPY version')
        size = struct.unpack('<H' if version == (1,0) else '<I', stream.read(2 if version == (1,0) else 4))[0]
        if size > 4096: raise ValueError('Oversized NPY header')
        header = ast.literal_eval(stream.read(size).decode('latin1'))
        if header['descr'] != '<f2' or header['fortran_order']: raise ValueError('Only C-layout FP16 supported')
        shape = header['shape']
        if len(shape) != 2 or not 0 < shape[0] <= 128 or not 0 < shape[1] <= 4096: raise ValueError('Unexpected tensor shape')
        data = stream.read(math.prod(shape)*2+1)
        if len(data) != math.prod(shape)*2: raise ValueError('Truncated NPY')
        values = struct.unpack('<' + str(math.prod(shape)) + 'e', data)
        if not all(math.isfinite(v) for v in values): raise ValueError('Nonfinite contribution')
        return shape, values


def verify_combine(folder, owner_layer, remote_layer, start, count):
    def read(row,key):
        name = row[key]
        if Path(name).name != name: raise ValueError('Non-local tensor path')
        return read_half_npy(Path(folder)/name)
    shape, a = read(owner_layer,'partial_output_file')
    other_shape, b = read(remote_layer,'partial_output_file')
    final_shape, observed = read(owner_layer,'combine_output_file')
    if shape != other_shape or final_shape != (count,shape[1]): raise ValueError('Collective tensor shape mismatch')
    lo=start*shape[1]; hi=(start+count)*shape[1]
    if hi > len(a): raise ValueError('Collective slice out of bounds')
    expected = [struct.unpack('<e',struct.pack('<e',x+y))[0] for x,y in zip(a[lo:hi],b[lo:hi])]
    mismatch=sum(x!=y for x,y in zip(expected,observed))
    omitted=sum(x!=y for x,y in zip(a[lo:hi],observed))
    return dict(verified=mismatch==0 and omitted>0, values=len(observed), exact_FP16_sum_mismatches=mismatch,
                remote_omission_negative_control_mismatches=omitted,
                rule='Two FP16 native contributions summed and rounded to FP16; no fitted tolerance',
                files=[owner_layer['partial_output_file'],remote_layer['partial_output_file'],owner_layer['combine_output_file']])


def validate_episode(external_id, expected_owner, join, core, workers, numerical_check=None):
    """Reports measured links separately from native-semantics-derived dependencies."""
    rid = join['internal_request_id']
    admissions = [e for e in core if e['event'] == 'request_accepted' and e['request_id'] == rid]
    if len(admissions) != 1 or admissions[0]['dp_rank'] != expected_owner:
        raise ValueError('Missing/duplicate/wrong authoritative scheduler ownership')
    submitted = [e for e in core if e['event'] == 'schedule' and rid in e['grants']]
    applied = [e for e in core if e['event'] == 'output_applied' and rid in e['grants']]
    if not submitted or not applied: raise ValueError('Missing grants/output application')
    if any(e['dp_rank'] != expected_owner or e['grants'][rid] <= 0 for e in submitted + applied):
        raise ValueError('Wrong owner or non-positive grant')
    if {e['decision'] for e in submitted} != {e['decision'] for e in applied}:
        raise ValueError('Submitted/output-applied decisions do not reconcile')
    if len({e['decision'] for e in submitted}) != len(submitted):
        raise ValueError('Duplicate scheduler decisions')
    allocations = [e for e in core if e['event'] == 'allocation' and e['request_id'] == rid]
    frees = [e for e in core if e['event'] == 'kv_free' and e['request_id'] == rid]
    if not allocations or not frees or any(e['dp_rank'] != expected_owner for e in allocations + frees):
        raise ValueError('Missing/incorrect request-owned KV lifecycle')
    if not any(e['accepted'] and any(e['after_blocks']) for e in allocations):
        raise ValueError('No positive request-associated KV allocation observed')
    if any(e.get('remaining_blocks') and any(e['remaining_blocks']) for e in frees):
        raise ValueError('Request retained live block references after observed free')
    if not all(e['ref_cnt'] == 0 for row in frees for group in row['released_blocks'] for e in group):
        raise ValueError('Released request blocks retain references in APC-off scope')
    initial = [e for e in core if e['event'] == 'core_initialized' and e['dp_rank'] == expected_owner]
    if len(initial) != 1 or frees[-1]['free_blocks'] != initial[0]['free_blocks']:
        raise ValueError('KV free population did not return to initial baseline')
    if applied[-1]['after'] or applied[-1]['deferred_frees']:
        raise ValueError('Episode retained active requests/deferred frees')
    forward_by_rank = {r: workers[r]['forwards'] for r in (0, 1)}
    owner_real = [f for f in forward_by_rank[expected_owner] if f['grants'].get(rid, 0) > 0 and f['layers']]
    if not owner_real: raise ValueError('No actual worker submission for owned request')
    for f in owner_real:
        n = f['grants'][rid]
        positions = f.get('positions', [])[:n]
        matches = [e for e in submitted if e['grants'][rid] == n and positions
            and e['before'][rid]['num_computed_tokens'] == positions[0]
            and positions == list(range(positions[0], positions[0]+n))]
        if len(matches) != 1: raise ValueError('Ambiguous scheduler-grant to actual worker-position join')
        f['validated_core_decision'] = matches[0]['decision']
    remote = 1 - expected_owner
    if any(f['grants'] for f in forward_by_rank[remote]):
        raise ValueError('Nominal idle rank actually received request work')
    # For this one-request episode only: join owner-local token positions into the
    # native DP all-gather layout. Match global routing content and native sizes;
    # never equate local forward sequence numbers across processes.
    links = []; matched_layers = 0; assigned_remote = 0
    # First prompt forward is unique in this episode. Decode routing can repeat;
    # retain it raw without manufacturing one-to-one cross-rank matches.
    prompt_n = len(join['prompt_token_ids'])
    first = [f for f in owner_real if f['grants'][rid] == prompt_n and f['positions'][:prompt_n] == list(range(prompt_n))]
    if len(first) != 1: raise ValueError('No unique unchunked first-prompt forward; stop this bounded claim')
    for f in first:
        if f.get('input_ids', [])[:prompt_n] != join['prompt_token_ids']:
            raise ValueError('Actual worker input differs from joined prompt token material')
        if f.get('error'): continue
        for layer in f['layers']:
            if layer.get('error') or layer.get('global_to_local_map') is None: continue
            sizes = layer['dp_sizes']
            ids = layer['topk_ids']
            if len(sizes) != 2 or sum(sizes) != len(ids): continue
            start = sum(sizes[:expected_owner]); n = f['grants'][rid]
            if sizes[expected_owner] != n or len(f.get('positions', [])) < n: continue
            if any(len(row) != 8 or any(e < 0 or e >= 64 for e in row) or len(set(row)) != 8 for row in ids[start:start+n]):
                raise ValueError('Invalid real-token top-k expert assignments')
            matches = [(rf, rl) for rf in forward_by_rank[remote] for rl in rf['layers']
                if not rl.get('error') and rl['layer'] == layer['layer'] and rl['topk_ids'] == ids
                and rl['dp_sizes'] == sizes and rf['dummy']]
            # Repeated identical decode routing is ambiguous; don't guess a join.
            if len(matches) != 1: continue
            rf, rl = matches[0]
            maps = [layer['global_to_local_map'], rl['global_to_local_map']]
            if any(len(m) != 64 or sorted(x for x in m if x >= 0) != list(range(32)) for m in maps):
                raise ValueError('Contradictory expert placement')
            if any(sum(m[e] >= 0 for m in maps) != 1 for e in range(64)):
                raise ValueError('Duplicate/orphan expert placement')
            matched_layers += 1
            assignments = sum(rl['global_to_local_map'][e] >= 0 for row in ids[start:start+n] for e in row)
            assigned_remote += assignments
            if assignments == 0: continue
            rows = [i for i in range(start, start+n) if any(rl['global_to_local_map'][e] >= 0 for e in ids[i])]
            nonzero = any(rl['output_abs_sums'][i] > 0 for i in rows)
            chain = (rl['cuda_completion_verified'] and rl['combine_input_matches_expert_output']
                     and rl['combine_returned'] and layer['combine_returned']
                     and (layer.get('dispatch') or {}).get('returned') and (rl.get('dispatch') or {}).get('returned'))
            chain = chain and all((r.get('dispatch') or {}).get('gathered_shape', [None])[0] == len(ids) for r in (layer, rl))
            if nonzero and chain:
                numerical = None
                if numerical_check is not None and layer.get('partial_output_file') and rl.get('partial_output_file'):
                    numerical = numerical_check(layer,rl,start,n)
                links.append(dict(layer=layer['layer'], owner_forward=f['forward'], remote_forward=rf['forward'],
                    owner_positions=f['positions'][:n], core_decision=f['validated_core_decision'], remote_real_assignments=assignments,
                    matching_native_layout_and_routing=True, remote_output_nonzero=True,
                    numerical_combine=numerical,
                    dependency='Observed expert-output tensor feeds native collective combine; source semantics require rank participation. Not an ablation or timing result.'))
    numerical_pass = any((l['numerical_combine'] or {}).get('verified') for l in links)
    return dict(external_request_id=external_id, internal_request_id=rid, verified_owner=expected_owner,
        scheduler_KV_worker_links='PASS', remote_expert_participation='PASS' if numerical_pass else
            ('INSUFFICIENT_EXCITATION' if matched_layers == 16 and assigned_remote == 0 else 'UNVERIFIED'),
        matched_first_prompt_layers=matched_layers, assigned_remote_real_experts=assigned_remote,
        links=links, ambiguous_or_missing_forward_joins_not_guessed=True,
        physical_timing='NOT_QUALIFIED', critical_path='NOT_CLAIMED')
