"""Frozen-trace sensitivity analysis; retrospective evidence, never a predictor."""
import copy
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def frozen_trace(root=ROOT):
    config = json.loads((root / 'load_sensitivity.json').read_text())
    raw = (root / 'load_replay_trace.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['trace_sha256']:
        raise ValueError('Frozen arrival trace hash mismatch')
    if hashlib.sha256((root / 'load_protocol.json').read_bytes()).hexdigest() != config['load_protocol_sha256']:
        raise ValueError('Original workload/protocol drift')
    trace = json.loads(raw)
    rows, windows = trace['arrivals'], trace['windows']
    if len(rows) != 85 or len(windows) != 4 or len({r['request_id'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate frozen workload')
    for i, w in enumerate(windows):
        if w['end_ns'] - w['start_ns'] != 25_000_000_000 or (i and w['start_ns'] != windows[i-1]['end_ns']):
            raise ValueError('Frozen phase boundaries changed')
    for i, row in enumerate(rows):
        window = next((w for w in windows if w['name'] == row['phase']), None)
        if (row['owner'] != i % 2 or row['family'] != 'narrative' or row['stage'] != 'trajectory'
                or window is None or not window['start_ns'] <= row['due_ns'] < window['end_ns']
                or (i and row['due_ns'] < rows[i-1]['due_ns'])):
            raise ValueError('Frozen ownership/content/calendar mismatch')
    return config, trace


def rebase(trace, start_ns):
    """Integer translation only: every interarrival and phase offset is preserved."""
    if type(start_ns) is not int:
        raise ValueError('Nanosecond anchor must be an integer')
    result = copy.deepcopy(trace)
    anchor = trace['windows'][0]['start_ns'] if trace['windows'] else min(r['due_ns'] for r in trace['arrivals'])
    shift = start_ns - anchor
    for row in result['arrivals']:
        row['due_ns'] += shift
    for window in result['windows']:
        window['start_ns'] += shift
        window['end_ns'] += shift
    return result


def schedule_identity(plan):
    translated = rebase(plan, 0)
    return digest(dict(arrivals=translated['arrivals'], windows=translated['windows']))


def configure(engine_args, limit, observer, scheduler_kind, async_scheduling):
    if type(limit) is not int or limit not in (4, 8) or observer not in ('observed', 'reference'):
        raise ValueError('Unsupported sensitivity cell')
    if scheduler_kind not in ('Scheduler', 'AsyncScheduler'):
        raise ValueError('Unsupported native scheduler')
    result = dict(engine_args, max_num_seqs=limit, async_scheduling=async_scheduling)
    if observer == 'observed':
        result.update(scheduler_cls='observation_scheduler.Observed' + scheduler_kind,
                      worker_extension_cls='observation_worker.ObservationWorker')
    else:
        result.pop('scheduler_cls', None)
        result['worker_extension_cls'] = 'model_observer.ReadOnlyRankState'
    return result


def process_members(worker_receipts, frontend_pid):
    """Read-only PID tracking, never signal the frontend or unrelated processes."""
    members = set()
    for pid, receipt in worker_receipts.items():
        if not receipt.get('present') or receipt.get('parent_pid') in (None, 0, 1):
            raise ValueError('Worker parent provenance unavailable')
        members.update((int(pid), receipt['parent_pid']))
    return sorted(members - {frontend_pid})


def client_assessment(stage, plan, client, joins, inputs, max_output):
    """Common terminal-only comparison used in BOTH observed and reference cells."""
    planned = {r['request_id']: r for r in plan['arrivals']}
    events = [e for e in client if e.get('stage') == stage]
    offers = [e for e in events if e['event'] == 'submitted']
    terminals = [e for e in events if e['event'] == 'terminal']
    if (len(planned) != len(plan['arrivals']) or len(offers) != len(planned)
            or len(terminals) != len(planned)
            or {e['request_id'] for e in offers} != set(planned)
            or {e['request_id'] for e in terminals} != set(planned)
            or any(e['event'] in ('unsent', 'request_error', 'client_cancelled', 'censored_before_cleanup') for e in events)):
        raise ValueError('Incomplete/duplicate/censored offered workload')
    by_offer = {e['request_id']: e for e in offers}
    by_terminal = {e['request_id']: e for e in terminals}
    internal = set()
    for rid, row in planned.items():
        offer, terminal = by_offer[rid], by_terminal[rid]
        if any(offer[k] != row[k] for k in ('due_ns', 'owner', 'family', 'phase')):
            raise ValueError('Actual offered workload differs from frozen plan')
        join = joins[rid]
        if join['internal_request_id'] in internal or join['prompt_token_ids'] != inputs[row['family']]:
            raise ValueError('Request identity/prompt join mismatch')
        internal.add(join['internal_request_id'])
        if (terminal['finish_reason'] not in ('length', 'stop') or not 0 <= len(terminal['token_ids']) <= max_output
                or terminal['monotonic_ns'] < offer['monotonic_ns'] or offer['monotonic_ns'] < row['due_ns']):
            raise ValueError('Invalid terminal or host event order')
    pending = peak = 0
    backlog = []
    for event in sorted(offers + terminals, key=lambda e: (e['monotonic_ns'], e['seq'])):
        pending += 1 if event['event'] == 'submitted' else -1
        if pending < 0:
            raise ValueError('Terminal precedes submission')
        peak = max(peak, pending)
        backlog.append([event['monotonic_ns'], pending])
    def outstanding(t):
        return sum(e['monotonic_ns'] <= t for e in offers) - sum(e['monotonic_ns'] <= t for e in terminals)
    end = plan['windows'][-1]['end_ns'] if plan['windows'] else max(e['due_ns'] for e in offers)
    phase_rows = []
    for w in plan['windows']:
        lo, hi = w['start_ns'], w['end_ns']
        phase_rows.append(dict(name=w['name'], pending_start=outstanding(lo), pending_end=outstanding(hi),
            submitted=sum(lo <= e['monotonic_ns'] < hi for e in offers),
            completed=sum(lo <= e['monotonic_ns'] < hi for e in terminals)))
    metrics = [dict(request_id=rid, owner_hint=planned[rid]['owner'],
                    residence_ns=by_terminal[rid]['monotonic_ns'] - by_offer[rid]['monotonic_ns'],
                    offer_lateness_ns=by_offer[rid]['monotonic_ns'] - planned[rid]['due_ns'],
                    output_tokens=len(by_terminal[rid]['token_ids']), output_digest=digest(by_terminal[rid]['token_ids']))
               for rid in planned]
    mismatches = [e['request_id'] for e in offers if plan['windows'] and e['monotonic_ns'] >= next(w['end_ns'] for w in plan['windows'] if w['name'] == e['phase'])]
    return dict(status='PASS_CLIENT_ACCOUNTING', requests=len(offers), normal_completions=len(terminals),
        schedule_digest=schedule_identity(plan), request_metrics=metrics, backlog_events=backlog,
        peak_outstanding=peak, phases=phase_rows, phase_drift=mismatches,
        median_residence_seconds=statistics.median(m['residence_ns'] for m in metrics) / 1e9,
        offer_lateness_ms=dict(median=statistics.median(m['offer_lateness_ns'] for m in metrics) / 1e6,
                               maximum=max(m['offer_lateness_ns'] for m in metrics) / 1e6),
        drain_seconds=max(0, max(e['monotonic_ns'] for e in terminals) - end) / 1e9,
        native_owner_verification='UNKNOWN in reference; separate native audit required for observed cells')


def comparison(cells, config):
    expected = {c['name']: c for c in config['order']}
    if len(cells) != 4 or {c['name'] for c in cells} != set(expected):
        raise ValueError('All four matched cells are required')
    for c in cells:
        if c['condition'] != expected[c['name']] or c['status'] != 'PASS_FIXED_REPLAY':
            raise ValueError('Incomplete or invalid sensitivity cell')
        if c['client']['phase_drift']:
            raise ValueError('Arrival phase drift prevents matched interpretation')
    by_name = {c['name']: c for c in cells}
    first = cells[0]
    for c in cells:
        if c['client']['schedule_digest'] != first['client']['schedule_digest'] or c['gpu_ids'] != first['gpu_ids']:
            raise ValueError('Workload or physical GPU identity changed')
        a, b = dict(c['engine_args']), dict(first['engine_args'])
        for args in (a, b):
            for key in ('max_num_seqs', 'scheduler_cls', 'worker_extension_cls'):
                args.pop(key, None)
        if a != b:
            raise ValueError('Unapproved secondary engine-argument change')
        a, b = dict(c['effective_scheduler']), dict(first['effective_scheduler'])
        if a.pop('max_num_seqs') != c['condition']['max_num_seqs']:
            raise ValueError('Effective active limit changed')
        if a.pop('active_requests') != c['condition']['max_num_seqs']:
            raise ValueError('Effective admission limit changed')
        b.pop('max_num_seqs')
        b.pop('active_requests')
        if a != b or c['native_sources'] != first['native_sources']:
            raise ValueError('Effective scheduler or native source drift')
        if c['condition']['observer'] == 'observed' and c['native_owner_audit'] != 'PASS':
            raise ValueError('Missing observed native ownership audit')
    contrasts = []
    for axis, pairs in [('active_limit', [('seq4_observed', 'seq8_observed'), ('seq4_reference', 'seq8_reference')]),
                        ('observer', [('seq4_reference', 'seq4_observed'), ('seq8_reference', 'seq8_observed')])]:
        for left, right in pairs:
            a, b = by_name[left]['client'], by_name[right]['client']
            am, bm = {m['request_id']: m for m in a['request_metrics']}, {m['request_id']: m for m in b['request_metrics']}
            if set(am) != set(bm):
                raise ValueError('Request population changed')
            contrasts.append(dict(axis=axis, left=left, right=right,
                median_residence_ratio_right_over_left=b['median_residence_seconds'] / a['median_residence_seconds'],
                paired_residence_delta_seconds={rid: (bm[rid]['residence_ns'] - am[rid]['residence_ns']) / 1e9 for rid in am},
                peak_outstanding_delta=b['peak_outstanding'] - a['peak_outstanding'],
                drain_seconds_delta=b['drain_seconds'] - a['drain_seconds'],
                output_length_changes=[rid for rid in am if am[rid]['output_tokens'] != bm[rid]['output_tokens']],
                output_token_changes=[rid for rid in am if am[rid]['output_digest'] != bm[rid]['output_digest']]))
    return dict(status='PASS_MATCHED_COLLECTION', contrasts=contrasts,
        kv_capacity_by_cell={c['name']: c['kv_capacity_blocks'] for c in cells},
        interpretation='One exploratory 2x2 block; no significance/materiality threshold invented. Inspect observer contrasts before interpreting the active-limit effect. Native owner/KV/work unobserved in references.',
        limits='Time drift, automatic KV profiling, batch-specific JIT, output variation and observer-by-batch interactions remain possible; no hardware maximum, pure decode, BLIS or D2 claim.')
