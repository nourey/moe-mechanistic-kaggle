"""Retrospective load accounting. No future-routing predictor or fitted cost law."""
import math
import statistics


def median(values): return statistics.median(values) if values else None


def common_clock(receipts):
    keys=('boot_id','time_namespace','implementation')
    if not receipts or any(not r.get('verified') for r in receipts):raise ValueError('Unverified host clock')
    if len({tuple(r[k] for k in keys) for r in receipts})!=1:raise ValueError('Different host clock domains')
    return receipts[0]


def make_plan(stage,t0,duration,rate,family='mixed',offset=0):
    if not all(math.isfinite(x) for x in (t0,duration,rate)) or duration<0 or rate<=0:raise ValueError('Invalid arrival plan')
    count=int(math.ceil(duration*rate-1e-12))
    pattern=[('identifiers',0),('narrative',1),('narrative',0),('identifiers',1)]
    rows=[]
    for i in range(count):
        kind,rank=pattern[(offset+i)%4]
        rows.append(dict(request_id=f'{stage}-{offset+i:04d}',stage=stage,family=kind if family=='mixed' else family,
                         owner=rank if family=='mixed' else (offset+i)%2,due_ns=int((t0+i/rate)*1e9)))
    return rows


def bounded_trajectory(probes,target_seconds,remaining_offers,remaining_seconds,protocol):
    """Freeze before treatment; only pilot evidence and explicit resource bounds enter."""
    preferred=next((p for p in probes if p['phase_reports'][0]['observed_service_deficit']),probes[-1])
    candidates=[preferred]+[p for p in reversed(probes) if p is not preferred]
    for chosen in candidates:
        rates=[f*chosen['tested_rate'] for f in protocol['phase_rate_factors']]
        # Four ceil operations can add <4 offers; reserve that rounding margin.
        offers_horizon=(remaining_offers-4)/sum(rates)
        time_horizon=(remaining_seconds-protocol['drain_timeout_seconds']-5)/len(rates)
        duration=min(target_seconds,offers_horizon,time_horizon)
        if duration>=protocol['phase_min_seconds']:
            return dict(high_rate=chosen['tested_rate'],rates=rates,phase_seconds=duration,
                target_phase_seconds=target_seconds,pressure_found_in_probe=chosen['phase_reports'][0]['observed_service_deficit'],
                high_rate_limited_by_budget=chosen is not preferred,duration_limited_by_budget=duration<target_seconds,
                bound_scope='Resource-constrained observation window; not a saturation/SLO threshold')
    raise ValueError('SAFE_STOP_NO_FULL_TRAJECTORY_FITS_BUDGET; no partial phase sequence started')


def queue_samples(core,rank):
    samples=[]
    for e in core:
        if e['dp_rank']!=rank:continue
        s=e.get('load_after',e.get('load_state'))
        if s is not None:samples.append(dict(time_ns=e['monotonic_ns'],**s))
    return samples


def validate_stage(stage,core,workers,client,joins,inputs,max_output,require_multi=False):
    offers=[c for c in client if c.get('stage')==stage and c['event']=='submitted']
    terminals=[c for c in client if c.get('stage')==stage and c['event']=='terminal']
    external={c['request_id']:c for c in offers}
    if len(external)!=len(offers) or len(terminals)!=len(offers) or {c['request_id'] for c in terminals}!=set(external):
        raise ValueError('Missing/duplicate client terminal accounting')
    if not offers:raise ValueError('No actual offers')
    expected={};terminal={c['request_id']:c for c in terminals}
    for rid,o in external.items():
        j=joins[rid]
        if j['prompt_token_ids']!=inputs[o['family']]:raise ValueError('Request prompt join drift')
        if j['internal_request_id'] in expected:raise ValueError('Duplicate internal request identity')
        expected[j['internal_request_id']]=rid
        t=terminal[rid]
        if t['finish_reason'] not in ('length','stop') or not 0<=len(t['token_ids'])<=max_output:raise ValueError('Non-normal terminal')
    selected=set(expected);accept=[e for e in core if e['event']=='request_accepted' and e['request_id'] in selected]
    if len(accept)!=len(selected) or {e['request_id'] for e in accept}!=selected:raise ValueError('Native admission accounting mismatch')
    if any(e['dp_rank']!=external[expected[e['request_id']]]['owner'] for e in accept):raise ValueError('Actual owner mismatch')
    if any(e['prompt_token_ids']!=inputs[external[expected[e['request_id']]]['family']] for e in accept):raise ValueError('Native prompt mismatch')
    clocks=[offers[0]['clock_identity']]+[w['clock_identity'] for w in workers.values()]
    clocks += [e['clock_identity'] for e in core if e['event']=='core_initialized']
    common_clock(clocks)
    for rank in (0,1):
        w=workers[rank]
        init=next(e for e in core if e['event']=='core_initialized' and e['dp_rank']==rank)
        if w['observed_parent_pid']!=init['pid']:raise ValueError('Worker/Core parent identity mismatch')
        if w['coverage']['overflow']:raise ValueError('Worker observation overflow')
        owned={rid for rid in selected if external[expected[rid]]['owner']==rank}
        live={};block_owner={}
        def block_keys(groups):return {(g,b['block_id']) for g,blocks in enumerate(groups) for b in blocks}
        for event in sorted((e for e in core if e['dp_rank']==rank and e.get('request_id') in owned),key=lambda e:e.get('seq',e['monotonic_ns'])):
            rid=event['request_id']
            if event['event']=='allocation':
                before=live.get(rid,set());after=block_keys(event['after_blocks'])
                if 'before_blocks' in event and block_keys(event['before_blocks'])!=before:raise ValueError('KV before-owner ledger mismatch')
                if not event['accepted'] and after!=before:raise ValueError('Failed allocation changed live ownership')
                if before-after:raise ValueError('Unrecorded KV release during allocation')
                for block in after:
                    if block in block_owner and block_owner[block]!=rid:raise ValueError('Duplicate live KV owner in APC-off domain')
                    block_owner[block]=rid
                live[rid]=after
            elif event['event']=='kv_free':
                released=block_keys(event['released_blocks'])
                if released!=live.get(rid,set()):raise ValueError('Double/unmatched KV release')
                for block in released:
                    if block_owner.pop(block,None)!=rid:raise ValueError('Wrong KV release owner')
                live[rid]=block_keys(event['remaining_blocks'])
        if block_owner or any(live.values()):raise ValueError('Live KV refs/owner remain after drain')
        frees=[e for e in core if e['event']=='kv_free' and e['request_id'] in owned]
        for rid in owned:
            alloc=[e for e in core if e['event']=='allocation' and e['request_id']==rid]
            fs=[e for e in frees if e['request_id']==rid]
            if not any(e['accepted'] and any(e['after_blocks']) for e in alloc) or not fs:raise ValueError('Missing KV allocation/release')
            if any(fs[-1]['remaining_blocks']) or any(b['ref_cnt']!=0 for f in fs for g in f['released_blocks'] for b in g):
                raise ValueError('Live KV refs retained')
        if frees and frees[-1]['free_blocks']!=init['free_blocks']:raise ValueError('KV population did not return to baseline')
        last=[e for e in core if e['event']=='output_applied' and e['dp_rank']==rank and owned.intersection(e['grants'])]
        if owned and (not last or owned.intersection(last[-1]['after']) or last[-1]['deferred_frees']):
            raise ValueError('Active request/deferred cleanup retained after drain')
    decisions={};applied=set()
    for e in core:
        if e['event']=='schedule' and selected.intersection(e['grants']):
            decisions[(e['dp_rank'],e['decision'])]=e
        if e['event']=='output_applied' and selected.intersection(e['grants']):applied.add((e['dp_rank'],e['decision']))
    if set(decisions)!=applied:raise ValueError('Grant/output-application ledger mismatch')
    matched=set();multi=0;decode_rows=0;verified_decode_rows=0;unknown_lookahead_rows=0
    for rank,w in workers.items():
        for f in w['forwards']:
            if not f['grants']:continue
            b=f['batch']
            if b is None or dict(zip(b['req_ids'],b['counts']))!=f['grants']:raise ValueError('Native batch/grants mismatch')
            if set(b['req_ids'])-selected:raise ValueError('Orphan native batch request')
            multi+=len(b['req_ids'])>1
            candidates=[]
            for i,rid in enumerate(b['req_ids']):
                lo,hi=b['bounds'][i:i+2]
                if hi-lo!=b['counts'][i]:raise ValueError('Incorrect native row slice')
                start=b['computed_cpu'][i]
                if f['sample']:
                    actual=f['positions'][lo:hi]
                    if not actual or actual!=list(range(actual[0],actual[0]+hi-lo)):raise ValueError('Non-contiguous actual token rows')
                    start=actual[0]
                    prompt=inputs[external[expected[rid]]['family']]
                    generated=terminal[expected[rid]]['token_ids']
                    for row,pos in zip(f['input_ids'][lo:hi],actual):
                        if pos<len(prompt) and row!=prompt[pos]:raise ValueError('Actual prompt token slice mismatch')
                        if pos>=len(prompt):
                            index=pos-len(prompt)
                            if index<len(generated):
                                if row!=generated[index]:raise ValueError('Actual decode token/request join mismatch')
                                verified_decode_rows+=1
                            else:unknown_lookahead_rows+=1
                    decode_rows+=sum(pos>=len(prompt) for pos in actual)
                options=[key for key,e in decisions.items() if key[0]==rank and rid in e['grants'] and
                    e['grants'][rid]==hi-lo and e['before'][rid]['num_computed_tokens']==start]
                if len(options)!=1:raise ValueError('Ambiguous grant/native-input position join')
                candidates.append(options[0])
            if len(set(candidates))!=1 or candidates[0] in matched:raise ValueError('Torn/duplicate batch-decision join')
            matched.add(candidates[0])
    if matched!=set(decisions):raise ValueError('Missing actual worker submission')
    native_request_metrics=[]
    for admission in accept:
        rid=admission['request_id'];rank=admission['dp_rank']
        first=min((e for (r,_),e in decisions.items() if r==rank and rid in e['grants']),key=lambda e:e['monotonic_ns'])
        native_request_metrics.append(dict(request_id=rid,owner=rank,
            accept_return_ns=admission.get('native_accept_return_ns'),
            accept_publication_ns=admission['monotonic_ns'],
            first_decision_interval_ns=first.get('native_decision_interval_ns'),
            first_decision_publication_ns=first['monotonic_ns'],
            scope='Native host call boundaries/publication; exact internal decision instant is interval-bounded'))
    if require_multi and (multi==0 or verified_decode_rows==0):raise ValueError('INSUFFICIENT_MULTI_REQUEST_OR_DECODE_EXCITATION')
    groups=[];samples=[]
    roots={r:[f for f in w['forwards'] if 'root_ordinal' in f] for r,w in workers.items()}
    if len(roots[0])!=len(roots[1]):raise ValueError('Incomplete two-rank model/collective stream')
    ordinal=0
    for a,b in zip(roots[0],roots[1]):
        if a['root_ordinal']!=b['root_ordinal'] or len(a['layers'])!=16 or len(b['layers'])!=16:
            raise ValueError('Unmatched model/layer stream')
        members=[]
        for rank,f in enumerate((a,b)):
            batch=f['batch']
            if f['grants']:
                members += [dict(request_id=rid,rank=rank,local_rows=batch['bounds'][i:i+2],
                                 phase=external[expected[rid]].get('phase')) for i,rid in enumerate(batch['req_ids'])]
        if a['sample']!=b['sample']:raise ValueError('Asymmetric diagnostic sampling')
        real_count=sum(f['batch']['num_tokens'] if f['batch'] is not None else 0 for f in (a,b))
        for la,lb in zip(a['layers'],b['layers']):
            ordinal+=1
            for l in (la,lb):
                if l['dispatch']['observer_collective_ordinal']!=ordinal or not l['combine_returned'] or not l['combine_input_matches_expert_output']:
                    raise ValueError('Broken collective/dataflow stream')
            if la['layer']!=lb['layer'] or la['dp_sizes']!=lb['dp_sizes']:raise ValueError('Collective layer/layout mismatch')
            if not a['sample']:continue
            ids=la['topk_ids'];sizes=la['dp_sizes']
            if ids!=lb['topk_ids'] or len(ids)!=sum(sizes):raise ValueError('Two-rank routing payload mismatch')
            maps=[la['global_to_local_map'],lb['global_to_local_map']]
            if any(sum(m[e]>=0 for m in maps)!=1 for e in range(64)):raise ValueError('Expert placement ambiguity')
            real_rows=set()
            for rank,f in enumerate((a,b)):
                if f['batch'] is None:continue
                offset=sum(sizes[:rank]);n=f['batch']['num_tokens']
                if n>sizes[rank]:raise ValueError('Real rows exceed DP chunk')
                real_rows.update(range(offset,offset+n))
            counts=[0]*64;dummy_assignments=0
            for row,experts in enumerate(ids):
                if row in real_rows:
                    if len(experts)!=8 or len(set(experts))!=8 or any(e<0 or e>=64 for e in experts):raise ValueError('Invalid real-token experts')
                    for e in experts:counts[e]+=1
                else:dummy_assignments+=sum(e>=0 for e in experts)
            if sum(counts)!=real_count*8:raise ValueError('Real assignment conservation failure')
            layer_members=[]
            for member in members:
                start=sum(sizes[:member['rank']]);lo,hi=member['local_rows'];work=[0]*64
                for row in ids[start+lo:start+hi]:
                    for expert in row:work[expert]+=1
                layer_members.append(dict(**member,global_rows=[start+lo,start+hi],expert_assignments=work,
                    cost_scope='Request-owned assignment work only; shared GPU interval is not allocated per request'))
            samples.append(dict(root_ordinal=a['root_ordinal'],layer=la['layer'],members=layer_members,
                real_assignments=sum(counts),expert_counts=counts,
                rank_assignments=[sum(counts[e] for e in range(64) if m[e]>=0) for m in maps],
                active_experts=[sum(counts[e]>0 and m[e]>=0 for e in range(64)) for m in maps],
                dummy_padding_assignments=dummy_assignments,
                native_padded_tokens=[l.get('native_padded_tokens') for l in (la,lb)],
                tile_config=[l.get('tile_config') for l in (la,lb)]))
        if a['sample']:
            for rank,f in enumerate((a,b)):
                if not f['completion_verified_at_drain']:raise ValueError('Uncompleted sample')
                forward=f['forward_timing'];total=0
                for l in f['layers']:
                    for t in (l['expert_timing'],l['combine_timing'],l['dispatch']['gpu_timing']):
                        if t['status']!='OBSERVED_STREAM_INTERVAL' or t['stream']!=forward['stream'] or t['device']!=forward['device']:
                            raise ValueError('Unqualified timing interval')
                        if not math.isfinite(t['milliseconds']) or t['milliseconds']<0:raise ValueError('Invalid elapsed interval')
                        total+=t['milliseconds']
                if forward['status']!='OBSERVED_STREAM_INTERVAL' or not math.isfinite(forward['milliseconds']) or total>forward['milliseconds']:
                    raise ValueError('Invalid forward/component containment')
        groups.append(dict(root_ordinal=a['root_ordinal'],sampled=a['sample'],members=members,
                           host_submit_ns=[f['submit_monotonic_ns'] for f in (a,b)]))
    if require_multi and not samples:raise ValueError('No MoE diagnostic coverage')
    return dict(status='PASS_ACCOUNTING',requests=len(offers),normal_completions=len(terminals),
        actual_outputs=[len(c['token_ids']) for c in terminals],multi_request_batches=multi,
        sampled_decode_rows=decode_rows,groups=groups,MoE_samples=samples,
        retrospectively_verified_decode_rows=verified_decode_rows,unknown_decode_lookahead_rows=unknown_lookahead_rows,
        token_join_scope='Realized terminal tokens used only after execution to audit input rows; never prospective workload inputs',
        native_request_metrics=native_request_metrics,
        observation_limits='Local CUDA intervals; complete validated collective stream, not timestamp/local-step equality; unsampled routing UNKNOWN')


def phase_report(core,client,stage,start_ns,end_ns):
    events=[e for e in client if e.get('stage')==stage]
    offers=[e for e in events if e['event']=='submitted'];terms=[e for e in events if e['event']=='terminal']
    def pending(t):return sum(e['monotonic_ns']<=t for e in offers)-sum(e['monotonic_ns']<=t for e in terms)
    cuts=[int(start_ns+(end_ns-start_ns)*x/3) for x in (1,2,3)]
    backlog=[pending(t) for t in cuts]
    waiting=[];non_deferred=[];active_limit=[];token_limit=[];input_limit=[];active_guard=[]
    for r in (0,1):
        rows=[s for s in queue_samples(core,r) if start_ns<=s['time_ns']<=end_ns]
        waiting.append(max((len(set(s['waiting']+s['kv_holding_waiting'])) for s in rows),default=0))
        non_deferred.append(max((len(set(s['waiting']+s['kv_holding_waiting'])-set(s.get('deferred',[]))) for s in rows),default=0))
    schedules=[e for e in core if e['event']=='schedule' and start_ns<=e['monotonic_ns']<=end_ns]
    for e in schedules:
        s=e.get('load_before')
        if not s:continue
        q=set(s['waiting']+s['kv_holding_waiting'])
        if q and len(s['running'])>=s['limits']['active']:active_limit.append([e['dp_rank'],e['decision']])
        if e['budget_used']>=s['limits']['scheduled_tokens']:token_limit.append([e['dp_rank'],e['decision']])
        if e['budget_used']>=s['limits'].get('input_tokens',math.inf):input_limit.append([e['dp_rank'],e['decision']])
        after=e.get('load_after')
        if (after and set(after['waiting']+after['kv_holding_waiting']) and
            len(after['running'])+after.get('streaming_slots',0)>=after['limits']['active'] and
            e['budget_used']<after['limits']['scheduled_tokens'] and
            e['budget_used']<after['limits']['input_tokens'] and not e.get('preempted_req_ids') and
            after.get('pause_state_name')=='UNPAUSED'):
            active_guard.append([e['dp_rank'],e['decision']])
    failures=[e for e in core if e['event']=='allocation' and not e['accepted'] and start_ns<=e['monotonic_ns']<=end_ns]
    accepted=[e for e in core if e['event']=='request_accepted' and start_ns<=e['monotonic_ns']<=end_ns]
    end_terms=[e for e in terms if start_ns<=e['monotonic_ns']<=end_ns]
    rising=backlog[2]>backlog[1]>backlog[0]
    return dict(start_ns=start_ns,end_ns=end_ns,client_pending_at_thirds=backlog,
        actual_submissions=sum(start_ns<=e['monotonic_ns']<=end_ns for e in offers),
        native_acceptances=len(accepted),normal_completions=len(end_terms),
        maximum_observed_waiting_by_rank=waiting,
        maximum_observed_non_deferred_waiting_by_rank=non_deferred,
        observed_service_deficit=bool(rising and any(waiting) and len(accepted)>len(end_terms)),
        active_limit_boundary_evidence=active_limit,token_budget_exhausted_decisions=token_limit,
        input_budget_exhausted_decisions=input_limit,active_guard_condition_evidence=active_guard,
        active_guard_scope='Pinned waiting-loop guard conditions; not an exclusive physical bottleneck claim',
        allocation_refusals=len(failures),
        offer_lateness_ns=[e['monotonic_ns']-e['due_ns'] for e in offers if start_ns<=e['monotonic_ns']<=end_ns],
        bottleneck='UNKNOWN; nonexclusive constraints and physical samples must be interpreted together',
        criterion='Finite-horizon rising pending at three cuts plus native waiting and accept/completion deficit; not steady-state capacity')
