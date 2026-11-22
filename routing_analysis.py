"""Bounded post-gate routing selection and descriptive analysis, no prediction fitting."""
import hashlib
import itertools
import json
from pathlib import Path

from observation_timing import summarize


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def layer_number(name): return int(name.split('.layers.',1)[1].split('.',1)[0])


def serving_summary(name, join, core, workers, prompt):
    rid=join['internal_request_id'];owner=0
    accepted=[e for e in core if e['event']=='request_accepted' and e['request_id']==rid]
    if len(accepted)!=1 or accepted[0]['dp_rank']!=owner:raise ValueError('Incorrect actual DP0 ownership')
    grants=[e for e in core if e['event']=='schedule' and rid in e['grants']]
    applied=[e for e in core if e['event']=='output_applied' and rid in e['grants']]
    if len(grants)!=1 or grants[0]['grants']!={rid:32} or len(applied)!=1 or grants[0]['decision']!=applied[0]['decision']:
        raise ValueError('Not one comparable 32-token prompt decision')
    if grants[0]['before'][rid]['num_computed_tokens']!=0:raise ValueError('Not fresh prompt state')
    frees=[e for e in core if e['event']=='kv_free' and e['request_id']==rid]
    initial=next(e for e in core if e['event']=='core_initialized' and e['dp_rank']==owner)
    if len(frees)!=1 or any(frees[0]['remaining_blocks']) or frees[0]['free_blocks']!=initial['free_blocks']:
        raise ValueError('KV request ownership not fully released')
    if any(b['ref_cnt']!=0 for group in frees[0]['released_blocks'] for b in group):raise ValueError('Live refs after free')
    if applied[0]['after'] or applied[0]['deferred_frees']:raise ValueError('Episode not clean')
    real=[f for f in workers[owner]['forwards'] if f['grants'].get(rid)==32]
    if len(real)!=1 or any(f['grants'] for f in workers[1]['forwards']):raise ValueError('Worker request ownership mismatch')
    frame=real[0]
    if workers[owner]['mode']!='control' and (frame['positions']!=list(range(32)) or frame['input_ids']!=prompt):
        raise ValueError('Worker token join mismatch')
    return dict(request_id=rid,verified_owner=owner,core_pid=accepted[0]['pid'],
        grants=[32],computed_before=0,KV_initial_free=initial['free_blocks'],
        KV_released_blocks=sum(len(g) for g in frees[0]['released_blocks']),clean_after=True,
        worker_input_join='UNOBSERVED_CONTROL' if workers[owner]['mode']=='control' else 'VERIFIED'),frame


def routing_summary(frame,remote):
    result=[]
    for l in sorted(frame['layers'],key=lambda x:layer_number(x['layer'])):
        sizes=l['dp_sizes'];ids=l['topk_ids']
        if sizes!=[32,1] or len(ids)!=33:raise ValueError('Unexpected native DP layout')
        real=ids[:32]
        if any(len(row)!=8 or len(set(row))!=8 or any(e<0 or e>=64 for e in row) for row in real):
            raise ValueError('Invalid real-token routing')
        matches=[rl for rf in remote['forwards'] for rl in rf['layers'] if rf['dummy'] and rl['layer']==l['layer'] and rl['dp_sizes']==sizes and rl['topk_ids']==ids]
        if len(matches)!=1:raise ValueError('Ambiguous cross-rank routing/layout join')
        peer=matches[0];maps=[l['global_to_local_map'],peer['global_to_local_map']]
        if not all(r['cuda_completion_verified'] and r['combine_returned'] for r in (l,peer)):
            raise ValueError('Native execution/combine completion not observed')
        if any(sum(m[e]>=0 for m in maps)!=1 for e in range(64)):raise ValueError('Expert ownership contradiction')
        counts=[sum(row.count(e) for row in real) for e in range(64)]
        by_rank=[sum(counts[e] for e in range(64) if m[e]>=0) for m in maps]
        result.append(dict(layer=l['layer'],expert_counts=counts,rank_counts=by_rank,
            real_assignments=sum(counts),active_experts=sum(n>0 for n in counts),
            maximum_expert_fraction=max(counts)/256,HHI=sum((n/256)**2 for n in counts),
            routing_sha256=digest(real),GPU_instruction_count='NOT_OBSERVED',
            dispatch_logical_bytes=[r['dispatch']['logical_input_bytes'] for r in (l,peer)],
            combine_logical_input_bytes=[r.get('combine_logical_input_bytes') for r in (l,peer)],
            communication_scope='Observed logical shapes/bytes and native calls; not wire traffic or pure network delay'))
    if len(result)!=16:raise ValueError('Missing routed layers')
    return dict(layers=result,total_assignments=sum(l['real_assignments'] for l in result),
                fixed_rank_assignments=[sum(l['rank_counts'][r] for l in result) for r in (0,1)])


def route_distance(a,b):
    la=a['layers'];lb=b['layers']
    if [x['layer'] for x in la]!=[x['layer'] for x in lb]:raise ValueError('Layer order mismatch')
    expert=[];rank=[]
    for x,y in zip(la,lb):
        expert.append(sum(abs(i-j) for i,j in zip(x['expert_counts'],y['expert_counts']))/(2*256))
        rank.append(sum(abs(i-j) for i,j in zip(x['rank_counts'],y['rank_counts']))/(2*256))
    return dict(mean_layer_expert_TV=sum(expert)/16,mean_layer_rank_TV=sum(rank)/16,
                per_layer_expert_TV=expert,per_layer_rank_TV=rank,
                aggregate_rank_TV=sum(abs(i-j) for i,j in zip(a['fixed_rank_assignments'],b['fixed_rank_assignments']))/(2*4096))


def select_pair(screens):
    pairs=[]
    for a,b in itertools.combinations(sorted(screens),2):pairs.append(dict(a=a,b=b,**route_distance(screens[a],screens[b])))
    # No timings, terminal text or execution outcomes enter this ordering.
    pairs.sort(key=lambda p:(-p['mean_layer_expert_TV'],-p['mean_layer_rank_TV'],p['a'],p['b']))
    best=pairs[0] if pairs and pairs[0]['mean_layer_expert_TV']>0 else None
    return best,pairs


def timing_summary(frame,remote,qualification):
    rows={0:frame}
    first=frame['layers'][0] if frame['layers'] else None
    matches=[f for f in remote['forwards'] if first and any(l['layer']==first['layer'] and l['topk_ids']==first['topk_ids'] and l['dp_sizes']==first['dp_sizes'] for l in f['layers'])]
    if len(matches)==1:rows[1]=matches[0]
    reports={}
    for rank,f in rows.items():
        valid=qualification[rank].get('usable',False)
        forward=f.get('forward_timing')
        if forward is None:valid=False
        values={k:[] for k in ('expert','dispatch','combine')}
        for l in f['layers']:
            ts={'expert':l.get('expert_timing'),'dispatch':(l.get('dispatch') or {}).get('gpu_timing'),'combine':l.get('combine_timing')}
            for key,t in ts.items():
                if not t or t['status']!='OBSERVED_STREAM_INTERVAL':valid=False
                if t and forward and (t['stream']!=forward['stream'] or t.get('device')!=forward.get('device')):valid=False
                values[key].append(t.get('milliseconds') if t else None)
        if forward and forward['status']!='OBSERVED_STREAM_INTERVAL':valid=False
        if any(len(v)!=16 or any(x is None for x in v) for v in values.values()):valid=False
        if valid and sum(sum(v) for v in values.values())>forward['milliseconds']:valid=False
        reports[rank]=dict(status='QUALIFIED_WITH_LIMITS' if valid else 'UNSUPPORTED',
            forward_stream_ms=forward['milliseconds'] if valid else None,
            layers=[l['layer'] for l in f['layers']],
            per_layer_ms=values,
            sums_ms={k:sum(v) if valid else None for k,v in values.items()},
            limits='Local same-device stream intervals; observed waits/enqueue gaps included; no global group clock or critical path')
    return reports


def compare_report(episodes,selected):
    measured=[e for e in episodes if not e.get('warmup')]
    report=dict(conditions=selected,timing={},host_elapsed_ns={},incremental_overhead_proxy_ns={},terminal_outcomes={},null_results=[],
        design='Three blocks, two held-out prompt variants, timed/diagnostic modes plus ledger-only overhead controls',
        confounders=['Prompt content changes activations, not just routing','Three repeats, no inferential power/equivalence claim',
                    'GPU clocks/cache/thermal state and common ledger overhead not controlled',
                    'CUDA intervals include queue and peer waits; they are not pure network/kernel duration'])
    for family in selected:
        rows=[e for e in measured if e['family']==family]
        report['terminal_outcomes'][family]=[dict(mode=e['mode'],block=e['block'],output_tokens=len(e['terminal']['token_ids']),finish_reason=e['terminal']['finish_reason']) for e in rows]
        report['host_elapsed_ns'][family]={mode:summarize([e['client_elapsed_ns'] for e in rows if e['mode']==mode]) for mode in ('control','timed','diagnostic')}
        paired=[]
        for block in (1,2,3):
            values={e['mode']:e['client_elapsed_ns'] for e in rows if e['block']==block}
            paired.append(dict(block=block,timed_minus_control=values['timed']-values['control'],
                diagnostic_minus_control=values['diagnostic']-values['control'],diagnostic_minus_timed=values['diagnostic']-values['timed']))
        report['incremental_overhead_proxy_ns'][family]=paired
        report['timing'][family]={rank:{key:summarize([(e.get('timing',{}).get(str(rank),e.get('timing',{}).get(rank,{}))).get('sums_ms',{}).get(key)
            for e in rows if e['mode']=='timed' and e.get('timing',{}).get(str(rank),e.get('timing',{}).get(rank,{})).get('status')=='QUALIFIED_WITH_LIMITS'])
            for key in ('expert','dispatch','combine')} for rank in (0,1)}
    a=[e['routing'] for e in measured if e['family']==selected[0] and e['mode']=='timed']
    b=[e['routing'] for e in measured if e['family']==selected[1] and e['mode']=='timed']
    report['heldout_routing_distances']=[route_distance(x,y) for x,y in zip(a,b)]
    observed_routes=[e['routing'] for e in measured if e['mode']!='control']
    patterns=[[(l.get('dispatch_logical_bytes'),l.get('combine_logical_input_bytes')) for l in r['layers']] for r in observed_routes]
    available=all(all(v is not None and all(x is not None for x in v) for pair in p for v in pair) for p in patterns)
    report['logical_communication_bytes_constant']=len({digest(p) for p in patterns})==1 if available else None
    shapes=[(e['serving']['grants'],e['serving']['computed_before'],e['serving']['KV_released_blocks'],e['serving']['KV_initial_free']) for e in measured]
    report['serving_shape_constant']=len({digest(s) for s in shapes})==1
    if not any(d['mean_layer_expert_TV']>0 for d in report['heldout_routing_distances']):report['null_results'].append('Screening contrast did not reproduce on held-out input variants')
    report['timing_coverage']='SUPPORTED_SUBSET' if any(v['n'] for r in report['timing'].values() for p in r.values() for v in p.values()) else 'UNSUPPORTED'
    report['timing_complete']=all(v['n']==3 for r in report['timing'].values() for p in r.values() for v in p.values())
    report['overhead_limits']='Paired client-window differences, signed and uncorrected. Common core ledger/CPU hook overhead unknown. Control routing unobserved, never imputed from other modes.'
    differences={}
    for rank in (0,1):
        differences[rank]={}
        for key in ('expert','dispatch','combine'):
            x=report['timing'][selected[0]][rank][key];y=report['timing'][selected[1]][rank][key]
            differences[rank][key]=dict(median_B_minus_A_ms=y['median']-x['median'] if x['n'] and y['n'] else None,
                observed_ranges_overlap=max(x['minimum'],y['minimum'])<=min(x['maximum'],y['maximum']) if x['n'] and y['n'] else None,
                interpretation='Descriptive three-repeat ranges, not CI/equivalence or causal attribution')
    report['component_differences']=differences
    report['interpretation']='Descriptive routing/physical-interval evidence; no causal latency attribution, essential-state or BLIS accuracy claim'
    return report
