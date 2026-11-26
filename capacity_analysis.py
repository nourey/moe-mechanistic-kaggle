"""Per-request cohort accounting and read-only native evidence joins, no fitting."""
from collections import Counter
import math
import statistics

from capacity_contract import validate_slo
from load_analysis import common_clock


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    def q(p):
        k=(len(values)-1)*p;i=int(k)
        return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(k-i)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p90=q(.9), p95=q(.95), maximum=max(values))


def request_metrics(plan, events, t0, end, slo, followup_deadline_ns=None):
    """Denominator is ALL planned offers; no post-hoc exclusion of short/failed requests."""
    validate_slo(slo)
    by_id={r['request_id']:r for r in plan}
    if len(by_id)!=len(plan):raise ValueError('Duplicate planned request')
    records=[]
    relevant=[e for e in events if e.get('request_id') in by_id]
    for rid,row in by_id.items():
        selected=[e for e in relevant if e['request_id']==rid]
        offers=[e for e in selected if e['event']=='submitted']
        terms=[e for e in selected if e['event']=='terminal']
        if len(offers)>1 or len(terms)>1:raise ValueError('Duplicate submission/terminal')
        errors=[e for e in selected if e['event'] in ('request_error','censored','unsent')]
        result=dict(request_id=rid,group=row['group'],owner=row['owner'],planned_offset_ns=row['offset_ns'],
                    status='UNSENT' if not offers else 'CENSORED',ttft_seconds=None,mean_tpot_seconds=None,
                    end_to_end_seconds=None,output_tokens=None,slo='FAIL' if any(v is not None for v in slo.values()) else 'UNDEFINED',metric_provenance='Frontend cumulative chunk receipt timestamps; native acceptance is separate')
        if offers:
            o=offers[0];result['submitted_ns']=o['monotonic_ns']
            if o['monotonic_ns']<t0+row['offset_ns']:raise ValueError('Early offer')
            if any(o[k]!=row[k] for k in ('group','owner','family')):raise ValueError('Dispatch binding changed')
            result['offer_lateness_ns']=o['monotonic_ns']-(t0+row['offset_ns'])
        if terms and not offers:raise ValueError('Orphan terminal')
        if terms:
            t=terms[0];o=offers[0]
            if t['monotonic_ns']<o['monotonic_ns']:raise ValueError('Terminal time precedes offer')
            tokens=t['token_ids'];normal=t['finish_reason'] in ('stop','length') and 0<len(tokens)<=8
            result.update(status='NORMAL' if normal else 'FAILED',terminal_ns=t['monotonic_ns'],
                          output_tokens=len(tokens),token_ids=tokens,
                          end_to_end_seconds=(t['monotonic_ns']-o['monotonic_ns'])/1e9)
            chunks=[e for e in selected if e['event']=='output_chunk' and e['token_ids']]
            previous=[];single=True;previous_time=o['monotonic_ns'];increments=[]
            for chunk in chunks:
                current=chunk['token_ids']
                if current[:len(previous)]!=previous or chunk['monotonic_ns']<previous_time or chunk['monotonic_ns']>t['monotonic_ns']:
                    raise ValueError('Non-cumulative or wrongly ordered chunk stream')
                if len(current)>len(previous):
                    single &= len(current)-len(previous)==1
                    increments.append(chunk)
                previous=current
                previous_time=chunk['monotonic_ns']
            if normal and (not chunks or previous!=tokens):raise ValueError('Missing terminal/token stream join')
            if increments:
                result['ttft_seconds']=(increments[0]['monotonic_ns']-o['monotonic_ns'])/1e9
                result['mean_tpot_chunk_proxy_seconds']=(increments[-1]['monotonic_ns']-increments[0]['monotonic_ns'])/1e9/(len(tokens)-1) if len(tokens)>1 else None
                if single and len(tokens)>1:result['mean_tpot_seconds']=result['mean_tpot_chunk_proxy_seconds']
                result['token_timing_coverage']='ONE_TOKEN_PER_NONEMPTY_INCREMENT' if single else 'COALESCED; per-token timing UNKNOWN'
            if not any(v is not None for v in slo.values()):result['slo']='UNDEFINED'
            elif normal:
                required=[(key,value) for key,value in slo.items() if value is not None]
                values=[result.get(key) for key,_ in required]
                result['slo']=('FAIL' if any(result.get(key) is not None and result[key]>limit for key,limit in required)
                               else 'UNKNOWN' if any(v is None for v in values) else 'PASS')
            if followup_deadline_ns is not None and t['monotonic_ns']>followup_deadline_ns:
                result.update(status='CENSORED_LATE_TERMINAL',slo='FAIL' if any(v is not None for v in slo.values()) else 'UNDEFINED',failure_reason='Beyond fixed follow-up')
        elif errors:result['failure_reason']=errors[-1].get('reason',errors[-1]['event'])
        records.append(result)
    return records


def summarize(records, duration_ns, t0, slo):
    defined=validate_slo(slo);seconds=duration_ns/1e9
    statuses=Counter(r['status'] for r in records);normal=[r for r in records if r['status']=='NORMAL']
    known_pass=sum(r['slo']=='PASS' for r in records)
    unknown=sum(r['slo']=='UNKNOWN' for r in records)
    last=max((r['terminal_ns'] for r in normal),default=t0)
    return dict(planned=len(records),statuses=dict(statuses),normal_completions=len(normal),
      offered_requests_per_second=len(records)/seconds,completed_within_offer_window=sum(r['terminal_ns']<=t0+duration_ns for r in normal),
      completion_rate_during_offer_window=sum(r['terminal_ns']<=t0+duration_ns for r in normal)/seconds,
      cohort_completion_rate=len(normal)/((last-t0)/1e9) if last>t0 else None,
      slo_goodput=dict(status='DEFINED' if defined else 'UNDEFINED_OWNER_THRESHOLDS',denominator='All frozen planned offers; fixed follow-up included',
                        verified_passes=known_pass if defined else None,unknown=unknown if defined else None,
                        verified_requests_per_second=known_pass/seconds if defined else None,
                        possible_upper_requests_per_second=(known_pass+unknown)/seconds if defined else None),
      end_to_end_seconds=distribution([r['end_to_end_seconds'] for r in normal]),
      ttft_seconds=distribution([r['ttft_seconds'] for r in normal if r['ttft_seconds'] is not None]),
      mean_tpot_seconds=distribution([r['mean_tpot_seconds'] for r in normal if r['mean_tpot_seconds'] is not None]),
      drain_seconds=max(0,last-(t0+duration_ns))/1e9,
      scope='Finite offered cohort, not a steady-state capacity estimate; requests within one run are dependent')


def native_accounting(stage, rows, core, joins, dp, inputs, diagnostic=False):
    wanted={joins[r['request_id']]['internal_request_id']:r for r in rows}
    accepted=[e for e in core if e['event']=='request_accepted' and e['request_id'] in wanted]
    if len(accepted)!=len(wanted) or {e['request_id'] for e in accepted}!=set(wanted):raise ValueError('Native acceptance missing or duplicate')
    for e in accepted:
        row=wanted[e['request_id']]
        if e['dp_rank']!=row['owner'] or e['prompt_token_ids']!=inputs[row['family']]:raise ValueError('Actual rank/prompt differs from fixed binding')
        init=next(x for x in core if x['event']=='core_initialized' and x['dp_rank']==e['dp_rank'])
        if e['pid']!=init['pid']:raise ValueError('Core process incarnation changed')
    if not diagnostic:return dict(owners='PASS',scheduler_KV='UNKNOWN_REFERENCE')
    rank_reports=[]
    for rank in range(dp):
        owned={rid for rid,r in wanted.items() if r['owner']==rank}
        init=next(e for e in core if e['event']=='core_initialized' and e['dp_rank']==rank)
        live={};owners={}
        def blocks(groups):return {(i,b['block_id']) for i,g in enumerate(groups) for b in g}
        relevant=[e for e in core if e['dp_rank']==rank and e.get('request_id') in owned]
        for e in sorted((e for e in relevant if e['event'] in ('allocation','kv_free')),key=lambda e:e.get('seq',e['monotonic_ns'])):
            rid=e['request_id']
            if e['event']=='allocation':
                before=live.get(rid,set());after=blocks(e['after_blocks'])
                if blocks(e['before_blocks'])!=before or before-after:raise ValueError('KV ownership transition mismatch')
                if not e['accepted'] and before!=after:raise ValueError('Failed allocation mutated KV')
                for b in after:
                    if b in owners and owners[b]!=rid:raise ValueError('Duplicate live KV owner')
                    owners[b]=rid
                live[rid]=after
            elif e['event']=='kv_free':
                if blocks(e['released_blocks'])!=live.get(rid,set()):raise ValueError('Unmatched KV release')
                for b in live.get(rid,set()):
                    if owners.pop(b,None)!=rid:raise ValueError('Wrong KV release owner')
                live[rid]=blocks(e['remaining_blocks'])
                if any(b['ref_cnt'] for g in e['released_blocks'] for b in g):raise ValueError('Live KV reference after release')
        if owners or any(live.values()):raise ValueError('Unreleased request KV')
        freed={e['request_id'] for e in relevant if e['event']=='kv_free'}
        if freed!=owned:raise ValueError('Missing request KV release')
        frees=[e for e in relevant if e['event']=='kv_free']
        if frees and frees[-1]['free_blocks']!=init['free_blocks']:raise ValueError('KV pool did not return to baseline')
        last=[e for e in core if e['dp_rank']==rank and e['event']=='output_applied' and owned.intersection(e['grants'])]
        if owned and (not last or owned.intersection(last[-1]['after']) or last[-1]['deferred_frees']):raise ValueError('Request/deferred membership remains after drain')
        rank_reports.append(dict(rank=rank,requests=len(owned),initial_free=init['free_blocks']))
    return dict(owners='PASS',scheduler_KV='PASS_REQUEST_OWNERSHIP',ranks=rank_reports)


def pre_boundary(core, boundary_ns, dp):
    ranks=[]
    for rank in range(dp):
        candidates=[e for e in core if e['dp_rank']==rank and e['monotonic_ns']<=boundary_ns and ('load_after' in e or 'load_state' in e)]
        if not candidates:ranks.append(dict(rank=rank,status='UNKNOWN_NO_PUBLISHED_SNAPSHOT'));continue
        e=max(candidates,key=lambda e:e['monotonic_ns']);s=e.get('load_after',e.get('load_state'))
        ranks.append(dict(rank=rank,status='OBSERVED_PUBLISHED_BEFORE_BOUNDARY',publication_ns=e['monotonic_ns'],
                          age_ns=boundary_ns-e['monotonic_ns'],state=s))
    return dict(ranks=ranks,coherence='Separate rank-local publication snapshots, not globally atomic',
                routing_features='UNAVAILABLE_ONLINE: GPU routing clones are exported after drain; retrospective diagnosis only',
                future_output_lengths='NOT_USED')


def merge_group_records(plan, groups):
    records=[r for g in groups for r in g['records']]
    expected={r['request_id']:r for r in plan['arrivals']}
    if len(records)!=len(expected) or {r['request_id'] for r in records}!=set(expected):raise ValueError('Global duplicate/orphan/missing request record')
    for r in records:
        if (r['group'],r['owner'])!=(expected[r['request_id']]['group'],expected[r['request_id']]['owner']):raise ValueError('Cross-group binding changed')
    return sorted(records,key=lambda r:r['request_id'])


def contrasts(trials):
    """All prespecified pairs/repetitions; never rank/select favorable outcomes."""
    rows=[]
    for repeat in sorted({t['repeat'] for t in trials}):
        selected={t['arm']:t for t in trials if t['repeat']==repeat}
        for left,right in [('A','B'),('B','C1'),('C0','C1')]:
            if left not in selected or right not in selected:continue
            a,b=selected[left],selected[right]
            if a['workload_sha256']!=b['workload_sha256'] or a['mode']!=b['mode']:raise ValueError('Unmatched comparison')
            aa,bb=a['summary'],b['summary']
            gp_a,gp_b=aa['slo_goodput'],bb['slo_goodput']
            rows.append(dict(repeat=repeat,left=left,right=right,mode=a['mode'],
                mean_residence_delta=(bb['end_to_end_seconds']['mean']-aa['end_to_end_seconds']['mean']) if aa['end_to_end_seconds'] and bb['end_to_end_seconds'] else None,
                normal_completion_delta=bb['normal_completions']-aa['normal_completions'],
                verified_slo_goodput_delta=(gp_b['verified_requests_per_second']-gp_a['verified_requests_per_second']) if gp_a['status']=='DEFINED' and gp_b['status']=='DEFINED' and gp_a['unknown']==gp_b['unknown']==0 else None,
                slo_goodput_delta_interval=[gp_b['verified_requests_per_second']-gp_a['possible_upper_requests_per_second'],gp_b['possible_upper_requests_per_second']-gp_a['verified_requests_per_second']] if gp_a['status']=='DEFINED' and gp_b['status']=='DEFINED' else None,
                qualifications='A->B is composite DP/EP/layout change; B->C also changes residency. C0->C1 holds four GPUs/groups resident. Diagnostic timings are not primary performance.'))
    return dict(pairs=rows,inference='Descriptive independent-run contrasts; no statistical power/benefit threshold invented, no BLIS fitting')
