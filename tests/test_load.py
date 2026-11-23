"""Synthetic CPU falsification checks; no native/GPU execution claim."""
import asyncio
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from load_analysis import make_plan,validate_stage,phase_report,common_clock,bounded_trajectory
from load_experiment import scheduled_offers
from observation_contract import native_batch_record
from observation_worker import ObservationWorker
from load_run import retained_result

CLOCK=dict(verified=True,boot_id='SYNTHETIC_BOOT',time_namespace='SYNTHETIC_NS',implementation='SYNTHETIC_CLOCK')

def fixture():
    inputs=dict(identifiers=list(range(32)),narrative=list(range(100,132)))
    core=[];client=[];joins={};workers={}
    def stamp(event,rank,t,**kw):return dict(event=event,dp_rank=rank,monotonic_ns=t,clock_identity=CLOCK,**kw)
    for rank in (0,1):
        core.append(stamp('core_initialized',rank,0,pid=10+rank,free_blocks=100))
        rids=[f'internal-{rank}-{i}' for i in (0,1)]
        for i,rid in enumerate(rids):
            external=f'check-{rank}-{i}';family=['identifiers','narrative'][i]
            joins[external]=dict(internal_request_id=rid,prompt_token_ids=inputs[family])
            client += [stamp('submitted',-1,10,stage='check',request_id=external,family=family,owner=rank,due_ns=10),
                       stamp('terminal',-1,1000,stage='check',request_id=external,token_ids=[900,901],finish_reason='length')]
            core += [stamp('request_accepted',rank,20,request_id=rid,prompt_token_ids=inputs[family]),
                     stamp('allocation',rank,30,request_id=rid,accepted=True,after_blocks=[[dict(block_id=i,ref_cnt=1)]]),
                     stamp('kv_free',rank,800+i,request_id=rid,remaining_blocks=[[]],free_blocks=99+i,
                           released_blocks=[[dict(block_id=i,ref_cnt=0)]])]
        frames=[]
        for decision,(n,start) in enumerate(((32,0),(1,32)),1):
            grants={rid:n for rid in rids}
            core.append(stamp('schedule',rank,100*decision,decision=decision,grants=grants,
                before={rid:dict(num_computed_tokens=start) for rid in rids}))
            core.append(stamp('output_applied',rank,100*decision+50,decision=decision,grants=grants,
                after={} if decision==2 else {rid:dict(num_computed_tokens=32) for rid in rids},deferred_frees=0))
            ordered=rids[::-1]
            batch=dict(req_ids=ordered,bounds=[0,n,2*n],counts=[n,n],computed_cpu=[start,start],
                is_prefilling=[decision==1]*2,contexts_cpu_upper_bound=[32]*2,num_tokens=2*n,padded_tokens=2*n)
            frame=dict(root_ordinal=decision,grants=grants,batch=batch,sample=True,layers=[],submit_monotonic_ns=100*decision,
                completion_verified_at_drain=True,positions=list(range(start,start+n))*2,
                input_ids=(inputs['narrative']+inputs['identifiers']) if decision==1 else [900,900],
                forward_timing=dict(status='OBSERVED_STREAM_INTERVAL',milliseconds=1000,stream=7,device=rank))
            for layer in range(16):
                timing=dict(status='OBSERVED_STREAM_INTERVAL',milliseconds=1,stream=7,device=rank)
                frame['layers'].append(dict(layer='model.layers.'+str(layer),dp_sizes=[2*n,2*n],
                    topk_ids=[[0,1,2,3,32,33,34,35] for _ in range(4*n)],
                    global_to_local_map=[e-rank*32 if rank*32<=e<(rank+1)*32 else -1 for e in range(64)],
                    dispatch=dict(observer_collective_ordinal=(decision-1)*16+layer+1,gpu_timing=copy.copy(timing)),
                    combine_returned=True,combine_input_matches_expert_output=True,
                    expert_timing=copy.copy(timing),combine_timing=copy.copy(timing)))
            frames.append(frame)
        workers[rank]=dict(clock_identity=CLOCK,observed_parent_pid=10+rank,coverage=dict(overflow=False),forwards=frames)
    return core,workers,client,joins,inputs


class LoadAccountingTests(unittest.TestCase):
    def assess(self,data):return validate_stage('check',*data,8,require_multi=True)

    def test_actual_batch_reordering_and_decode_map_pass(self):
        result=self.assess(fixture())
        self.assertEqual(result['normal_completions'],4)
        self.assertEqual(result['multi_request_batches'],4)
        self.assertEqual(result['sampled_decode_rows'],4)
        self.assertEqual(len(result['MoE_samples']),32)

    def test_grant_dict_order_cannot_substitute_for_actual_input_order(self):
        data=fixture();data[1][0]['forwards'][0]['input_ids']=list(range(32))+list(range(100,132))
        with self.assertRaisesRegex(ValueError,'prompt token slice'):self.assess(data)

    def test_decode_token_from_wrong_request_cannot_pass(self):
        data=fixture();data[1][0]['forwards'][1]['input_ids'][0]=901
        with self.assertRaisesRegex(ValueError,'decode token/request'):self.assess(data)

    def test_wrong_owner_missing_terminal_and_orphan_are_rejected(self):
        data=fixture();next(e for e in data[0] if e['event']=='request_accepted')['dp_rank']=1
        with self.assertRaisesRegex(ValueError,'owner'):self.assess(data)
        data=fixture();data[2].pop()
        with self.assertRaisesRegex(ValueError,'terminal accounting'):self.assess(data)
        data=fixture();data[1][0]['forwards'][0]['grants']['orphan']=1
        with self.assertRaisesRegex(ValueError,'batch/grants'):self.assess(data)

    def test_kv_release_and_pending_cleanup_negative_controls(self):
        data=fixture();next(e for e in data[0] if e['event']=='kv_free')['remaining_blocks']=[[dict(block_id=1)]]
        with self.assertRaisesRegex(ValueError,'KV refs'):self.assess(data)
        data=fixture();next(e for e in data[0] if e['event']=='output_applied' and e['decision']==2)['deferred_frees']=1
        with self.assertRaisesRegex(ValueError,'cleanup'):self.assess(data)

    def test_aggregate_free_count_cannot_hide_wrong_live_block_ownership(self):
        data=fixture()
        allocations=[e for e in data[0] if e['event']=='allocation' and e['dp_rank']==0]
        allocations[1]['after_blocks']=copy.deepcopy(allocations[0]['after_blocks'])
        with self.assertRaisesRegex(ValueError,'Duplicate live KV owner'):self.assess(data)
        data=fixture();freed=next(e for e in data[0] if e['event']=='kv_free')
        data[0].append(copy.deepcopy(freed))
        with self.assertRaisesRegex(ValueError,'release'):self.assess(data)

    def test_incorrect_collective_ordinal_or_routing_cannot_pass(self):
        data=fixture();data[1][1]['forwards'][0]['layers'][0]['dispatch']['observer_collective_ordinal']=99
        with self.assertRaisesRegex(ValueError,'collective'):self.assess(data)
        data=fixture();data[1][1]['forwards'][0]['layers'][0]['topk_ids'][0][0]=10
        with self.assertRaisesRegex(ValueError,'routing payload'):self.assess(data)

    def test_invalid_stream_nan_and_incomplete_timing_fail(self):
        for key,value in [('stream',999),('milliseconds',float('nan')),('status','UNSUPPORTED')]:
            data=fixture();data[1][0]['forwards'][0]['layers'][0]['expert_timing'][key]=value
            with self.assertRaises(ValueError):self.assess(data)

    def test_cross_clock_and_missing_multi_request_proof_fail(self):
        data=fixture();data[1][0]['clock_identity']={**CLOCK,'time_namespace':'OTHER'}
        with self.assertRaisesRegex(ValueError,'clock'):self.assess(data)
        with self.assertRaisesRegex(ValueError,'clock'):common_clock([dict(verified=False)])

    def test_no_decode_excitation_stops_mapping_gate(self):
        data=fixture()
        data[0][:]=[e for e in data[0] if e.get('decision')!=2]
        for e in data[0]:
            if e['event']=='output_applied':e['after']={}
        for w in data[1].values():w['forwards']=w['forwards'][:1]
        for e in data[2]:
            if e['event']=='terminal':e['token_ids']=[900]
        with self.assertRaisesRegex(ValueError,'EXCITATION'):self.assess(data)

    def test_optional_padding_unknown_is_never_zero(self):
        result=self.assess(fixture())
        self.assertEqual(result['MoE_samples'][0]['native_padded_tokens'],[None,None])


class NativeBatchTests(unittest.TestCase):
    def batch(self):return types.SimpleNamespace(req_ids=['b','a'],query_start_loc_np=[0,2,3],
        num_scheduled_tokens=[2,1],num_computed_tokens_np=[4,8],is_prefilling_np=[True,False],
        seq_lens_cpu_upper_bound=[6,9],num_tokens=3,num_tokens_after_padding=4)

    def test_native_cpu_mapping_preserves_real_order_and_copy(self):
        b=self.batch();r=native_batch_record(b)
        self.assertEqual(r['req_ids'],['b','a']);self.assertEqual(r['bounds'],[0,2,3])
        b.req_ids.reverse();self.assertEqual(r['req_ids'],['b','a'])

    def test_duplicate_ids_or_bad_offsets_refused(self):
        b=self.batch();b.req_ids=['a','a']
        with self.assertRaises(ValueError):native_batch_record(b)
        b=self.batch();b.query_start_loc_np=[0,1,3]
        with self.assertRaises(ValueError):native_batch_record(b)

    def test_wrapper_returns_identical_native_batch(self):
        worker=ObservationWorker.__new__(ObservationWorker);batch=self.batch();calls=[]
        worker.model_runner=types.SimpleNamespace(prepare_inputs=lambda *a,**kw:(calls.append((a,kw)) or batch))
        worker._mechanistic_obs=dict(active={},episode='fixture')
        worker.observation_install=lambda *a:dict(dp_rank=0)
        worker.load_install('/SYNTHETIC_NO_GPU')
        token=object();returned=worker.model_runner.prepare_inputs(token)
        self.assertIs(returned,batch);self.assertEqual(calls,[((token,),{})])
        self.assertEqual(worker._mechanistic_obs['active']['batch']['req_ids'],['b','a'])


class OpenLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_offers_continue_while_all_responses_are_pending(self):
        current=[0.0];launched=[];events=[];finish=asyncio.Event();original_sleep=asyncio.sleep
        async def wait(delay):current[0]+=delay;await original_sleep(0)
        async def response():await finish.wait()
        def launch(item):launched.append((item['request_id'],current[0]));return asyncio.create_task(response())
        plan=make_plan('run',0,3,1)
        with patch('load_experiment.time.monotonic',lambda:current[0]),patch('load_experiment.asyncio.sleep',wait):
            tasks,stop=await scheduled_offers(plan,launch,lambda:0,lambda *a,**k:events.append((a,k)),32,100)
        self.assertEqual([v for _,v in launched],[0,1,2]);self.assertIsNone(stop)
        self.assertTrue(all(not t.done() for t in tasks))
        finish.set();await asyncio.gather(*tasks)

    async def test_safety_stop_records_unsent_without_waiting_rest_of_calendar(self):
        current=[0.0];tasks=[];events=[];original_sleep=asyncio.sleep
        async def wait(delay):current[0]+=delay;await original_sleep(0)
        async def response():await asyncio.Event().wait()
        def launch(item):t=asyncio.create_task(response());tasks.append(t);return t
        with patch('load_experiment.time.monotonic',lambda:current[0]),patch('load_experiment.asyncio.sleep',wait):
            _,stop=await scheduled_offers(make_plan('run',0,100,1),launch,lambda:len(tasks),lambda *a,**k:events.append((a,k)),2,200)
        self.assertEqual(stop,'PENDING_SAFETY_LIMIT');self.assertEqual(len(tasks),2)
        self.assertEqual(len(events),98);self.assertLess(current[0],4)
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)


class BoundsAndPressureTests(unittest.TestCase):
    def test_full_trajectory_is_budgeted_before_any_phase_is_started(self):
        probes=[dict(tested_rate=r,phase_reports=[dict(observed_service_deficit=False)]) for r in (1.3,2.6)]
        protocol=dict(phase_rate_factors=[.25,.75,1,.25],phase_min_seconds=12,drain_timeout_seconds=60)
        frozen=bounded_trajectory(probes,18,86,200,protocol)
        count=sum(len(make_plan('x',0,frozen['phase_seconds'],rate)) for rate in frozen['rates'])
        self.assertLessEqual(count,86)
        self.assertTrue(frozen['duration_limited_by_budget'])
        with self.assertRaisesRegex(ValueError,'NO_FULL_TRAJECTORY'):
            bounded_trajectory(probes,18,10,70,protocol)

    def test_truncated_native_result_does_not_lose_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'load-result.json').write_text('{')
            result=retained_result(out,dict(exit_code=-9))
            self.assertEqual(result['status'],'BLOCKED')
            self.assertEqual(result['process_exit_code'],-9)
            self.assertEqual((out/'load-result.json').read_text(),'{')

    def test_sampling_budget_refuses_gpu_copies_without_affecting_progress(self):
        worker=ObservationWorker.__new__(ObservationWorker)
        worker._mechanistic_obs=dict(load=dict(root=0,diagnostic=True,sample_every=8,samples=0,max_samples=16,trace_bytes=0,max_trace_bytes=1))
        frame={};worker._load_pre_forward(None,frame,{})
        self.assertFalse(frame['sample']);self.assertEqual(frame['root_ordinal'],1)

    def test_conservation_pressure_is_not_just_high_occupancy_or_pending(self):
        clock=CLOCK
        client=[dict(event='submitted',stage='x',monotonic_ns=t,due_ns=t,clock_identity=clock) for t in (1,4,5,7,8,9)]
        core=[dict(event='request_accepted',monotonic_ns=t,dp_rank=0) for t in (1,4,5,7,8,9)]
        state=dict(waiting=[],kv_holding_waiting=[],running=[],limits=dict(active=4,scheduled_tokens=512))
        core.append(dict(event='output_applied',dp_rank=0,monotonic_ns=5,load_state=state))
        self.assertFalse(phase_report(core,client,'x',0,10)['observed_service_deficit'])
        state['waiting']=['queued']
        self.assertTrue(phase_report(core,client,'x',0,10)['observed_service_deficit'])

    def test_active_guard_is_separate_from_token_and_allocation_limits(self):
        state=dict(waiting=['queued'],kv_holding_waiting=[],running=['a','b'],
            limits=dict(active=2,scheduled_tokens=512,input_tokens=512),pause_state='0',pause_state_name='UNPAUSED')
        core=[dict(event='schedule',monotonic_ns=5,dp_rank=0,decision=3,load_before=state,
                   load_after=state,budget_used=2,preempted_req_ids=[])]
        report=phase_report(core,[],'x',0,10)
        self.assertEqual(report['active_guard_condition_evidence'],[[0,3]])
        self.assertEqual(report['token_budget_exhausted_decisions'],[])
        self.assertEqual(report['allocation_refusals'],0)

    def test_bad_rates_refused_and_plan_is_reproducible(self):
        for rate in (0,-1,float('nan')):
            with self.assertRaises(ValueError):make_plan('x',0,1,rate)
        self.assertEqual(make_plan('x',100,1,4),make_plan('x',100,1,4))

    def test_unsampled_expert_wrapper_preserves_native_call_and_result(self):
        worker=ObservationWorker.__new__(ObservationWorker);calls=[];sentinel=object()
        output=types.SimpleNamespace(data_ptr=lambda:123,shape=(3,4),dtype='SYNTHETIC',device='SYNTHETIC')
        class CPU:
            def tolist(self):return [2,1]
        metadata=types.SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda:[2,1],num_tokens_across_dp_cpu=CPU())
        frame=dict(sample=False,layers=[],_dispatch={})
        arg=object();kw=dict(output=output)
        def native(*a,**k):calls.append((a,k));return sentinel
        returned=worker._load_expert(None,frame,'layer',[],native,(arg,),kw,lambda:types.SimpleNamespace(dp_metadata=metadata))
        self.assertIs(returned,sentinel);self.assertEqual(calls,[((arg,),kw)])


if __name__=='__main__':unittest.main()
