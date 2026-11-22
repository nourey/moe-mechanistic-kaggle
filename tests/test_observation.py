"""Synthetic CPU negative controls; never reported as GPU experiment results."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import struct
import types
import unittest
from unittest.mock import patch

from observation_contract import validate_episode, Writer, submission_work, verify_combine
from observation_worker import expert_target


def fixture(owner=0):
    rid = 'internal-random-id'
    join = dict(internal_request_id=rid, prompt_token_ids=[1, 2, 3, 4])
    core = [dict(event='core_initialized', dp_rank=r, free_blocks=100) for r in (0, 1)]
    core += [dict(event='request_accepted', request_id=rid, dp_rank=owner),
        dict(event='schedule', decision=1, grants={rid:4}, dp_rank=owner, before={rid:dict(num_computed_tokens=0)}),
        dict(event='output_applied', decision=1, grants={rid:4}, dp_rank=owner, after={}, deferred_frees=0),
        dict(event='allocation', request_id=rid, dp_rank=owner, accepted=True, after_blocks=[[dict(block_id=3,ref_cnt=1)]]),
        dict(event='kv_free', request_id=rid, dp_rank=owner, remaining_blocks=[[]], free_blocks=100,
             released_blocks=[[dict(block_id=3,ref_cnt=0)]])]
    sizes=[4,1] if owner == 0 else [1,4]
    ids=[[0,1,2,3,32,33,34,35] for _ in range(4)]
    ids = ids+[[-1]*8] if owner==0 else [[-1]*8]+ids
    workers={}
    for rank in (0,1):
        mapping=[e-rank*32 if rank*32<=e<(rank+1)*32 else -1 for e in range(64)]
        layers=[dict(layer='layer.'+str(i), dp_sizes=sizes, topk_ids=ids,
            global_to_local_map=mapping, output_abs_sums=[1]*5, cuda_completion_verified=True,
            combine_input_matches_expert_output=True, combine_returned=True,
            dispatch=dict(returned=True,gathered_shape=[5,2048])) for i in range(16)]
        layers[0].update(partial_output_file='SYNTHETIC.npy',combine_output_file='SYNTHETIC.npy')
        frame=dict(forward=13 if rank==owner else 2, grants={rid:4} if rank==owner else {},
                   dummy=rank!=owner, layers=layers, positions=[0,1,2,3] if rank==owner else [0],
                   input_ids=[1,2,3,4] if rank==owner else [0])
        workers[rank]=dict(forwards=[frame])
    return join, core, workers


class EpisodeTests(unittest.TestCase):
    def assess(self, data, owner=0): return validate_episode('external',owner,*data,
        numerical_check=lambda *args:dict(verified=True,source='SYNTHETIC CPU fixture, no GPU claim'))

    def test_two_owner_orders_and_different_local_forward_numbers(self):
        for owner in (0,1):
            r=self.assess(fixture(owner),owner)
            self.assertEqual(r['remote_expert_participation'],'PASS')
            self.assertEqual(len(r['links']),16)

    def test_native_dummy_positive_tokens_are_not_request_grants(self):
        native={'_dummy_req_0':1}
        row=submission_work(native,True)
        self.assertEqual(row['grants'],{})
        self.assertEqual(row['native_scheduled_tokens'],native)
        self.assertEqual(submission_work({'real':32},False)['grants'],{'real':32})

    def test_hint_does_not_override_authoritative_wrong_owner(self):
        data=fixture();data[1][2]['dp_rank']=1
        with self.assertRaisesRegex(ValueError,'ownership'):self.assess(data)

    def test_orphan_output_decision_and_missing_allocation_fail(self):
        data=fixture();data[1][4]['decision']=99
        with self.assertRaisesRegex(ValueError,'reconcile'):self.assess(data)
        data=fixture();data[1][5]['after_blocks']=[[]]
        with self.assertRaisesRegex(ValueError,'positive request-associated'):self.assess(data)

    def test_wrong_prompt_and_token_positions_fail(self):
        data=fixture();data[2][0]['forwards'][0]['input_ids'][0]=999
        with self.assertRaisesRegex(ValueError,'input differs'):self.assess(data)
        data=fixture();data[2][0]['forwards'][0]['positions']=[1,2,3,4]
        with self.assertRaisesRegex(ValueError,'worker-position join'):self.assess(data)

    def test_unreleased_refs_and_free_population_fail(self):
        data=fixture();data[1][-1]['released_blocks'][0][0]['ref_cnt']=1
        with self.assertRaisesRegex(ValueError,'retain references'):self.assess(data)
        data=fixture();data[1][-1]['free_blocks']=99
        with self.assertRaisesRegex(ValueError,'baseline'):self.assess(data)

    def test_idle_rank_cannot_secretly_own_work(self):
        data=fixture();data[2][1]['forwards'][0]['grants']={'other':1}
        with self.assertRaisesRegex(ValueError,'idle rank'):self.assess(data)

    def test_kernel_completion_without_tensor_dataflow_is_unverified(self):
        data=fixture()
        for layer in data[2][1]['forwards'][0]['layers']:layer['combine_input_matches_expert_output']=False
        self.assertEqual(self.assess(data)['remote_expert_participation'],'UNVERIFIED')

    def test_missing_numerical_evidence_is_not_full_participation_pass(self):
        self.assertEqual(validate_episode('external',0,*fixture())['remote_expert_participation'],'UNVERIFIED')

    def test_repeated_routing_cannot_fabricate_cross_rank_join(self):
        data=fixture();data[2][1]['forwards'].append(copy.deepcopy(data[2][1]['forwards'][0]))
        self.assertEqual(self.assess(data)['remote_expert_participation'],'UNVERIFIED')

    def test_padding_and_no_remote_excitation_not_pass(self):
        data=fixture()
        for rank in (0,1):
            for layer in data[2][rank]['forwards'][0]['layers']:
                layer['topk_ids']=[list(range(8))]*4+[list(range(32,40))]
        # Only the dummy row hits remote experts. It must not count as useful work.
        self.assertEqual(self.assess(data)['remote_expert_participation'],'INSUFFICIENT_EXCITATION')

    def test_source_map_duplication_and_invalid_real_experts_fail(self):
        data=fixture();data[2][1]['forwards'][0]['layers'][0]['global_to_local_map']=data[2][0]['forwards'][0]['layers'][0]['global_to_local_map']
        with self.assertRaisesRegex(ValueError,'placement'):self.assess(data)
        data=fixture()
        for rank in (0,1):data[2][rank]['forwards'][0]['layers'][0]['topk_ids'][0]=[-1]*8
        with self.assertRaisesRegex(ValueError,'Invalid real-token'):self.assess(data)

    def test_writer_preserves_local_event_identity_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            w=Writer(d,'core',0);a=w.emit('first');b=w.emit('second')
            self.assertEqual((a['seq'],b['seq']),(1,2))
            self.assertEqual(a['observation_instance'],b['observation_instance'])
            with self.assertRaises(FileExistsError):Writer(d,'core',0)
            w.stream.close()


class SchedulerProxyTest(unittest.TestCase):
    def test_native_methods_called_once_with_identical_objects_and_arguments(self):
        # Load the real observer mixin over explicitly synthetic native classes.
        class Native:
            def __init__(self,*a,**kw):
                self.parallel_config=types.SimpleNamespace(data_parallel_rank=0)
                self.scheduler_config=types.SimpleNamespace(async_scheduling=False)
                self.cache_config=types.SimpleNamespace(enable_prefix_caching=False)
                self.kv_cache_manager=types.SimpleNamespace(block_pool=types.SimpleNamespace(num_gpu_blocks=10,get_num_free_blocks=lambda:9),
                    get_blocks=lambda rid:types.SimpleNamespace(blocks=([],)), allocate_slots=lambda *a,**kw:None,free=lambda req:None)
                self.calls=[]
            def add_request(self,request):self.calls.append(request);return 'UNCHANGED'
        modules={n:types.ModuleType(n) for n in ['vllm','vllm.v1','vllm.v1.core','vllm.v1.core.sched','vllm.v1.core.sched.scheduler','vllm.v1.core.sched.async_scheduler']}
        modules['vllm.v1.core.sched.scheduler'].Scheduler=Native
        modules['vllm.v1.core.sched.async_scheduler'].AsyncScheduler=type('AsyncNative',(Native,),{})
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,modules),patch.dict('os.environ',{'MOE_OBSERVATION_ROOT':d}):
            spec=importlib.util.spec_from_file_location('synthetic_scheduler_observer',Path(__file__).resolve().parents[1]/'observation_scheduler.py')
            m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
            obj=m.ObservedScheduler()
            request=types.SimpleNamespace(request_id='r',status='WAITING',num_computed_tokens=0,num_in_flight_tokens=0,num_output_tokens=0,prompt_token_ids=[1])
            self.assertEqual(obj.add_request(request),'UNCHANGED');self.assertEqual(obj.calls,[request])
            obj.obs.stream.close()


class NumericalTest(unittest.TestCase):
    def save(self,path,shape,values):
        header=repr(dict(descr='<f2',fortran_order=False,shape=shape)).encode()+b'\n'
        path.write_bytes(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<'+str(len(values))+'e',*values))

    def test_real_half_payloads_exact_sum_and_omission_control(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            self.save(p/'a.npy',(3,2),[1,2,3,4,5,6]);self.save(p/'b.npy',(3,2),[2,3,4,5,6,7])
            self.save(p/'c.npy',(2,2),[7,9,11,13])
            a=dict(partial_output_file='a.npy',combine_output_file='c.npy');b=dict(partial_output_file='b.npy')
            result=verify_combine(p,a,b,1,2)
            self.assertTrue(result['verified']);self.assertEqual(result['remote_omission_negative_control_mismatches'],4)
            self.save(p/'c.npy',(2,2),[3,4,5,6])
            self.assertFalse(verify_combine(p,a,b,1,2)['verified'])


class TapBoundaryTest(unittest.TestCase):
    def test_selects_real_post_dispatch_expert_not_already_combined_wrapper(self):
        noop=type('TopKWeightAndReduceNoOP',(),{})
        experts=type('TritonExperts',(),{'finalize_weight_and_reduce_impl':lambda self:noop()})()
        impl=type('FusedMoEKernelModularImpl',(),{})()
        impl.prepare_finalize=type('MoEPrepareAndFinalizeNaiveDPEPModular',(),{})()
        impl.fused_experts=experts
        layer=types.SimpleNamespace(quant_method=types.SimpleNamespace(moe_kernel=types.SimpleNamespace(impl=impl)))
        self.assertIs(expert_target(layer),experts)
        impl.prepare_finalize=object()
        with self.assertRaisesRegex(ValueError,'prepare/finalize'):expert_target(layer)


if __name__=='__main__':unittest.main()
