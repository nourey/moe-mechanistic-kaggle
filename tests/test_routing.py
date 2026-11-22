"""CPU tests for fixed routing selection, freeze fences and timing interpretation."""
import asyncio
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from observation_timing import finish_interval,summarize
from routing_analysis import route_distance,select_pair,timing_summary,routing_summary,compare_report
from routing_experiment import frozen_sequence,run,code_hashes
import routing_run


def profile(expert):
    rows=[]
    for i in range(16):
        counts=[0]*64;counts[expert]=256
        rows.append(dict(layer='model.layers.'+str(i)+'.experts',expert_counts=counts,rank_counts=[256,0] if expert<32 else [0,256],
            dispatch_logical_bytes=[10,2],combine_logical_input_bytes=[12,12]))
    return dict(layers=rows,fixed_rank_assignments=[4096,0] if expert<32 else [0,4096])


def timing_frame():
    def t(ms):return dict(status='OBSERVED_STREAM_INTERVAL',milliseconds=ms,stream=7)
    ids=[list(range(8)) for _ in range(32)]+[[-1]*8]
    layers=[dict(layer='model.layers.'+str(i)+'.experts',topk_ids=copy.deepcopy(ids),dp_sizes=[32,1],
        global_to_local_map=list(range(32))+[-1]*32,cuda_completion_verified=True,combine_returned=True,
        expert_timing=t(.02),combine_timing=t(.01),dispatch=dict(gpu_timing=t(.01),logical_input_bytes=10)) for i in range(16)]
    frame=dict(layers=layers,forward_timing=t(1.0))
    peer=copy.deepcopy(frame)
    for l in peer['layers']:l['global_to_local_map']=[-1]*32+list(range(32))
    peer['dummy']=True
    return frame,dict(forwards=[peer])


class SelectionTests(unittest.TestCase):
    def test_all_pairs_kept_and_no_timing_used(self):
        screens={'a':profile(0),'b':profile(1),'c':profile(32)}
        best,pairs=select_pair(screens)
        self.assertEqual(len(pairs),3);self.assertEqual((best['a'],best['b']),('a','c'))
        for name,row in screens.items():row['client_elapsed_ns']=1000000 if name=='a' else -100000
        self.assertEqual(select_pair(screens),(best,pairs))

    def test_no_contrast_is_not_fabricated(self):
        best,pairs=select_pair({'a':profile(0),'b':profile(0)})
        self.assertIsNone(best);self.assertEqual(pairs[0]['mean_layer_expert_TV'],0)

    def test_equal_aggregate_does_not_erase_layer_pattern(self):
        a=profile(0);b=profile(0)
        for i in range(16):
            a['layers'][i]=profile(0 if i%2 else 32)['layers'][i]
            c=[0]*64;c[0]=c[32]=128
            b['layers'][i].update(expert_counts=c,rank_counts=[128,128])
        a['fixed_rank_assignments']=b['fixed_rank_assignments']=[2048,2048]
        d=route_distance(a,b)
        self.assertEqual(d['aggregate_rank_TV'],0);self.assertEqual(d['mean_layer_rank_TV'],.5)

    def test_layer_sequence_mismatch_rejected(self):
        a=profile(0);b=profile(1);b['layers'].reverse()
        with self.assertRaises(ValueError):route_distance(a,b)

    def test_fixed_matrix_is_only_eighteen_main_runs(self):
        seq=frozen_sequence(['a','b'])
        self.assertEqual(len(seq),18)
        for family in ['a','b']:
            for mode in ['control','timed','diagnostic']:
                self.assertEqual(sum(r['family']==family and r['mode']==mode for r in seq),3)

    def test_input_variants_and_caps_are_predeclared(self):
        p=json.loads((Path(__file__).resolve().parents[1]/'routing_inputs.json').read_text())
        self.assertEqual(len(p['families']),6)
        self.assertTrue(all(f['screen']!=f['evaluate'] for f in p['families']))
        self.assertEqual((p['prompt_tokens'],p['max_tokens'],p['owner']),(32,1,0))


class TimingTests(unittest.TestCase):
    def test_unknown_not_zero_and_no_arbitrary_significance_score(self):
        self.assertEqual(summarize([None])['n'],0);self.assertIsNone(summarize([])['median'])
        self.assertEqual(summarize([1,2,3])['MAD'],1)

    def test_nan_negative_not_ready_and_stream_mismatch_are_unsupported(self):
        class Event:
            def __init__(self,ready=True,value=1):self.ready=ready;self.value=value
            def query(self):return self.ready
            def elapsed_time(self,other):return self.value
        for value,ready,same in [(float('nan'),True,True),(-1,True,True),(1,False,True),(1,True,False)]:
            row=finish_interval(dict(_start=Event(ready,value),_end=Event(),stream=7,native_stream_matches=same,end_stream_matches=True,end_device_matches=True))
            self.assertEqual(row['status'],'UNSUPPORTED');self.assertIsNone(row['milliseconds'])

    def test_only_same_device_valid_elapsed_value_is_reported(self):
        class Event:
            def query(self):return True
            def elapsed_time(self,other):return .25
        row=dict(_start=Event(),_end=Event(),stream=7,native_stream_matches=True,end_stream_matches=True,end_device_matches=True)
        self.assertEqual(finish_interval(copy.copy(row))['milliseconds'],.25)
        row['end_device_matches']=False
        self.assertIsNone(finish_interval(row)['milliseconds'])

    def test_different_streams_or_impossible_sum_cannot_qualify(self):
        frame,peer=timing_frame();q={0:dict(usable=True),1:dict(usable=True)}
        self.assertEqual(timing_summary(frame,peer,q)[0]['status'],'QUALIFIED_WITH_LIMITS')
        frame['layers'][0]['expert_timing']['stream']=8
        self.assertEqual(timing_summary(frame,peer,q)[0]['status'],'UNSUPPORTED')
        frame,peer=timing_frame();frame['forward_timing']['milliseconds']=.1
        self.assertIsNone(timing_summary(frame,peer,q)[0]['sums_ms']['expert'])

    def test_unavailable_timing_keeps_structural_routing(self):
        frame,peer=timing_frame()
        self.assertEqual(routing_summary(frame,peer)['total_assignments'],4096)
        result=timing_summary(frame,peer,{0:dict(usable=False),1:dict(usable=False)})
        self.assertEqual(result[0]['status'],'UNSUPPORTED')

    def test_complete_small_matrix_retains_signed_overhead_and_null_contrast(self):
        episodes=[]
        for spec in frozen_sequence(['a','b']):
            delta={'control':100,'timed':90,'diagnostic':130}[spec['mode']]
            episodes.append(dict(**spec,warmup=False,client_elapsed_ns=delta,
                serving=dict(grants=[32],computed_before=0,KV_released_blocks=2,KV_initial_free=100),
                routing=profile(0) if spec['mode']!='control' else None,
                terminal=dict(token_ids=[1],finish_reason='length'),
                timing={r:dict(status='QUALIFIED_WITH_LIMITS',sums_ms=dict(expert=.1,dispatch=.2,combine=.3)) for r in (0,1)} if spec['mode']=='timed' else None))
        report=compare_report(episodes,['a','b'])
        self.assertEqual(report['incremental_overhead_proxy_ns']['a'][0]['timed_minus_control'],-10)
        self.assertEqual(report['timing']['a'][0]['expert']['n'],3)
        self.assertTrue(report['null_results'])
        self.assertTrue(report['logical_communication_bytes_constant'])
        self.assertTrue(report['timing_complete'])
        next(row for row in episodes if row['mode']=='timed')['timing'][1]['status']='UNSUPPORTED'
        self.assertFalse(compare_report(episodes,['a','b'])['timing_complete'])
        for row in episodes:
            if row['timing']:
                for r in (0,1):row['timing'][r]['status']='UNSUPPORTED'
        self.assertEqual(compare_report(episodes,['a','b'])['timing_coverage'],'UNSUPPORTED')


class FreezeTests(unittest.TestCase):
    def test_missing_or_wrong_freeze_stops_before_runtime_restore(self):
        with patch.object(sys,'argv',['routing_run.py','compare']),patch.object(routing_run.model_qualify,'qualify') as prepare,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):routing_run.main()
            prepare.assert_not_called()

    def test_source_and_driver_drift_stop_before_native_import(self):
        for changed in ['source','driver']:
            with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=False),contextlib.redirect_stderr(io.StringIO()):
                out=Path(d);(out/'context.json').write_text(json.dumps(dict(driver_versions=['B'])))
                plan=dict(code_sha256={} if changed=='source' else code_hashes(),driver_versions=['A'])
                p=out/'freeze.json';p.write_text(json.dumps(plan));sha=hashlib.sha256(p.read_bytes()).hexdigest()
                result=asyncio.run(run(out,'SYNTHETIC_NO_MODEL',[], 'compare',str(p),sha))
                self.assertEqual(result['status'],'BLOCKED')
                self.assertIn('source changed' if changed=='source' else 'Driver drift',result['error'])


if __name__=='__main__':unittest.main()
