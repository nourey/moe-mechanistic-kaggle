"""Offline negative controls; synthetic fixtures never establish GPU behavior."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import load_sensitivity as s
import load_sensitivity_run as runner


def fixture():
    _, trace = s.frozen_trace()
    plan = s.rebase(trace, 10_000_000_000)
    client, joins = [], {}
    inputs = json.loads((s.ROOT / 'load_protocol.json').read_text())['inputs']
    for i, r in enumerate(plan['arrivals']):
        client.append(dict(r, event='submitted', seq=i*2, monotonic_ns=r['due_ns']+1_000_000))
        client.append(dict(event='terminal', stage='trajectory', request_id=r['request_id'], seq=i*2+1,
                           monotonic_ns=r['due_ns']+2_000_000_000, finish_reason='length', token_ids=[1]*8))
        joins[r['request_id']] = dict(internal_request_id='SYNTHETIC-' + r['request_id'], prompt_token_ids=inputs[r['family']])
    return plan, client, joins, inputs


def assess(data):
    return s.client_assessment('trajectory', *data, 8)


def cells():
    config, _ = s.frozen_trace()
    summary = assess(fixture())
    return [dict(name=c['name'], condition=c, status='PASS_FIXED_REPLAY', client=copy.deepcopy(summary),
                 engine_args=s.configure(dict(max_num_batched_tokens=512, seed=1525), c['max_num_seqs'], c['observer'], 'AsyncScheduler', True),
                 effective_scheduler=dict(max_num_seqs=c['max_num_seqs'], active_requests=c['max_num_seqs'], max_num_batched_tokens=512, async_scheduling=True, max_concurrent_batches=2),
                 gpu_ids=['SYNTHETIC-0', 'SYNTHETIC-1'], kv_capacity_blocks=[2237,2237],
                 native_sources={'SYNTHETIC_native_source':'SYNTHETIC_hash'},
                 native_owner_audit='PASS' if c['observer']=='observed' else 'UNKNOWN_REFERENCE') for c in config['order']]


class FrozenTraceTests(unittest.TestCase):
    def test_exact_original_trace_and_85_owners(self):
        config, trace = s.frozen_trace()
        self.assertEqual(config['baseline_commit'], '433c7316829118cac726bf7998e64a89ba08d2b7')
        self.assertEqual([sum(r['phase']==w['name'] for r in trace['arrivals']) for w in trace['windows']], [10,28,37,10])
        self.assertEqual([r['owner'] for r in trace['arrivals']], [i%2 for i in range(85)])

    def test_integer_translation_preserves_every_interarrival(self):
        _, trace = s.frozen_trace(); before=copy.deepcopy(trace)
        a=s.rebase(trace,100); b=s.rebase(trace,9_000_000_000_000_000)
        self.assertEqual(trace,before)
        self.assertEqual(s.schedule_identity(a),s.schedule_identity(b))
        self.assertEqual([r['due_ns']-trace['arrivals'][i]['due_ns'] for i,r in enumerate(a['arrivals'])], [100-trace['windows'][0]['start_ns']]*85)

    def test_float_or_mutated_trace_is_rejected(self):
        with self.assertRaises(ValueError):s.rebase(s.frozen_trace()[1],1.1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('load_sensitivity.json','load_protocol.json','load_replay_trace.json'):
                (root/name).write_bytes((s.ROOT/name).read_bytes())
            (root/'load_replay_trace.json').write_bytes((root/'load_replay_trace.json').read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'trace hash'):s.frozen_trace(root)

    def test_only_limit_and_observation_adapters_change(self):
        original=json.loads((s.ROOT/'model-qualification.json').read_text())['engine_args']; saved=copy.deepcopy(original)
        for mode in ('observed','reference'):
            a=s.configure(original,4,mode,'AsyncScheduler',True)
            b=s.configure(original,8,mode,'AsyncScheduler',True)
            self.assertEqual({k for k in a if a[k]!=b[k]}, {'max_num_seqs'})
            self.assertEqual(a['max_num_batched_tokens'],512)
            if mode=='reference':
                self.assertNotIn('scheduler_cls',a)
                self.assertEqual(a['worker_extension_cls'],'model_observer.ReadOnlyRankState')
        self.assertEqual(original,saved)
        for limit in (0,16,True):
            with self.assertRaises(ValueError):s.configure(original,limit,'observed','AsyncScheduler',True)


class ClientTests(unittest.TestCase):
    def test_frontend_reference_keeps_terminals_but_not_native_claims(self):
        v=assess(fixture())
        self.assertEqual(v['normal_completions'],85)
        self.assertEqual(v['offer_lateness_ms']['maximum'],1)
        self.assertFalse(v['phase_drift'])
        self.assertIn('UNKNOWN',v['native_owner_verification'])

    def test_terminal_duplicate_missing_or_censored_cannot_pass(self):
        for mutation in ('missing','duplicate','censored'):
            data=fixture()
            if mutation=='missing':data[1].pop()
            if mutation=='duplicate':data[1][-1]=copy.deepcopy(data[1][1])
            if mutation=='censored':data[1].append(dict(stage='trajectory',event='censored_before_cleanup'))
            with self.assertRaisesRegex(ValueError,'workload'):assess(data)

    def test_owner_due_or_prompt_changes_cannot_pass(self):
        for key in ('owner','due_ns','family'):
            data=fixture();data[1][0][key]='wrong'
            with self.assertRaises(ValueError):assess(data)
        data=fixture();next(iter(data[2].values()))['prompt_token_ids']=[99]
        with self.assertRaisesRegex(ValueError,'prompt'):assess(data)

    def test_early_offer_terminal_and_orphan_join_rejected(self):
        data=fixture();data[1][0]['monotonic_ns']-=2_000_000
        with self.assertRaisesRegex(ValueError,'event order'):assess(data)
        data=fixture();data[1][1]['monotonic_ns']=0
        with self.assertRaisesRegex(ValueError,'event order'):assess(data)
        data=fixture();joins=list(data[2].values());joins[1]['internal_request_id']=joins[0]['internal_request_id']
        with self.assertRaisesRegex(ValueError,'identity'):assess(data)

    def test_phase_drift_is_exposed_not_silently_reassigned(self):
        data=fixture();rid=data[1][0]['request_id'];hi=data[0]['windows'][0]['end_ns']
        data[1][0]['monotonic_ns']=hi+1;data[1][1]['monotonic_ns']=hi+100
        self.assertIn(rid,assess(data)['phase_drift'])


class ComparisonTests(unittest.TestCase):
    def test_both_limit_contrasts_and_matched_observer_controls(self):
        result=s.comparison(cells(),s.frozen_trace()[0])
        self.assertEqual(result['status'],'PASS_MATCHED_COLLECTION')
        self.assertEqual([c['axis'] for c in result['contrasts']], ['active_limit']*2+['observer']*2)
        self.assertTrue(all(c['median_residence_ratio_right_over_left']==1 for c in result['contrasts']))

    def test_missing_cell_different_gpu_or_native_source_fails(self):
        for kind in ('missing','gpu','source','owner','token_budget','effective'):
            values=cells()
            if kind=='missing':values.pop()
            elif kind=='gpu':values[1]['gpu_ids'][1]='OTHER-GPU'
            elif kind=='source':values[1]['native_sources']={}
            elif kind=='owner':values[0]['native_owner_audit']='UNKNOWN'
            elif kind=='token_budget':values[1]['engine_args']['max_num_batched_tokens']=1024
            elif kind=='effective':values[1]['effective_scheduler']['max_num_seqs']=4
            with self.assertRaises(ValueError):s.comparison(values,s.frozen_trace()[0])

    def test_phase_drift_or_changed_schedule_rejected(self):
        for key,value in [('phase_drift',['a']),('schedule_digest','OTHER')]:
            values=cells();values[1]['client'][key]=value
            with self.assertRaises(ValueError):s.comparison(values,s.frozen_trace()[0])

    def test_output_variation_and_kv_profile_difference_are_explicit(self):
        values=cells();values[2]['client']['request_metrics'][0]['output_tokens']=7
        values[2]['kv_capacity_blocks']=[2236,2236]
        result=s.comparison(values,s.frozen_trace()[0])
        self.assertEqual(len(result['contrasts'][0]['output_length_changes']),1)
        self.assertEqual(result['kv_capacity_by_cell']['seq8_observed'],[2236,2236])


class LauncherTests(unittest.TestCase):
    def test_reference_cleanup_excludes_frontend_with_in_core_workers(self):
        self.assertEqual(s.process_members({'20':dict(present=True,parent_pid=10),
                                           '21':dict(present=True,parent_pid=10)},10),[20,21])
        self.assertEqual(s.process_members({'30':dict(present=True,parent_pid=20),
                                           '31':dict(present=True,parent_pid=21)},10),[20,21,30,31])
        with self.assertRaises(ValueError):s.process_members({'20':dict(present=False)},10)

    def invoke(self,fail=False):
        values=cells();config=s.frozen_trace()[0];calls=[]
        def execute(label,argv,timeout):
            calls.append((label,argv))
            if label.startswith('gpu-before-'):return dict(exit_code=0,stdout='SYNTHETIC GPU fixture',stderr='')
            trial=Path(argv[argv.index('--output')+1]);cell=next(c for c in values if c['name']==label)
            native=dict(status='BLOCKED' if fail else 'PASS_FIXED_REPLAY',shutdown_returned=True,
                owned_processes_remaining=[],cleanup_unknown_processes=[],client_summary=cell['client'],
                native_owner_audit='PASS_ACCOUNTING' if cell['condition']['observer']=='observed' else 'UNKNOWN_REFERENCE')
            (trial/'load-result.json').write_text(json.dumps(native))
            (trial/'configuration.json').write_text(json.dumps(dict(source=runner.hashes(),engine_args=cell['engine_args'],effective_scheduler=cell['effective_scheduler'])))
            (trial/'native-sources.json').write_text(json.dumps(cell['native_sources']))
            for r in (0,1):(trial/f'loaded-dp{r}.json').write_text(json.dumps(dict(gpu_uuid=cell['gpu_ids'][r],kv_capacity_blocks=2237)))
            return dict(exit_code=0,stdout='SYNTHETIC native process fixture',stderr='',timed_out=False)
        with tempfile.TemporaryDirectory() as d:
            result=runner.run_trials(Path(d),dict(python='SYNTHETIC-PYTHON',model_path='SYNTHETIC-MODEL',gpu_ids=['SYNTHETIC-0','SYNTHETIC-1']),execute)
        return result,calls

    def test_four_actual_entry_commands_and_sealed_comparison(self):
        result,calls=self.invoke()
        self.assertEqual(result['status'],'PASS_MATCHED_COLLECTION')
        native=[(label,argv) for label,argv in calls if not label.startswith('gpu-before-')]
        self.assertEqual([label for label,argv in native],[c['name'] for c in s.frozen_trace()[0]['order']])
        for label,argv in native:self.assertEqual(argv[argv.index('--sensitivity-condition')+1],label)

    def test_first_failure_stops_later_cells_and_keeps_status(self):
        result,calls=self.invoke(fail=True)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertIn('STOP_seq4_observed',result['error'])
        self.assertEqual(len(calls),2)


if __name__=='__main__':unittest.main()
