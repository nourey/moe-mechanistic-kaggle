"""Instance-local observation taps; native methods/arguments/results are preserved."""
import json
import os
from pathlib import Path
import time

from model_observer import ReadOnlyRankState
from observation_contract import submission_work
from observation_timing import start_interval, end_interval, finish_interval


def tensor_identity(t):
    return dict(pointer=hex(t.data_ptr()), shape=list(t.shape), dtype=str(t.dtype), device=str(t.device))


def expert_target(layer):
    # In this pinned build, RoutedExperts.forward_modular returns AFTER combine.
    # Tap the actual Triton expert call AFTER prepare/dispatch, BEFORE finalize.
    impl = layer.quant_method.moe_kernel.impl
    if type(impl).__name__ != 'FusedMoEKernelModularImpl': raise ValueError('Unsupported kernel implementation')
    if type(impl.prepare_finalize).__name__ != 'MoEPrepareAndFinalizeNaiveDPEPModular':
        raise ValueError('Unsupported prepare/finalize path')
    experts = impl.fused_experts
    if type(experts).__name__ != 'TritonExperts': raise ValueError('Unsupported expert implementation')
    if type(experts.finalize_weight_and_reduce_impl()).__name__ != 'TopKWeightAndReduceNoOP':
        raise ValueError('Expert output not yet weighted/reduced; no guessed combine input')
    return experts


class ObservationWorker(ReadOnlyRankState):
    def observation_install(self, folder, qualify_timing=False):
        import torch
        from vllm.distributed import get_ep_group
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        if hasattr(self, '_mechanistic_obs'): raise ValueError('Observation already installed')
        if not self.use_v2_model_runner: raise ValueError('This bounded observer requires the qualified native ModelRunnerV2')
        if self.cache_config.enable_prefix_caching: raise ValueError('APC must remain disabled')
        state = dict(folder=folder, episode=None, forwards=[], active=None, seq=0, installed_layers=0, mode='diagnostic')
        self._mechanistic_obs = state
        runner = self.model_runner
        execute = runner.execute_model

        def observed_execute(scheduler_output, *args, **kwargs):
            if state['episode'] is None: return execute(scheduler_output, *args, **kwargs)
            if len(state['forwards']) >= 32: raise RuntimeError('Bounded observation exceeded 32 forwards per episode')
            state['seq'] += 1
            frame = dict(forward=state['seq'], **submission_work(scheduler_output.num_scheduled_tokens, kwargs.get('dummy_run', False)), layers=[],
                submit_monotonic_ns=time.monotonic_ns(), phase='Worker model-runner submission; not GPU completion')
            state['forwards'].append(frame); state['active'] = frame
            try:
                result = execute(scheduler_output, *args, **kwargs)
                frame['return_monotonic_ns'] = time.monotonic_ns()
                frame['return_note'] = 'Native execute_model returned; async outputs may still be pending'
                return result
            finally: state['active'] = None

        runner.execute_model = observed_execute

        def pre_forward(module, args, kwargs):
            frame = state['active']
            if frame is None or state['mode']=='control': return
            positions = kwargs.get('positions')
            inputs = kwargs.get('input_ids')
            if positions is not None: frame['_positions'] = positions.detach().clone()
            if inputs is not None: frame['_input_ids'] = inputs.detach().clone()
            if state['mode']=='timed' and state.get('timing_usable',False): frame['forward_timing'] = start_interval(torch)

        def post_forward(module,args,kwargs,result):
            frame=state['active']
            if frame is not None and 'forward_timing' in frame: end_interval(torch,frame['forward_timing'])

        self.get_model().register_forward_pre_hook(pre_forward, with_kwargs=True)
        self.get_model().register_forward_hook(post_forward, with_kwargs=True)
        for name, layer in self.get_model().named_modules():
            if not isinstance(layer, RoutedExperts): continue
            if layer.quant_method.is_monolithic: raise ValueError('Only qualified modular Triton path supported')
            mapping = layer.expert_map.detach().cpu().tolist()
            experts = expert_target(layer)
            original = experts.apply

            def observe_expert(*args, _name=name, _mapping=mapping, _original=original, **kwargs):
                frame = state['active']
                if frame is None or state['mode']=='control': return _original(*args, **kwargs)
                ids = kwargs['topk_ids']
                ctx = get_forward_context()
                metadata = ctx.dp_metadata
                record = dict(layer=_name, global_to_local_map=_mapping,
                    dp_sizes=None if metadata is None else list(metadata.get_chunk_sizes_across_dp_rank()),
                    native_dp_token_counts=None if metadata is None else metadata.num_tokens_across_dp_cpu.tolist(),
                    expert_entry_monotonic_ns=time.monotonic_ns(),
                    dispatch=frame.pop('_dispatch', None), combine_returned=False,
                    combine_input_matches_expert_output=False)
                frame['layers'].append(record)
                record['_topk'] = ids.detach().clone()
                if state['mode']=='timed' and state.get('timing_usable',False): record['expert_timing'] = start_interval(torch)
                record['native_expert_call_enter_ns']=time.monotonic_ns()
                result = _original(*args, **kwargs)
                record['native_expert_call_return_ns']=time.monotonic_ns()
                if state['mode']=='timed' and state.get('timing_usable',False): end_interval(torch,record['expert_timing'])
                # TritonExperts.apply fills the supplied output buffer in place.
                output = kwargs['output']
                if result is not None or not isinstance(output, torch.Tensor):
                    raise ValueError('Unfinalized/unsupported Triton expert result')
                record['expert_output_tensor'] = tensor_identity(output)
                if state['mode']=='diagnostic': record['_output'] = output.detach().clone()
                record['_event'] = torch.cuda.Event(enable_timing=False)
                record['_event'].record()
                frame['_last_layer'] = record
                return result

            experts.apply = observe_expert
            state['installed_layers'] += 1
        if state['installed_layers'] != 16: raise ValueError('Expected exactly 16 modular expert taps')
        manager = get_ep_group().device_communicator.all2all_manager
        if type(manager).__name__ != 'AgRsAll2AllManager': raise ValueError('Expected native AGRS manager')
        dispatch = manager.dispatch
        combine = manager.combine

        def observed_dispatch(hidden_states, topk_weights, topk_ids, *args, **kwargs):
            frame = state['active']
            if frame is None or state['mode']=='control': return dispatch(hidden_states, topk_weights, topk_ids, *args, **kwargs)
            row = dict(input_shape=list(hidden_states.shape), local_topk_shape=list(topk_ids.shape),
                       logical_input_bytes=sum(t.numel()*t.element_size() for t in (hidden_states,topk_weights,topk_ids)),
                       byte_scope='Logical input tensors, not physical wire traffic',
                       enter_monotonic_ns=time.monotonic_ns(), backend='native AgRsAll2AllManager.dispatch')
            if state['mode']=='timed' and state.get('timing_usable',False): row['gpu_timing'] = start_interval(torch)
            result = dispatch(hidden_states, topk_weights, topk_ids, *args, **kwargs)
            if state['mode']=='timed' and state.get('timing_usable',False): end_interval(torch,row['gpu_timing'])
            row.update(return_monotonic_ns=time.monotonic_ns(), returned=True,
                       gathered_shape=list(result[0].shape))
            frame['_dispatch'] = row
            return result

        def observed_combine(hidden_states, *args, **kwargs):
            frame = state['active']
            if frame is None or state['mode']=='control': return combine(hidden_states, *args, **kwargs)
            row = frame.get('_last_layer')
            if row is None: raise ValueError('Combine without observed expert output')
            row['combine_input_tensor'] = tensor_identity(hidden_states)
            row['combine_logical_input_bytes']=hidden_states.numel()*hidden_states.element_size()
            row['combine_input_matches_expert_output'] = row['combine_input_tensor'] == row['expert_output_tensor']
            row['combine_enter_monotonic_ns'] = time.monotonic_ns()
            if state['mode']=='timed' and state.get('timing_usable',False): row['combine_timing'] = start_interval(torch)
            result = combine(hidden_states, *args, **kwargs)
            if state['mode']=='timed' and state.get('timing_usable',False): end_interval(torch,row['combine_timing'])
            if state['mode']=='diagnostic' and '.layers.0.' in row['layer'] and row['_topk'].shape[0] >= 32:
                row['_combined'] = result.detach().clone()
            row['combine_returned'] = True
            row['combine_output_shape']=list(result.shape)
            row['combine_return_monotonic_ns'] = time.monotonic_ns()
            row['_combine_event'] = torch.cuda.Event(enable_timing=False)
            row['_combine_event'].record()
            return result

        manager.dispatch = observed_dispatch
        manager.combine = observed_combine
        if qualify_timing:
            try:
                from vllm.distributed import get_dp_group
                communicator=get_dp_group().device_communicator.pynccl_comm
                interval=start_interval(torch)
                tensor=torch.ones(4096,device='cuda',dtype=torch.float16)
                tensor=tensor+1
                end_interval(torch,interval);interval['_end'].synchronize()
                receipt=finish_interval(interval)
                receipt.update(elementwise_result_verified=bool((tensor==2).all()),
                    pynccl_available=communicator is not None and not communicator.disabled,
                    scope='One-device CUDA-event sanity only; resolution/absolute accuracy/cross-device alignment not qualified')
                receipt['usable']=receipt['status']=='OBSERVED_STREAM_INTERVAL' and receipt['milliseconds']>0 and receipt['elementwise_result_verified'] and receipt['pynccl_available']
            except Exception as error:
                receipt=dict(status='UNSUPPORTED',usable=False,error=repr(error))
            state['timing_usable']=receipt['usable']
            identity=json.loads(Path(folder,f'loaded-dp{self.parallel_config.data_parallel_rank}.json').read_text())
            receipt.update(dp_rank=self.parallel_config.data_parallel_rank,worker_pid=os.getpid(),gpu_uuid=identity['gpu_uuid'])
            Path(folder,f'timing-qualification-dp{self.parallel_config.data_parallel_rank}.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return dict(installed_layers=state['installed_layers'], dp_rank=self.parallel_config.data_parallel_rank)

    def observation_begin(self, episode, mode='diagnostic'):
        state = self._mechanistic_obs
        if state['episode'] is not None: raise ValueError('Previous episode not drained')
        if mode not in ('diagnostic','timed','control'): raise ValueError('Unknown bounded observation mode')
        state.update(episode=episode, forwards=[], seq=0, mode=mode)
        return dict(dp_rank=self.parallel_config.data_parallel_rank, episode=episode)

    def observation_end(self, episode):
        import torch
        state = self._mechanistic_obs
        if state['episode'] != episode or state['active'] is not None: raise ValueError('Episode mismatch/in-flight callback')
        state['episode'] = None
        frames = state['forwards']
        # Only collect after terminal + native group drain. No per-layer CPU copy,
        # no profiler or global cuda.synchronize. Timed mode uses CUDA events;
        # their completion is collected here, outside the request window.
        events = [r.get('_combine_event', r['_event']) for f in frames for r in f['layers']]
        forward_ends=[f['forward_timing']['_end'] for f in frames if 'forward_timing' in f]
        if forward_ends: forward_ends[-1].synchronize()
        elif events: events[-1].synchronize()
        for frame in frames:
            if 'forward_timing' in frame: frame['forward_timing']=finish_interval(frame['forward_timing'])
            for key in ['positions', 'input_ids']:
                tensor = frame.pop('_' + key, None)
                if tensor is not None: frame[key] = tensor.cpu().tolist()
            frame.pop('_dispatch', None); frame.pop('_last_layer', None)
            for row in frame['layers']:
                event = row.pop('_event')
                combine_event = row.pop('_combine_event', None)
                row['cuda_completion_verified'] = event.query() and combine_event is not None and combine_event.query()
                row['topk_ids'] = row.pop('_topk').cpu().tolist()
                for key in ['expert_timing','combine_timing']:
                    if key in row: row[key]=finish_interval(row[key])
                if row.get('dispatch') and 'gpu_timing' in row['dispatch']:
                    row['dispatch']['gpu_timing']=finish_interval(row['dispatch']['gpu_timing'])
                output = row.pop('_output',None)
                if output is not None:
                    output=output.cpu()
                    row['output_abs_sums'] = output.float().abs().sum(dim=-1).tolist()
                    row['output_finite'] = bool(torch.isfinite(output).all())
                    if not row['output_finite']: raise ValueError('Non-finite observed expert contribution')
                else:
                    row['output_abs_sums']=None;row['output_finite']=None
                combined = row.pop('_combined', None)
                if combined is not None:
                    import numpy as np
                    stem = episode + '-dp' + str(self.parallel_config.data_parallel_rank) + '-f' + str(frame['forward']) + '-layer0'
                    row['partial_output_file'] = stem + '-partial.npy'
                    row['combine_output_file'] = stem + '-combined.npy'
                    np.save(Path(state['folder']) / row['partial_output_file'], output.numpy(), allow_pickle=False)
                    np.save(Path(state['folder']) / row['combine_output_file'], combined.cpu().numpy(), allow_pickle=False)
        record = dict(episode=episode, dp_rank=self.parallel_config.data_parallel_rank, worker_pid=os.getpid(),
            observed_parent_pid=os.getppid(),
            mode=state['mode'],
            forwards=frames, clock='Per-process monotonic_ns; timestamps do not establish cross-process order',
            scope='Diagnostic dataflow, or qualified per-device timed intervals, or control CPU submission only. Mode-specific missing fields remain unobserved.')
        path = Path(state['folder']) / (episode + '-worker-dp' + str(record['dp_rank']) + '.json')
        with path.open('x') as f: json.dump(record, f, indent=2)
        state['forwards'] = []
        return dict(dp_rank=record['dp_rank'], written=str(path))
