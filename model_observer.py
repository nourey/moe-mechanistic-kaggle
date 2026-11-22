"""Named, read-only vLLM worker extension. No constructor or inference overrides."""
import json
import os
from pathlib import Path
import time


class ReadOnlyRankState:
    def research_rank_state(self, folder, phase):
        if phase not in ('loaded', 'after'):
            raise ValueError('Unsupported observation phase')
        import torch
        from vllm.distributed import get_dp_group, get_ep_group, get_tp_group, get_pp_group
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        pc = self.parallel_config
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        hf = self.model_config.hf_config
        record = dict(phase=phase, pid=os.getpid(), observed_monotonic_ns=time.monotonic_ns(),
                      clock='Worker-local monotonic_ns; no cross-rank decision-phase claim',
                      dp_rank=pc.data_parallel_rank, device_index=device,
                      gpu_uuid=str(getattr(props, 'uuid', 'UNAVAILABLE')),
                      model_type=hf.model_type, model_dtype=str(self.model_config.dtype),
                      model_num_layers=hf.num_hidden_layers, global_num_experts=hf.num_experts,
                      dp_world_size=get_dp_group().world_size, ep_world_size=get_ep_group().world_size,
                      ep_rank=get_ep_group().rank_in_group, tp_world_size=get_tp_group().world_size,
                      pp_world_size=get_pp_group().world_size, all2all_backend=pc.all2all_backend,
                      engine_core_nonce='NOT_OBSERVED; worker PID is not an incarnation nonce',
                      scheduler_request_ledger='NOT_OBSERVED', KV_request_ownership='NOT_OBSERVED',
                      expert_layers=[], kv_tensors=[])
        model = self.get_model()
        for name, layer in model.named_modules():
            if isinstance(layer, RoutedExperts):
                mapping = layer.expert_map
                record['expert_layers'].append(dict(name=name, w13_shape=list(layer.w13_weight.shape),
                    w2_shape=list(layer.w2_weight.shape), dtype=str(layer.w13_weight.dtype),
                    device=str(layer.w13_weight.device), is_cuda=layer.w13_weight.is_cuda,
                    local_num_experts=layer.local_num_experts,
                    global_to_local_map=None if mapping is None else mapping.detach().cpu().tolist()))
        runner = self.model_runner
        cfg = getattr(runner, 'kv_cache_config', None)
        record['kv_capacity_blocks'] = None if cfg is None else cfg.num_blocks
        record['kv_note'] = 'Allocated cache capacity only; no live page ownership/reservation/population inference'
        seen = set()
        for tensor in getattr(runner, 'kv_caches', []):
            if isinstance(tensor, torch.Tensor):
                storage = tensor.untyped_storage()
                key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
                if key in seen: continue
                seen.add(key)
                record['kv_tensors'].append(dict(device=str(tensor.device), is_cuda=tensor.is_cuda,
                    dtype=str(tensor.dtype), shape=list(tensor.shape), storage_bytes=storage.nbytes(),
                    storage_ptr=hex(storage.data_ptr()), pointer_namespace='This PID + GPU UUID only'))
        comm = get_ep_group().device_communicator
        manager = getattr(comm, 'all2all_manager', None)
        record['ep_communication_manager'] = None if manager is None else type(manager).__name__
        path = Path(folder) / (phase + '-dp' + str(record['dp_rank']) + '.json')
        with path.open('x') as f: json.dump(record, f, indent=2)
        # DPLB returns only the first worker result; the per-rank files preserve both.
        return dict(dp_rank=record['dp_rank'], written=str(path))
