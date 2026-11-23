"""Observation-only subclasses selected to preserve native sync/async policy."""
import os
import time
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from observation_contract import Writer, kv_rows, request_row


class OwnershipOnlyMixin:
    """Minimal reference: native scheduling with only authoritative acceptance I/O."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner_obs = Writer(os.environ['MOE_OBSERVATION_ROOT'], 'core', self.parallel_config.data_parallel_rank)
        self.owner_obs.emit('core_initialized', scheduler=type(self).__name__,
            scope='Acceptance-only reference; KV/grants/batch/work unobserved')

    def add_request(self, request):
        result = super().add_request(request)
        returned = time.monotonic_ns()
        self.owner_obs.emit('request_accepted', request_id=request.request_id,
            prompt_token_ids=list(request.prompt_token_ids), native_accept_return_ns=returned,
            scope='Native add_request returned in this rank EngineCore; not a scheduling grant')
        return result


class OwnershipOnlyScheduler(OwnershipOnlyMixin, Scheduler):
    pass


class OwnershipOnlyAsyncScheduler(OwnershipOnlyMixin, AsyncScheduler):
    pass


class ObservationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs = Writer(os.environ['MOE_OBSERVATION_ROOT'], 'core', self.parallel_config.data_parallel_rank)
        self.obs_decision = 0
        self.obs_outputs = {}
        self.load_mode = os.environ.get('MOE_LOAD_MODE') == '1'
        self.obs.emit('core_initialized', scheduler=type(self).__name__, async_scheduling=self.scheduler_config.async_scheduling,
            total_blocks=self.kv_cache_manager.block_pool.num_gpu_blocks,
            free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
            prefix_caching=self.cache_config.enable_prefix_caching)
        manager = self.kv_cache_manager
        allocate = manager.allocate_slots
        free = manager.free

        def allocate_observed(request, *a, **kw):
            before = kv_rows(manager, request.request_id)
            begin=time.monotonic_ns() if self.load_mode else None
            result = allocate(request, *a, **kw)
            end=time.monotonic_ns() if self.load_mode else None
            self.obs.emit('allocation', request_id=request.request_id, decision=self.obs_decision,
                num_new_tokens=kw.get('num_new_tokens', a[0] if a else None),
                accepted=result is not None, before_blocks=before, after_blocks=kv_rows(manager, request.request_id),
                free_blocks=manager.block_pool.get_num_free_blocks(),
                **(dict(native_call_interval_ns=[begin,end]) if self.load_mode else {}))
            return result

        def free_observed(request):
            before_objects = [list(group) for group in manager.get_blocks(request.request_id).blocks]
            begin=time.monotonic_ns() if self.load_mode else None
            result = free(request)
            end=time.monotonic_ns() if self.load_mode else None
            self.obs.emit('kv_free', request_id=request.request_id,
                terminal_request_state=request_row(request),
                released_blocks=[[dict(block_id=b.block_id, ref_cnt=b.ref_cnt) for b in g] for g in before_objects],
                remaining_blocks=kv_rows(manager, request.request_id),
                free_blocks=manager.block_pool.get_num_free_blocks(),
                **(dict(native_call_interval_ns=[begin,end]) if self.load_mode else {}))
            return result

        manager.allocate_slots = allocate_observed
        manager.free = free_observed

    def add_request(self, request):
        result = super().add_request(request)
        accepted_return=time.monotonic_ns() if self.load_mode else None
        self.obs.emit('request_accepted', **request_row(request), prompt_token_ids=list(request.prompt_token_ids),
            **(dict(load_state=self.load_state(),native_accept_return_ns=accepted_return) if self.load_mode else {}))
        return result

    def load_state(self):
        return dict(waiting=[r.request_id for r in self.waiting],
            kv_holding_waiting=[r.request_id for r in self.kv_holding_waiting],
            deferred=sorted(r.request_id for r in self.deferred_waiting),
            running=[r.request_id for r in self.running],
            requests={rid:{**request_row(r), 'preemptions':r.num_preemptions} for rid,r in self.requests.items()},
            free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
            limits=dict(active=self.max_num_active_reqs,slots=self.max_num_running_reqs,
                        scheduled_tokens=self.max_num_scheduled_tokens,input_tokens=self.scheduler_config.max_num_batched_tokens),
            streaming_slots=self.num_waiting_for_streaming_input,
            pause_state=str(self._pause_state),
            pause_state_name=getattr(self._pause_state,'name',None),
            reason_scope='Boundary facts; native branch/exclusive bottleneck not instrumented')

    def schedule(self, *args, **kwargs):
        self.obs_decision += 1
        before = {rid: request_row(r) for rid, r in self.requests.items()}
        load_before = self.load_state() if self.load_mode else None
        decision_enter=time.monotonic_ns() if self.load_mode else None
        result = super().schedule(*args, **kwargs)
        decision_return=time.monotonic_ns() if self.load_mode else None
        self.obs_outputs[id(result)] = self.obs_decision
        if before or result.num_scheduled_tokens:
            self.obs.emit('schedule', decision=self.obs_decision, grants=dict(result.num_scheduled_tokens),
                before=before, after={rid: request_row(r) for rid, r in self.requests.items()},
                waiting=[r.request_id for r in self.waiting], running=[r.request_id for r in self.running],
                **(dict(load_before=load_before,load_after=self.load_state(),
                    native_decision_interval_ns=[decision_enter,decision_return],
                    preempted_req_ids=sorted(result.preempted_req_ids),
                    budget_used=sum(result.num_scheduled_tokens.values())) if self.load_mode else {}))
        return result

    def update_from_output(self, scheduler_output, model_runner_output):
        decision = self.obs_outputs.pop(id(scheduler_output))
        before = {rid: request_row(r) for rid, r in self.requests.items()}
        result = super().update_from_output(scheduler_output, model_runner_output)
        if before or scheduler_output.num_scheduled_tokens:
            self.obs.emit('output_applied', decision=decision, grants=dict(scheduler_output.num_scheduled_tokens),
                before=before, after={rid: request_row(r) for rid, r in self.requests.items()},
                free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                deferred_frees=len(self.deferred_frees),
                **(dict(load_state=self.load_state()) if self.load_mode else {}))
        return result


class ObservedScheduler(ObservationMixin, Scheduler):
    pass


class ObservedAsyncScheduler(ObservationMixin, AsyncScheduler):
    pass
