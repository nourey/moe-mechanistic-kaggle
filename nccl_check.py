#!/usr/bin/env python3
"""Stage C: fixed-size vLLM PyNccl CUDA collectives; no model or serving."""
import argparse
import ctypes
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import uuid
import zipfile

import prepare_runtime as runtime
from moe_kernel import cuda_events

ROOT = Path(__file__).resolve().parent


def gather_input(rank, count):
    return [i % 16 + 32 * rank for i in range(count)]


def scatter_input(rank, count):
    return [i % 8 + 16 * (owner + 1) + 4 * rank for owner in range(2) for i in range(count)]


def expected_outputs(rank, count):
    return gather_input(0, count) + gather_input(1, count), [32 * (rank + 1) + 4 + 2 * (i % 8) for i in range(count)]


def exact_outputs(rank, count, gathered, scattered):
    expected_gather, expected_scatter = expected_outputs(rank, count)
    return dict(all_gather_exact=list(gathered) == expected_gather,
                reduce_scatter_exact=list(scattered) == expected_scatter)


def aggregate(rows, expected_ids):
    if len(rows) != 2 or sorted(r.get('rank', -1) for r in rows) != [0, 1]: return 'FAIL'
    if any(r.get('status') not in ('PASS', 'PARTIAL') for r in rows): return 'FAIL'
    for row in rows:
        if row.get('gpu_uuid') != expected_ids[row['rank']]: return 'FAIL'
        if not row.get('all_gather_exact') or not row.get('reduce_scatter_exact'): return 'FAIL'
        if not row.get('native_destroy_returned') or not row.get('gloo_destroy_returned'): return 'FAIL'
    return 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'PARTIAL'


def rank_worker(rank, folder, config, expected_ids):
    out = Path(folder) / ('rank-' + str(rank)); out.mkdir()
    # Capture C/C++ NCCL output as well as Python stdout/stderr, in this child only.
    for fd, stream in ((1, 'stdout'), (2, 'stderr')):
        with open(out / stream, 'w') as f: os.dup2(f.fileno(), fd)
    # Override only this child's log destination, never its transport policy.
    os.environ['NCCL_DEBUG_FILE'] = str(out / 'nccl-native.log')
    result = dict(rank=rank, status='FAIL', rank_namespace='Temporary two-member PyNccl test communicator', phase='imports',
                  native_abort_completion_verified=False)
    comm, dist = None, None
    def save(): (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def mark(phase):
        result['phase'] = phase
        (out / 'progress.json').write_text(json.dumps({'phase': phase, 'rank': rank, 'pid': os.getpid()}) + '\n')
    try:
        import numpy as np
        import torch
        import torch.distributed as dist
        from vllm.distributed.device_communicators import pynccl, pynccl_wrapper
        versions = json.loads((ROOT / 'kernel.json').read_text())['expected_versions']
        result['package_versions'] = {name: importlib.metadata.version(name) for name in versions}
        if any(result['package_versions'][name].split('+')[0] != wanted for name, wanted in versions.items()):
            raise RuntimeError('Reviewed runtime versions changed')
        for name, module in [('pynccl.py', pynccl), ('pynccl_wrapper.py', pynccl_wrapper)]:
            actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            result.setdefault('installed_api_sha256', {})[name] = actual
            if actual != config['expected_api_source_sha256'][name]: raise RuntimeError('Installed API source differs: ' + name)
        if torch.cuda.device_count() != 2: raise RuntimeError('Expected two visible GPUs')
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        props = torch.cuda.get_device_properties(rank)
        result.update(gpu_uuid=str(getattr(props, 'uuid', 'UNAVAILABLE')), gpu_name=props.name, device=str(device))
        if result['gpu_uuid'] != expected_ids[rank] or 'T4' not in props.name:
            raise RuntimeError('GPU identity changed since the accepted kernel fixture')
        library = pynccl_wrapper.NCCLLibrary()
        result.update(nccl_raw_version=library.ncclGetRawVersion(), nccl_version=library.ncclGetVersion(),
                      nccl_library_name=str(library.lib._name), bootstrap_backend='gloo/FileStore', payload_backend='vllm.PyNcclCommunicator/NCCL')
        if result['nccl_raw_version'] != config['expected_nccl_raw_version']:
            raise RuntimeError('Default vLLM NCCL library differs from the accepted private stack; no alternative tried')
        mark('gloo_bootstrap')
        dist.init_process_group('gloo', init_method=(Path(folder) / 'rendezvous').as_uri(), rank=rank,
                                world_size=2, timeout=dt.timedelta(seconds=config['group_timeout_seconds']))
        mark('native_NCCL_initialization_and_upstream_warmup')
        comm = pynccl.PyNcclCommunicator(dist.group.WORLD, device=device, library_path=library.lib._name)
        if not comm.available or comm.disabled or comm.world_size != 2 or comm.rank != rank:
            raise RuntimeError('Native communicator unavailable, disabled or misbound')
        if comm.nccl.lib is not library.lib: raise RuntimeError('NCCL library binding changed')
        count = config['elements_per_rank']
        local = torch.tensor(gather_input(rank, count), dtype=torch.float16, device=device)
        reduce_input = torch.tensor(scatter_input(rank, count), dtype=torch.float16, device=device)
        gathered = torch.full((2 * count,), float('nan'), dtype=torch.float16, device=device)
        scattered = torch.full((count,), float('nan'), dtype=torch.float16, device=device)
        mark('profiled_AllGather_then_ReduceScatter_SUM')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            comm.all_gather(gathered, local)
            comm.reduce_scatter(scattered, reduce_input, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(rank)
        if comm.disabled or not comm.available: raise RuntimeError('Communicator became unavailable')
        if gathered.device != device or scattered.device != device: raise RuntimeError('Payload left its CUDA device')
        result.update(exact_outputs(rank, count, gathered.cpu().tolist(), scattered.cpu().tolist()))
        np.save(out / 'all-gather.npy', gathered.cpu().numpy(), allow_pickle=False)
        np.save(out / 'reduce-scatter.npy', scattered.cpu().numpy(), allow_pickle=False)
        result['cuda_kernel_events'] = cuda_events(prof.events())
        result['nccl_cuda_event_count'] = sum(n for name, n in result['cuda_kernel_events'].items() if 'nccl' in name.lower())
        try: prof.export_chrome_trace(str(out / 'cuda-trace.json'))
        except Exception: result['trace_export_error'] = traceback.format_exc()
        result['application_buffer_bytes'] = {'all_gather_input': count * 2, 'all_gather_output': count * 4,
                                               'reduce_scatter_input': count * 4, 'reduce_scatter_output': count * 2}
        result['physical_transport'] = 'Not inferred from backend name; inspect raw NCCL logs. No bandwidth claim.'
        mark('post_collective_CPU_barrier')
        dist.barrier()
        if not result['all_gather_exact'] or not result['reduce_scatter_exact']:
            raise RuntimeError('Remote data or destination ownership mismatch')
        result['status'] = 'PASS' if result['nccl_cuda_event_count'] >= 2 and 'trace_export_error' not in result else 'PARTIAL'
        mark('cleanup')
    except Exception:
        result['error'] = traceback.format_exc(); traceback.print_exc(); save()
    finally:
        if comm is not None:
            try: comm.destroy(); result['native_destroy_returned'] = True
            except Exception: result.update(status='FAIL', native_cleanup_error=traceback.format_exc())
        if dist is not None and dist.is_initialized():
            try: dist.destroy_process_group(); result['gloo_destroy_returned'] = True
            except Exception: result.update(status='FAIL', gloo_cleanup_error=traceback.format_exc())
        sys.stdout.flush(); sys.stderr.flush()
        try:
            flush = ctypes.CDLL(None).fflush
            flush.argtypes = [ctypes.c_void_p]; flush.restype = ctypes.c_int
            result['native_stdio_flush_return'] = flush(None)
        except Exception: result['log_flush_error'] = traceback.format_exc()
        save()
    if result['status'] == 'FAIL': raise RuntimeError('Rank failed; see its result.json and stderr')


def coordinate(folder, config, expected_ids):
    os.environ.update(NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH,COLL,P2P,SHM,NET')
    result = dict(C_TWO_GPU_COMMUNICATION='FAIL', all_spawned_processes_joined=False,
                  scope=config['scope'], native_abort_completion='Wrapper returns after a bounded thread join; completion is not independently observed')
    try:
        import torch.multiprocessing as mp
        mp.spawn(rank_worker, args=(str(folder), config, expected_ids), nprocs=2, join=True)
        result['all_spawned_processes_joined'] = True
        result['ranks'] = [json.loads((Path(folder) / ('rank-' + str(r)) / 'result.json').read_text()) for r in (0, 1)]
        result['C_TWO_GPU_COMMUNICATION'] = aggregate(result['ranks'], expected_ids)
    except Exception: result['error'] = traceback.format_exc(); traceback.print_exc()
    (Path(folder) / 'coordinator.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def qualify(prior_archive, prior_sha256, output_root, execute=runtime.run):
    root = Path(output_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    out = root / ('moe-comm-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    out.mkdir()
    config = json.loads((ROOT / 'communication.json').read_text())
    status = dict(C_TWO_GPU_COMMUNICATION='BLOCKED', dependency_installation_attempted=False, later_stages='D/E/M NOT_EXECUTED', scope=config['scope'])
    commands = []
    try:
        if platform.system() != 'Linux' or platform.machine() != 'x86_64' or not Path('/kaggle/working').is_dir():
            raise RuntimeError('Requires the reviewed Kaggle Linux x86_64 runtime')
        data = Path(prior_archive).read_bytes()
        if not prior_sha256 or hashlib.sha256(data).hexdigest() != prior_sha256: raise RuntimeError('Prior B archive hash mismatch')
        with zipfile.ZipFile(prior_archive) as z:
            if z.testzip() is not None: raise RuntimeError('Prior B archive CRC failure')
            for line in z.read('SHA256SUMS').decode().splitlines():
                want, name = line.split('  ', 1)
                if hashlib.sha256(z.read(name)).hexdigest() != want: raise RuntimeError('Prior file hash mismatch')
            prior = json.loads(z.read('status.json'))
            if prior['B_FP16_MOE_KERNEL'] != 'PASS': raise RuntimeError('Prior kernel qualification has not passed')
            expected_ids = [w['torch_device_uuid'] for w in sorted(prior['workers'], key=lambda w: w['gpu_index'])]
            if len(expected_ids) != 2 or len(set(expected_ids)) != 2 or 'UNAVAILABLE' in expected_ids:
                raise RuntimeError('Two distinct accepted GPU identities required')
            py = Path(prior['isolated_python']); py.parent.parent.resolve().relative_to(root)
            if not py.is_file() or py.parent.name != 'bin': raise RuntimeError('Prior venv missing; no installation attempted')
        (out / Path(prior_archive).name).write_bytes(data)
        (out / 'communication.json').write_bytes((ROOT / 'communication.json').read_bytes())
        (out / 'expected-gpu-ids.json').write_text(json.dumps(expected_ids) + '\n')
        if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
            raise RuntimeError('Runner checkout changed')
        status['git_head'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines()
        (out / 'SOURCE_SHA256SUMS').write_text(''.join(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in tracked))
        status.update(prerequisite_archive_sha256=prior_sha256, isolated_python=str(py))
        command = execute('two-rank-NCCL', [str(py), '-B', str(ROOT / 'nccl_check.py'), '--coordinator', str(out)], config['coordinator_timeout_seconds'])
        for stream in ('stdout', 'stderr'): (out / ('coordinator.' + stream)).write_text(command[stream])
        commands.append({k: v for k, v in command.items() if k not in ('stdout', 'stderr')})
        if (out / 'coordinator.json').is_file():
            result = json.loads((out / 'coordinator.json').read_text())
            status['coordinator'] = result
            status['C_TWO_GPU_COMMUNICATION'] = result['C_TWO_GPU_COMMUNICATION']
            if command['exit_code'] != 0 and status['C_TWO_GPU_COMMUNICATION'] == 'PASS': status['C_TWO_GPU_COMMUNICATION'] = 'FAIL'
        else: status.update(C_TWO_GPU_COMMUNICATION='FAIL', error='Coordinator terminated without a receipt; partial rank logs/progress retained')
    except Exception: status['error'] = traceback.format_exc()
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    return dict(**runtime.seal(out), **status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinator')
    parser.add_argument('--prior-archive')
    parser.add_argument('--prior-sha256')
    parser.add_argument('--output-root', default='/kaggle/working')
    args = parser.parse_args()
    if args.coordinator:
        folder = Path(args.coordinator)
        result = coordinate(folder, json.loads((folder / 'communication.json').read_text()), json.loads((folder / 'expected-gpu-ids.json').read_text()))
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['C_TWO_GPU_COMMUNICATION'] in ('PASS', 'PARTIAL') else 1)
    if not args.prior_archive or not args.prior_sha256: parser.error('Exact prior B archive and SHA256 required')
    result = qualify(args.prior_archive, args.prior_sha256, args.output_root)
    print(json.dumps(result, indent=2))
    print('EVIDENCE_JSON:' + json.dumps(result, separators=(',', ':')))
