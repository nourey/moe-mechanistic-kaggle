#!/usr/bin/env python3
"""Stage B: one real vLLM FP16 MoE call per GPU, no model or EP communication."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import uuid
import zipfile
import datetime as dt

import prepare_runtime as runtime

ROOT = Path(__file__).resolve().parent


def compare(actual, reference, atol, rtol):
    if len(actual) != len(reference) or not reference:
        raise ValueError('Missing or misaligned output population')
    invalid, errors, failures, zero_failures = 0, [], 0, 0
    for a, b in zip(actual, reference):
        if not math.isfinite(a) or not math.isfinite(b):
            invalid += 1
            continue
        error = abs(a - b)
        errors.append(error)
        failures += error > atol + rtol * abs(b)
        zero_failures += abs(b) > atol + rtol * abs(b)
    return dict(passed=invalid == 0 and failures == 0 and zero_failures > 0,
                nonfinite_pairs=invalid, mismatching_values=failures,
                max_abs_error=max(errors) if errors else None,
                all_zero_negative_control_rejected=zero_failures > 0)


def cuda_events(events):
    # A CPU record_function label containing 'fused_moe_kernel' is not GPU evidence.
    counts = Counter()
    for event in events:
        kind = getattr(event.device_type, 'name', str(event.device_type).rsplit('.', 1)[-1])
        if kind == 'CUDA': counts[event.name] += 1
    return dict(counts)


def cpu_reference(torch, x, w13, w2, weights, ids):
    """Independent full-FP32 CPU formula from the same already-rounded FP16 inputs."""
    f = torch.nn.functional
    out = torch.zeros(x.shape, dtype=torch.float32, device='cpu')
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate, up = f.linear(x[token].float(), w13[expert].float()).chunk(2)
            value = f.linear(f.silu(gate) * up, w2[expert].float())
            out[token] += float(weights[token, slot]) * value
    return out


def worker(index, out, config):
    out = Path(out); out.mkdir(exist_ok=False)
    result = dict(status='FAIL', gpu_index=index, scope='Single-device FP16 MoE; not an EP rank or serving engine',
                  phase='imports', timing_use='Functional qualification only; includes first-call JIT/profiling')
    workspace_started = False
    try:
        import numpy as np
        import torch
        import vllm
        import triton
        import vllm._C_stable_libtorch
        import vllm._moe_C_stable_libtorch
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

        versions = {name: importlib.metadata.version(name) for name in config['expected_versions']}
        result['package_versions'] = versions
        if any(versions[name].split('+')[0] != wanted for name, wanted in config['expected_versions'].items()):
            raise RuntimeError('Reviewed runtime versions changed; no fallback attempted')
        if any(observed.split('+')[0] != config['expected_versions'][name] for name, observed in
               [('torch', torch.__version__), ('vllm', vllm.__version__), ('triton', triton.__version__)]):
            raise RuntimeError('Imported modules disagree with the reviewed versions')
        if torch.cuda.device_count() != 2: raise RuntimeError('Expected two visible GPU devices')
        torch.cuda.set_device(index)
        props = torch.cuda.get_device_properties(index)
        if 'T4' not in props.name or (props.major, props.minor) != (7, 5):
            raise RuntimeError('Unexpected GPU domain')
        result.update(torch=torch.__version__, vllm=vllm.__version__, triton=triton.__version__,
                      gpu_name=props.name, torch_device_uuid=str(getattr(props, 'uuid', 'UNAVAILABLE')),
                      cuda_build=torch.version.cuda,
                      module_paths={'fused_experts': sys.modules[fused_experts.__module__].__file__,
                                    'native': vllm._C_stable_libtorch.__file__,
                                    'moe_native': vllm._moe_C_stable_libtorch.__file__})
        result['installed_fused_experts_source_sha256'] = hashlib.sha256(Path(result['module_paths']['fused_experts']).read_bytes()).hexdigest()
        result['phase'] = 'fixture_and_CPU_reference'
        torch.set_num_threads(2)
        m, e, k, h, n = (config[key] for key in ('tokens', 'experts', 'top_k', 'hidden_size', 'intermediate_size'))
        if m * k != e: raise ValueError('Fixture must cover every expert exactly once')
        generator = torch.Generator(device='cpu').manual_seed(config['seed'])
        x = torch.randn((m, h), generator=generator).half()
        w13 = (torch.randn((e, 2 * n, h), generator=generator) / math.sqrt(h)).half()
        w2 = (torch.randn((e, h, n), generator=generator) / math.sqrt(n)).half()
        logits = torch.full((m, e), -2.0, dtype=torch.float32)
        for token in range(m): logits[token, token * k:(token + 1) * k] = torch.linspace(2.0, 2.7, k)
        weights, ids = torch.softmax(logits, dim=-1).topk(k, dim=-1)
        ids = ids.to(torch.int32)
        tensors = dict(x=x, w13=w13, w2=w2, topk_weights=weights, topk_ids=ids)
        result['input_sha256'] = {name: hashlib.sha256(memoryview(t.numpy()).cast('B')).hexdigest() for name, t in tensors.items()}
        result['routing'] = {'expert_ids': ids.tolist(), 'weights': weights.tolist(),
                             'source': 'Known synthetic CPU routing, not observed model routing'}
        reference = cpu_reference(torch, x, w13, w2, weights, ids)
        np.save(out / 'cpu-reference.npy', reference.numpy(), allow_pickle=False)
        result['phase'] = 'GPU_allocation'
        device = torch.device('cuda', index)
        torch.cuda.reset_peak_memory_stats(index)
        x_gpu, w13_gpu, w2_gpu, weights_gpu, ids_gpu = (t.to(device) for t in tensors.values())
        init_workspace_manager(device); workspace_started = True
        result['phase'] = 'profiled_vllm_fused_experts'
        with set_current_vllm_config(VllmConfig()), torch.inference_mode():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA]) as prof:
                actual = fused_experts(x_gpu, w13_gpu, w2_gpu, weights_gpu, ids_gpu,
                                       global_num_experts=e)
                torch.cuda.synchronize(index)
        if actual.device != device or actual.dtype != torch.float16 or list(actual.shape) != [m, h]:
            raise RuntimeError('Wrong output device, dtype or shape')
        values = actual.float().cpu()
        np.save(out / 'gpu-output.npy', values.numpy(), allow_pickle=False)
        result['phase'] = 'comparison_and_trace'
        result['comparison'] = compare(values.flatten().tolist(), reference.flatten().tolist(), config['atol'], config['rtol'])
        result['cuda_kernel_events'] = cuda_events(prof.events())
        result['fused_moe_cuda_event_count'] = sum(count for name, count in result['cuda_kernel_events'].items() if 'fused_moe_kernel' in name)
        prof.export_chrome_trace(str(out / 'cuda-trace.json'))
        result['peak_torch_allocator_bytes'] = torch.cuda.max_memory_allocated(index)
        if not result['comparison']['passed']:
            raise RuntimeError('FP16 MoE output failed the fixed CPU reference/negative-control criteria')
        if result['fused_moe_cuda_event_count'] < config['minimum_fused_moe_cuda_events']:
            result.update(status='PARTIAL', reason='Numerical comparison passed; required CUDA MoE event identity/count unavailable')
        else:
            result.update(status='PASS', phase='complete')
    except Exception:
        result['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if workspace_started:
            try: reset_workspace_manager()
            except Exception: result.update(status='FAIL', cleanup_error=traceback.format_exc())
        (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def qualify(prior_archive, prior_sha256, output_root, execute=runtime.run):
    root = Path(output_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    out = root / ('moe-kernel-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    out.mkdir(); (out / 'logs').mkdir()
    config = json.loads((ROOT / 'kernel.json').read_text())
    status = dict(B_FP16_MOE_KERNEL='BLOCKED', workers=[], later_stages='C/D/E/M NOT_EXECUTED', dependency_installation_attempted=False,
                  scope='Synthetic local FP16 MoE kernel only; no pretrained model, EP2 communication or serving')
    commands = []
    try:
        if platform.system() != 'Linux' or platform.machine() != 'x86_64' or not Path('/kaggle/working').is_dir():
            raise RuntimeError('Requires the reviewed Kaggle Linux x86_64 environment')
        data = Path(prior_archive).read_bytes()
        if not prior_sha256 or hashlib.sha256(data).hexdigest() != prior_sha256:
            raise RuntimeError('Prior runtime evidence SHA256 mismatch')
        with zipfile.ZipFile(prior_archive) as z:
            if z.testzip() is not None: raise RuntimeError('Prior archive CRC failure')
            for line in z.read('SHA256SUMS').decode().splitlines():
                want, name = line.split('  ', 1)
                if hashlib.sha256(z.read(name)).hexdigest() != want: raise RuntimeError('Prior file hash mismatch: ' + name)
            prior = json.loads(z.read('status.json'))
            if prior['runtime_preparation'] != 'PASS': raise RuntimeError('Stage A runtime has not passed')
            runtime.verify_install_report(json.loads(z.read('pip-install-report.json')))
            py = Path(prior['isolated_python'])
            # The venv directory must be inside the working root. Its Python executable
            # normally symlinks OUTSIDE that directory; preserve its invocation path.
            py.parent.parent.resolve().relative_to(root)
            if py.parent.name != 'bin' or not py.is_file(): raise RuntimeError('Previous venv unavailable; no installation attempted')
            if py.parent.parent.resolve() != Path(prior['venv_preserved_at']).resolve():
                raise RuntimeError('Prior interpreter/venv identities disagree')
        (out / Path(prior_archive).name).write_bytes(data)
        (out / 'kernel.json').write_bytes((ROOT / 'kernel.json').read_bytes())
        commands.append({'label': 'source', 'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()})
        tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines()
        (out / 'SOURCE_SHA256SUMS').write_text(''.join(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in tracked))
        if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
            raise RuntimeError('Runner checkout changed; refusing kernel execution')
        status.update(prerequisite_archive_sha256=prior_sha256, isolated_python=str(py))
        for index in config['gpu_indices']:
            folder = out / ('gpu-' + str(index))
            label = 'gpu-' + str(index)
            print('Running ' + label + ' FP16 MoE check ...', flush=True)
            result = execute(label, [str(py), '-B', str(ROOT / 'moe_kernel.py'), '--worker', str(index),
                                    '--worker-output', str(folder)], config['worker_timeout_seconds'])
            for stream in ('stdout', 'stderr'): (out / 'logs' / (label + '.' + stream)).write_text(result[stream])
            commands.append({k: v for k, v in result.items() if k not in ('stdout', 'stderr')})
            receipt = json.loads((folder / 'result.json').read_text()) if (folder / 'result.json').is_file() else dict(status='FAIL', reason='Worker terminated without a receipt')
            if receipt.get('gpu_index') != index and receipt.get('status') == 'PASS':
                receipt.update(status='FAIL', reason='Worker device identity does not match its invocation')
            status['workers'].append(receipt)
            if result['exit_code'] != 0 or receipt['status'] != 'PASS':
                status['B_FP16_MOE_KERNEL'] = receipt['status'] if receipt['status'] in ('FAIL', 'PARTIAL') else 'FAIL'
                break
        else:
            if len(status['workers']) != 2 or status['workers'][0]['input_sha256'] != status['workers'][1]['input_sha256']:
                raise RuntimeError('GPU trials did not use identical input/routing tensors')
            identities = [w.get('torch_device_uuid', 'UNAVAILABLE') for w in status['workers']]
            if 'UNAVAILABLE' not in identities and len(set(identities)) != 2:
                raise RuntimeError('Duplicate GPU UUIDs in supposedly distinct device trials')
            status['B_FP16_MOE_KERNEL'] = 'PASS'
    except Exception:
        status['error'] = traceback.format_exc()
    (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    return dict(**runtime.seal(out), **status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=[0, 1])
    parser.add_argument('--worker-output')
    parser.add_argument('--prior-archive')
    parser.add_argument('--prior-sha256')
    parser.add_argument('--output-root', default='/kaggle/working')
    args = parser.parse_args()
    if args.worker is not None:
        if not args.worker_output: parser.error('--worker-output required')
        receipt = worker(args.worker, args.worker_output, json.loads((ROOT / 'kernel.json').read_text()))
        print(json.dumps(receipt, indent=2))
        raise SystemExit(0 if receipt['status'] == 'PASS' else 1)
    if not args.prior_archive or not args.prior_sha256: parser.error('Exact prior archive and SHA256 required')
    receipt = qualify(args.prior_archive, args.prior_sha256, args.output_root)
    print(json.dumps(receipt, indent=2))
    print('EVIDENCE_JSON:' + json.dumps(receipt, separators=(',', ':')))
