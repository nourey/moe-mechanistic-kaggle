#!/usr/bin/env python3
"""Manual Kaggle 2x2 fixed-trace sensitivity block. No GPU execution at import."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import uuid
import zipfile

import model_qualify
import prepare_runtime as runtime
from load_experiment import CODE, hashes
from load_sensitivity import frozen_trace, comparison
from load_run import retained_result

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run_trials(out, prep, execute=runtime.run):
    config, _ = frozen_trace()
    started = time.monotonic()
    cells = []
    initial = hashes()
    result = dict(status='BLOCKED', cells=cells, condition_order=config['order'],
                  scope=config['scope'], original_evidence=config['original_archive_sha256'])
    try:
        for condition in config['order']:
            if hashes() != initial:
                raise ValueError('Source changed between cells')
            remaining = config['total_child_timeout_seconds'] - (time.monotonic() - started)
            if remaining < config['child_timeout_seconds']:
                raise RuntimeError('Block watchdog budget exhausted; no partial next cell')
            trial = out / condition['name']
            trial.mkdir()
            (trial / 'logs').mkdir()
            # Passive environmental record; do not alter clocks, GPU limits or other jobs.
            inventory = execute('gpu-before-' + condition['name'], ['nvidia-smi', '-q'], 15)
            write(trial / 'gpu-inventory-command.json', inventory)
            print('Running ' + condition['name'] + ': fixed warmup + original 85 offers', flush=True)
            receipt = execute(condition['name'], [prep['python'], '-u', '-B', str(ROOT / 'load_experiment.py'),
                '--output', str(trial), '--model', prep['model_path'], '--gpu-ids', json.dumps(prep['gpu_ids']),
                '--sensitivity-condition', condition['name']], config['child_timeout_seconds'])
            for stream in ('stdout', 'stderr'):
                (trial / 'logs' / ('load.' + stream)).write_text(model_qualify.safe_log(receipt.pop(stream)))
            write(trial / 'command.json', receipt)
            native = retained_result(trial, receipt)
            write(trial / 'status.json', native)
            if (native['status'] != 'PASS_FIXED_REPLAY' or native.get('error')
                    or native.get('owned_processes_remaining') or native.get('cleanup_unknown_processes')
                    or not native.get('shutdown_returned')):
                raise RuntimeError('STOP_' + condition['name'] + ': inspect retained status/logs; no later cell')
            cfg = json.loads((trial / 'configuration.json').read_text())
            if cfg['source'] != initial:
                raise ValueError('Child source provenance mismatch')
            loaded = [json.loads((trial / f'loaded-dp{r}.json').read_text()) for r in (0, 1)]
            cells.append(dict(name=condition['name'], condition=condition, status=native['status'],
                client=native['client_summary'], engine_args=cfg['engine_args'], effective_scheduler=cfg['effective_scheduler'],
                gpu_ids=[w['gpu_uuid'] for w in loaded], kv_capacity_blocks=[w['kv_capacity_blocks'] for w in loaded],
                native_sources=json.loads((trial / 'native-sources.json').read_text()),
                native_owner_audit='PASS' if native.get('native_owner_audit') == 'PASS_ACCOUNTING' else 'UNKNOWN_REFERENCE',
                raw_path=condition['name']))
            write(out / 'status.json', result)
        if hashes() != initial:
            raise ValueError('Source changed during final cell')
        result.update(comparison(cells, config))
    except Exception:
        result['error'] = traceback.format_exc()
        result['status'] = 'PARTIAL' if cells else 'BLOCKED'
    write(out / 'status.json', result)
    return result


def main():
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Manual Kaggle T4 x2 / Internet On only')
    frozen_trace()  # Fail before restoring a runtime if the calendar drifted.
    print('Reusing exact runtime/model; no kernel, NCCL or model qualification rerun.', flush=True)
    prep = model_qualify.qualify('/kaggle/working', prepare_only=True)
    if prep.get('runtime_and_checkpoint') != 'READY':
        print('EVIDENCE_JSON:' + json.dumps(dict(status='BLOCKED', archive=prep['archive'])))
        return
    out = Path('/kaggle/working') / ('moe-load-sensitivity-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    out.mkdir()
    (out / 'source').mkdir()
    shutil.copyfile(prep['archive'], out / 'runtime-preparation.zip')
    with zipfile.ZipFile(prep['archive']) as z:
        (out / 'SOURCE_COMMIT').write_bytes(z.read('logs/source-commit.stdout'))
    for name in CODE:
        shutil.copyfile(ROOT / name, out / 'source' / name)
    (out / 'SOURCE_SHA256SUMS').write_text(''.join(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in CODE))
    result = run_trials(out, prep)
    print('EVIDENCE_JSON:' + json.dumps(dict(status=result['status'], **runtime.seal(out))))


if __name__ == '__main__':
    main()
