"""CPU negative controls; these never claim real model/GPU qualification."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import model_qualify as q
import native_model as native


def workers():
    result = []
    for rank in (0, 1):
        mapping = [-1] * 64
        for e in range(32): mapping[32 * rank + e] = e
        result.append(dict(dp_rank=rank, ep_rank=rank, pid=100 + rank, gpu_uuid='SYNTHETIC-' + str(rank),
            device_index=rank, tp_world_size=1, pp_world_size=1, dp_world_size=2, ep_world_size=2,
            model_type='olmoe', model_dtype='torch.float16', global_num_experts=64,
            ep_communication_manager='AgRsAll2AllManager',
            all2all_backend='allgather_reducescatter', kv_capacity_blocks=16,
            kv_tensors=[dict(is_cuda=True)], expert_layers=[dict(name='layer.' + str(i),
                is_cuda=True, dtype='torch.float16', w13_shape=[32, 2048, 2048],
                w2_shape=[32, 2048, 1024], global_to_local_map=mapping.copy()) for i in range(16)]))
    return result


class ResidencyTests(unittest.TestCase):
    def check(self, rows): return native.validate_residency(rows, ['SYNTHETIC-0', 'SYNTHETIC-1'])

    def test_disjoint_authoritative_expert_ownership(self):
        self.assertTrue(self.check(workers()))

    def test_aggregate_32_plus_32_is_insufficient_when_experts_overlap(self):
        rows = workers()
        for layer in rows[1]['expert_layers']: layer['global_to_local_map'] = rows[0]['expert_layers'][0]['global_to_local_map'].copy()
        with self.assertRaisesRegex(ValueError, 'Duplicate or orphan'): self.check(rows)

    def test_tp2_is_not_dp2_ep2(self):
        rows = workers(); rows[0]['tp_world_size'] = 2
        with self.assertRaisesRegex(ValueError, 'Not TP1'): self.check(rows)

    def test_wrong_physical_device_fails(self):
        rows = workers(); rows[1]['gpu_uuid'] = 'SYNTHETIC-0'
        with self.assertRaisesRegex(ValueError, 'Wrong GPU'): self.check(rows)

    def test_missing_rank_and_kv_capacity_fail(self):
        with self.assertRaises(ValueError): self.check(workers()[:1])
        rows = workers(); rows[1]['kv_capacity_blocks'] = 0
        with self.assertRaisesRegex(ValueError, 'KV capacity'): self.check(rows)

    def test_duplicate_layer_and_ep_rank_fail(self):
        rows = workers(); rows[1]['expert_layers'][1]['name'] = 'layer.0'
        with self.assertRaises(ValueError): self.check(rows)
        rows = workers(); rows[1]['ep_rank'] = 0
        with self.assertRaisesRegex(ValueError, 'EP rank'): self.check(rows)

    def test_terminal_missing_duplicate_or_abort_not_normal(self):
        specs = [dict(request_id='a', max_tokens=8), dict(request_id='b', max_tokens=8)]
        good = [dict(request_id=s['request_id'], finished=True, finish_reason='length', token_ids=[1, 2]) for s in specs]
        self.assertTrue(native.validate_terminals(good, specs))
        for bad in (good[:1], [good[0], good[0]], [dict(good[0], finish_reason='abort'), good[1]]):
            with self.assertRaises(ValueError): native.validate_terminals(bad, specs)

    def test_trace_joins_worker_pid_and_device_not_filename(self):
        rows = workers()
        events = [dict(cat='cpu_op', pid=100), dict(cat='kernel', name='fused_moe_kernel', args=dict(device=0)),
                  dict(cat='kernel', name='ncclAllGather', args=dict(device=0))]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'misleading-rank1.json.gz'
            with gzip.open(path, 'wt') as f: json.dump(dict(traceEvents=events), f)
            result = native.inspect_traces(d, rows)
            self.assertEqual([r['dp_rank'] for r in result], [0])
            self.assertEqual(result[0]['moe_calls'], 1)
            events[0]['pid'] = 999
            with gzip.open(path, 'wt') as f: json.dump(dict(traceEvents=events), f)
            self.assertEqual(native.inspect_traces(d, rows), [])


class AcquisitionTests(unittest.TestCase):
    def test_lfs_hash_is_not_its_git_pointer_object(self):
        data = b'SYNTHETIC weights'
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'weights'; path.write_bytes(data)
            row = dict(path='weights', bytes=len(data), lfs_sha256=hashlib.sha256(data).hexdigest(), git_blob_sha1='0' * 40)
            self.assertEqual(q.verify_model_file(path, row)['bytes'], len(data))
            with self.assertRaises(ValueError): q.verify_model_file(path, dict(row, lfs_sha256='0' * 64))

    def test_small_file_checks_git_object_and_length(self):
        data = b'{}'
        spec = dict(path='config', bytes=2, git_blob_sha1=hashlib.sha1(b'blob 2\0' + data).hexdigest())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config'; path.write_bytes(data)
            q.verify_model_file(path, spec)
            with self.assertRaises(ValueError): q.verify_model_file(path, dict(spec, bytes=3))
            with self.assertRaises(ValueError): q.verify_model_file(path, dict(spec, git_blob_sha1='0' * 40))

    def test_lock_is_exact_199_artifact_cohort(self):
        config = json.loads((q.ROOT / 'model-qualification.json').read_text())
        data = (q.ROOT / 'runtime-lock.txt').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), config['runtime_lock_sha256'])
        entries = q.lock_entries(data.decode())
        self.assertEqual(len(entries), 199)
        self.assertEqual(entries['vllm'][1], q.runtime.VLLM_SHA256)
        with self.assertRaises(ValueError): q.lock_entries('thing>=1')
        with self.assertRaises(ValueError): q.lock_entries('thing==1 --hash=sha256:' + '0' * 64 + '\nthing==2 --hash=sha256:' + '1' * 64)

    def test_gpu_preflight_rejects_cpu_one_gpu_low_memory_duplicate(self):
        lines = ['GPU-a, Tesla T4, 15360, 14912, 580.178.04', 'GPU-b, Tesla T4, 15360, 14912, 580.178.04']
        self.assertEqual(len(q.parse_gpu_inventory('\n'.join(lines))), 2)
        for bad in ('', lines[0], '\n'.join([lines[0], lines[0]]), '\n'.join(lines).replace('14912', '9000')):
            with self.assertRaises(ValueError): q.parse_gpu_inventory(bad)

    def test_cuda_order_can_differ_from_nvml_order(self):
        nv = q.parse_gpu_inventory('GPU-a, Tesla T4, 15360, 14912, 580\nGPU-b, Tesla T4, 15360, 14912, 580')
        rows = [dict(ordinal=0, uuid='b', capability=[7, 5]), dict(ordinal=1, uuid='a', capability=[7, 5])]
        self.assertEqual(q.validate_cuda_identity(rows, nv), ['b', 'a'])
        rows[0]['uuid'] = 'c'
        with self.assertRaises(ValueError): q.validate_cuda_identity(rows, nv)

    def test_signed_download_urls_and_tokens_are_not_published(self):
        text = q.safe_log('download https://host/file?X-Amz-Signature=SENSITIVE failed')
        self.assertNotIn('SENSITIVE', text)
        self.assertIn('QUERY_REDACTED', text)

    def test_missing_second_gpu_stops_before_install_and_preserves_evidence(self):
        calls = []
        def execute(label, argv, timeout):
            calls.append(label)
            stdout = {'source-status': '', 'base-before': '{"versions":{}}', 'base-after': '{"versions":{}}',
                      'gpu-preflight': 'GPU-a, Tesla T4, 15360, 14912, 580'}.get(label, 'SYNTHETIC')
            return dict(label=label, argv=argv, exit_code=0, stdout=stdout, stderr='')
        with tempfile.TemporaryDirectory() as d, patch.object(q.platform, 'system', return_value='Linux'), \
                patch.object(q.platform, 'machine', return_value='x86_64'), patch.object(q.sys, 'version_info', (3, 13)):
            result = q.qualify(d, execute=execute, work_root=d)
            self.assertEqual(result['D_MODEL_RESIDENCY'], 'BLOCKED')
            self.assertNotIn('install', calls)
            self.assertNotIn('model-acquisition', calls)
            self.assertTrue(Path(result['archive']).is_file())
            self.assertIn('two distinct T4', result['error'])


if __name__ == '__main__': unittest.main()
