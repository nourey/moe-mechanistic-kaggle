import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import nccl_check as check


def rows():
    return [dict(rank=r, status='PASS', gpu_uuid='SYNTHETIC-' + str(r), all_gather_exact=True,
                 reduce_scatter_exact=True, native_destroy_returned=True, gloo_destroy_returned=True) for r in (0, 1)]


class CommunicationTests(unittest.TestCase):
    def test_known_rank_payloads_and_exact_reduction(self):
        self.assertEqual(check.gather_input(0, 2), [0, 1])
        self.assertEqual(check.gather_input(1, 2), [32, 33])
        self.assertEqual(check.scatter_input(0, 2), [16, 17, 32, 33])
        self.assertEqual(check.scatter_input(1, 2), [20, 21, 36, 37])
        self.assertEqual(check.expected_outputs(0, 2), ([0, 1, 32, 33], [36, 38]))
        self.assertEqual(check.expected_outputs(1, 2), ([0, 1, 32, 33], [68, 70]))

    def test_correct_global_sum_does_not_excuse_wrong_owner_or_peer_order(self):
        result = check.exact_outputs(0, 2, [32, 33, 0, 1], [68, 70])
        self.assertFalse(result['all_gather_exact'])
        self.assertFalse(result['reduce_scatter_exact'])
        result = check.exact_outputs(0, 2, [float('nan')] * 4, [float('nan')] * 2)
        self.assertFalse(result['all_gather_exact'])
        self.assertFalse(result['reduce_scatter_exact'])

    def test_missing_rank_wrong_identity_or_incomplete_cleanup_is_not_pass(self):
        good = rows(); ids = ['SYNTHETIC-0', 'SYNTHETIC-1']
        self.assertEqual(check.aggregate(good, ids), 'PASS')
        self.assertEqual(check.aggregate(good[:1], ids), 'FAIL')
        for key, value in [('gpu_uuid', 'SYNTHETIC-WRONG'), ('native_destroy_returned', False), ('all_gather_exact', False)]:
            changed = rows(); changed[1][key] = value
            self.assertEqual(check.aggregate(changed, ids), 'FAIL')

    def test_unavailable_GPU_trace_is_partial_not_pass(self):
        current = rows(); current[0]['status'] = 'PARTIAL'
        self.assertEqual(check.aggregate(current, ['SYNTHETIC-0', 'SYNTHETIC-1']), 'PARTIAL')

    def prior(self, root):
        venv = Path(root) / 'SYNTHETIC-venv'; (venv / 'bin').mkdir(parents=True)
        (venv / 'bin/python').symlink_to(sys.executable)
        status = dict(B_FP16_MOE_KERNEL='PASS', isolated_python=str(venv / 'bin/python'),
                      workers=[dict(gpu_index=r, torch_device_uuid='SYNTHETIC-' + str(r)) for r in (0, 1)])
        data = json.dumps(status).encode()
        archive = Path(root) / 'SYNTHETIC-B.zip'
        with zipfile.ZipFile(archive, 'x') as z:
            z.writestr('status.json', data)
            z.writestr('SHA256SUMS', hashlib.sha256(data).hexdigest() + '  status.json\n')
        return archive, hashlib.sha256(archive.read_bytes()).hexdigest()

    def qualify(self, root, archive, sha, calls, code=0):
        git = subprocess.check_output
        def source(argv, **kwargs):
            return '' if 'status' in argv else git(argv, **kwargs)
        def execute(label, argv, timeout):
            calls.append(argv)
            out = Path(argv[argv.index('--coordinator') + 1])
            (out / 'coordinator.json').write_text(json.dumps({'C_TWO_GPU_COMMUNICATION': 'PASS', 'ranks': rows()}))
            return dict(label=label, argv=argv, exit_code=code, stdout='SYNTHETIC', stderr='SYNTHETIC test-only')
        with patch.object(check.platform, 'system', return_value='Linux'), \
             patch.object(check.platform, 'machine', return_value='x86_64'), \
             patch.object(Path, 'is_dir', return_value=True), \
             patch.object(subprocess, 'check_output', side_effect=source), \
             contextlib.redirect_stdout(io.StringIO()):
            return check.qualify(archive, sha, root, execute=execute)

    def test_exact_existing_venv_and_prior_archive_are_reused_without_install(self):
        calls = []
        with tempfile.TemporaryDirectory() as root:
            archive, sha = self.prior(root)
            result = self.qualify(root, archive, sha, calls)
            self.assertEqual(result['C_TWO_GPU_COMMUNICATION'], 'PASS')
            self.assertEqual(len(calls), 1)
            self.assertIn('SYNTHETIC-venv/bin/python', calls[0][0])
            self.assertFalse(any('pip' in x or 'serve' in x for x in calls[0]))
            self.assertFalse(result['dependency_installation_attempted'])
            self.assertEqual(result['later_stages'], 'D/E/M NOT_EXECUTED')
            with zipfile.ZipFile(result['archive']) as z: self.assertEqual(z.read(archive.name), archive.read_bytes())

    def test_wrong_archive_hash_prevents_execution(self):
        calls = []
        with tempfile.TemporaryDirectory() as root:
            archive, sha = self.prior(root)
            result = self.qualify(root, archive, '0' * 64, calls)
            self.assertEqual(result['C_TWO_GPU_COMMUNICATION'], 'BLOCKED')
            self.assertEqual(calls, [])

    def test_nonzero_coordinator_exit_cannot_be_reported_as_pass(self):
        with tempfile.TemporaryDirectory() as root:
            archive, sha = self.prior(root)
            result = self.qualify(root, archive, sha, [], code=1)
            self.assertEqual(result['C_TWO_GPU_COMMUNICATION'], 'FAIL')


if __name__ == '__main__': unittest.main()
