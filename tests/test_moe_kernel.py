import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import moe_kernel as kernel
from test_prepare_runtime import report


class KernelTests(unittest.TestCase):
    def test_comparison_rejects_nonfinite_or_wrong_outputs_and_degenerate_zeros(self):
        self.assertTrue(kernel.compare([1.0, -2.0], [1.01, -2.0], .02, 0)['passed'])
        self.assertFalse(kernel.compare([1.0], [2.0], .02, 0)['passed'])
        self.assertFalse(kernel.compare([float('nan')], [1.0], .02, 0)['passed'])
        self.assertFalse(kernel.compare([0.0], [0.0], .02, 0)['passed'])
        self.assertFalse(kernel.compare([0.0, 0.0], [1.0, -2.0], .02, 0)['passed'])
        # Equal aggregates do not excuse exchanging the two output positions.
        self.assertFalse(kernel.compare([1.0, 3.0], [3.0, 1.0], .02, 0)['passed'])
        with self.assertRaises(ValueError): kernel.compare([1.0], [1.0, 2.0], .02, 0)

    def test_CPU_label_cannot_fabricate_CUDA_kernel_evidence(self):
        events = [SimpleNamespace(device_type='DeviceType.CPU', name='fused_moe_kernel')]
        self.assertEqual(kernel.cuda_events(events), {})
        events += [SimpleNamespace(device_type='DeviceType.CUDA', name='fused_moe_kernel_a')] * 2
        self.assertEqual(kernel.cuda_events(events), {'fused_moe_kernel_a': 2})

    def prior(self, root):
        # Real venv Python links to a system interpreter; this is NOT a GPU fixture.
        venv = Path(root) / 'SYNTHETIC-venv'
        (venv / 'bin').mkdir(parents=True)
        (venv / 'bin/python').symlink_to(sys.executable)
        files = {'status.json': json.dumps({'runtime_preparation': 'PASS', 'isolated_python': str(venv / 'bin/python'), 'venv_preserved_at': str(venv)}).encode(),
                 'pip-install-report.json': json.dumps(report()).encode()}
        files['SHA256SUMS'] = ''.join(hashlib.sha256(b).hexdigest() + '  ' + n + '\n' for n, b in files.items()).encode()
        archive = Path(root) / 'SYNTHETIC-prior.zip'
        with zipfile.ZipFile(archive, 'x') as z:
            for name, data in files.items(): z.writestr(name, data)
        return archive, hashlib.sha256(archive.read_bytes()).hexdigest()

    def qualify(self, root, archive, sha, calls, first='PASS', changed_inputs=False, duplicate_uuid=False):
        git = subprocess.check_output
        def source_command(argv, **kwargs):
            if 'status' in argv: return ''
            return git(argv, **kwargs)
        def execute(label, argv, timeout):
            calls.append(argv)
            index = int(argv[argv.index('--worker') + 1])
            folder = Path(argv[argv.index('--worker-output') + 1]); folder.mkdir()
            state = first if index == 0 else 'PASS'
            receipt = {'status': state, 'gpu_index': index, 'torch_device_uuid': 'SYNTHETIC-' + str(0 if duplicate_uuid else index),
                       'input_sha256': {'x': 'CHANGED' if index == 1 and changed_inputs else 'SYNTHETIC-same'}}
            (folder / 'result.json').write_text(json.dumps(receipt))
            return {'label': label, 'argv': argv, 'exit_code': 0 if state == 'PASS' else 1, 'stdout': 'SYNTHETIC', 'stderr': ''}
        with patch.object(kernel.platform, 'system', return_value='Linux'), \
             patch.object(kernel.platform, 'machine', return_value='x86_64'), \
             patch.object(Path, 'is_dir', return_value=True), \
             patch.object(subprocess, 'check_output', side_effect=source_command), \
             contextlib.redirect_stdout(io.StringIO()):
            return kernel.qualify(archive, sha, root, execute=execute)

    def test_existing_symlinked_venv_is_used_without_install_or_model_commands(self):
        calls = []
        with tempfile.TemporaryDirectory() as root:
            archive, sha = self.prior(root)
            result = self.qualify(root, archive, sha, calls)
            self.assertEqual(result['B_FP16_MOE_KERNEL'], 'PASS')
            self.assertEqual(len(calls), 2)
            self.assertTrue(all('SYNTHETIC-venv/bin/python' in c[0] for c in calls))
            self.assertFalse(any('pip' in arg or 'serve' in arg for c in calls for arg in c))
            self.assertIn('C/D/E/M NOT_EXECUTED', result['later_stages'])
            with zipfile.ZipFile(result['archive']) as z:
                self.assertEqual(z.read(archive.name), archive.read_bytes())

    def test_failed_first_device_stops_before_second(self):
        calls = []
        with tempfile.TemporaryDirectory() as root:
            archive, sha = self.prior(root)
            result = self.qualify(root, archive, sha, calls, first='FAIL')
            self.assertEqual(result['B_FP16_MOE_KERNEL'], 'FAIL')
            self.assertEqual(len(calls), 1)

    def test_missing_GPU_trace_remains_partial_and_stops(self):
        calls = []
        with tempfile.TemporaryDirectory() as root:
            archive, sha = self.prior(root)
            result = self.qualify(root, archive, sha, calls, first='PARTIAL')
            self.assertEqual(result['B_FP16_MOE_KERNEL'], 'PARTIAL')
            self.assertEqual(len(calls), 1)

    def test_wrong_receipt_hash_prevents_execution(self):
        calls = []
        with tempfile.TemporaryDirectory() as root:
            archive, sha = self.prior(root)
            result = self.qualify(root, archive, '0' * 64, calls)
            self.assertEqual(result['B_FP16_MOE_KERNEL'], 'BLOCKED')
            self.assertEqual(calls, [])

    def test_changed_inputs_or_duplicate_GPU_identity_cannot_pass(self):
        for option in ['changed_inputs', 'duplicate_uuid']:
            with self.subTest(option=option), tempfile.TemporaryDirectory() as root:
                archive, sha = self.prior(root)
                result = self.qualify(root, archive, sha, [], **{option: True})
                self.assertNotEqual(result['B_FP16_MOE_KERNEL'], 'PASS')


if __name__ == '__main__': unittest.main()
