import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import prepare_runtime as runtime


def report(sha=runtime.VLLM_SHA256, host='files.pythonhosted.org'):
    return {'install': [{'metadata': {'name': 'vllm', 'version': '0.31.0'},
        'download_info': {'url': 'https://' + host + '/packages/' + runtime.VLLM_WHEEL,
                          'archive_info': {'hashes': {'sha256': sha}}}}]}


class Fixture:
    """Synthetic orchestration results only; never executes pip or imports vLLM."""
    def __init__(self, fail=None, sha=runtime.VLLM_SHA256, changed_base=False, missing_gpu=False):
        self.labels = []
        self.fail, self.sha = fail, sha
        self.changed_base, self.missing_gpu = changed_base, missing_gpu

    def inventory(self, root):
        root = Path(root); root.mkdir(parents=True)
        gpus = [{'name': 'Tesla T4', 'uuid': 'SYNTHETIC-' + str(i), 'compute_capability': '7.5'} for i in range(1 if self.missing_gpu else 2)]
        (root / 'environment.json').write_text(json.dumps({'gpu_inventory': gpus}))
        return {'evidence_directory': str(root), 'qualification': {'A_environment': {'status': 'FAIL'}}}

    def execute(self, label, argv, timeout):
        self.labels.append(label)
        text = ''
        if label.startswith('base-'):
            text = json.dumps({'versions': {'torch': 'CHANGED' if label == 'base-after' and self.changed_base else 'SYNTHETIC'}})
        if label == 'install' and self.fail != label:
            Path(argv[argv.index('--report') + 1]).write_text(json.dumps(report(self.sha)))
        if label == 'isolated-inventory':
            text = 'EVIDENCE_JSON:' + json.dumps({'qualification': {'A_environment': {'status': 'PASS'}}})
        code = 1 if label == self.fail else 0
        return {'argv': argv, 'label': label, 'exit_code': code, 'timed_out': False,
                'stdout': text, 'stderr': 'SYNTHETIC first failure' if code else ''}


class RuntimeTests(unittest.TestCase):
    def prepare(self, root, fixture, **kwargs):
        with patch.object(runtime.platform, 'system', return_value='Linux'), \
             patch.object(runtime.platform, 'machine', return_value='x86_64'), \
             patch.object(Path, 'is_dir', return_value=True), \
             contextlib.redirect_stdout(io.StringIO()):
            return runtime.prepare(root, execute=fixture.execute, inventory=fixture.inventory, **kwargs)

    def resume_fixture(self, root):
        root = Path(root)
        name = 'moe-runtime-SYNTHETIC-previous'
        venv = root / (name + '-venv')
        (venv / 'bin').mkdir(parents=True)
        (venv / 'bin/python').write_text('SYNTHETIC NOT EXECUTABLE')
        files = {'status.json': json.dumps({'failed_step': 'native-import', 'venv_preserved_at': str(venv)}).encode(),
                 'commands.json': json.dumps([{'label': label, 'exit_code': 0} for label in ('install', 'pip-check')]).encode(),
                 'pip-install-report.json': json.dumps(report()).encode()}
        files['SHA256SUMS'] = ''.join(hashlib.sha256(data).hexdigest() + '  ' + n + '\n' for n, data in files.items()).encode()
        archive = root / (name + '.zip')
        with zipfile.ZipFile(archive, 'x') as z:
            for n, data in files.items(): z.writestr(n, data)
        return archive, hashlib.sha256(archive.read_bytes()).hexdigest()

    def test_unsupported_host_archives_refusal_without_install(self):
        fixture = Fixture()
        with tempfile.TemporaryDirectory() as root, patch.object(runtime.platform, 'system', return_value='Darwin'):
            result = runtime.prepare(root, execute=fixture.execute, inventory=fixture.inventory)
            self.assertEqual(result['runtime_preparation'], 'FAIL')
            self.assertEqual(fixture.labels, [])
            self.assertTrue(Path(result['archive']).is_file())

    def test_binary_only_install_and_no_ensurepip_or_model_stage(self):
        fixture = Fixture()
        with tempfile.TemporaryDirectory() as root:
            result = self.prepare(root, fixture)
            self.assertEqual(result['runtime_preparation'], 'PASS')
            with zipfile.ZipFile(result['archive']) as z:
                commands = json.loads(z.read('commands.json'))
                create = next(c for c in commands if c['label'] == 'create-venv')
                self.assertIn('--without-pip', create['argv'])
                installs = [c for c in commands if c['label'] in ('seed-pip', 'install')]
                self.assertEqual(len(installs), 2)
                self.assertTrue(all('--only-binary=:all:' in c['argv'] for c in installs))
                self.assertIn('B/C/D/E/M NOT_EXECUTED', result['scope'])
                self.assertFalse(any('-venv/' in name for name in z.namelist()))

    def test_first_install_failure_stops_and_retains_stderr(self):
        fixture = Fixture(fail='install')
        with tempfile.TemporaryDirectory() as root:
            result = self.prepare(root, fixture)
            self.assertEqual(result['failed_step'], 'install')
            self.assertEqual(result['runtime_preparation'], 'FAIL')
            self.assertNotIn('native-import', fixture.labels)
            self.assertEqual(fixture.labels.count('install'), 1)
            with zipfile.ZipFile(result['archive']) as z:
                self.assertIn(b'SYNTHETIC first failure', z.read('logs/install.stderr'))

    def test_bootstrap_failure_prevents_dependency_install(self):
        fixture = Fixture(fail='seed-pip')
        with tempfile.TemporaryDirectory() as root:
            result = self.prepare(root, fixture)
            self.assertEqual(result['failed_step'], 'seed-pip')
            self.assertNotIn('install', fixture.labels)

    def test_wrong_wheel_hash_stops_before_import(self):
        fixture = Fixture(sha='0' * 64)
        with tempfile.TemporaryDirectory() as root:
            result = self.prepare(root, fixture)
            self.assertEqual(result['runtime_preparation'], 'FAIL')
            self.assertIn('wheel differs', result['error'])
            self.assertNotIn('native-import', fixture.labels)

    def test_non_pypi_artifact_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-PyPI'):
            runtime.verify_install_report(report(host='unreviewed.invalid'))

    def test_one_gpu_prevents_install(self):
        fixture = Fixture(missing_gpu=True)
        with tempfile.TemporaryDirectory() as root:
            result = self.prepare(root, fixture)
            self.assertEqual(result['runtime_preparation'], 'FAIL')
            self.assertNotIn('create-venv', fixture.labels)

    def test_native_import_failure_is_not_runtime_pass(self):
        fixture = Fixture(fail='native-import')
        with tempfile.TemporaryDirectory() as root:
            result = self.prepare(root, fixture)
            self.assertEqual(result['failed_step'], 'native-import')
            self.assertEqual(result['runtime_preparation'], 'FAIL')
            self.assertNotIn('isolated-inventory', fixture.labels)

    def test_base_change_is_not_silently_accepted(self):
        fixture = Fixture(changed_base=True)
        with tempfile.TemporaryDirectory() as root:
            result = self.prepare(root, fixture)
            self.assertFalse(result['base_selected_versions_unchanged'])
            self.assertEqual(result['runtime_preparation'], 'FAIL')

    def test_subprocess_uses_isolated_index_configuration_without_secrets(self):
        code = "import os,json; print(json.dumps({k: os.environ.get(k) for k in ['PIP_CONFIG_FILE','PIP_EXTRA_INDEX_URL','PYTHONPATH','GH_TOKEN','HF_HUB_OFFLINE']}))"
        with patch.dict(os.environ, {'PIP_EXTRA_INDEX_URL': 'https://unreviewed.invalid',
                                     'PYTHONPATH': '/unreviewed', 'GH_TOKEN': 'SYNTHETIC_SECRET'}):
            result = runtime.run('env-test', [sys.executable, '-c', code], 10)
        self.assertEqual(result['exit_code'], 0)
        env = json.loads(result['stdout'])
        self.assertEqual(env['PIP_CONFIG_FILE'], os.devnull)
        self.assertIsNone(env['PIP_EXTRA_INDEX_URL'])
        self.assertIsNone(env['PYTHONPATH'])
        self.assertIsNone(env['GH_TOKEN'])
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')

    def test_timed_out_process_retains_partial_output(self):
        result = runtime.run('timeout-test', [sys.executable, '-u', '-c',
            "import time; print('SYNTHETIC partial output', flush=True); time.sleep(10)"], 0.15)
        self.assertEqual(result['exit_code'], 124)
        self.assertTrue(result['timed_out'])
        self.assertIn('SYNTHETIC partial output', result['stdout'])

    def test_resume_reuses_verified_attempt_without_installing_and_preserves_old_zip(self):
        fixture = Fixture()
        with tempfile.TemporaryDirectory() as root:
            archive, sha = self.resume_fixture(root)
            original = archive.read_bytes()
            result = self.prepare(root, fixture, resume_archive=archive, resume_sha256=sha)
            self.assertEqual(result['runtime_preparation'], 'PASS')
            self.assertFalse(result['installation_attempted'])
            self.assertEqual(result['resumed_from_archive_sha256'], sha)
            self.assertNotIn('create-venv', fixture.labels)
            self.assertNotIn('seed-pip', fixture.labels)
            self.assertNotIn('install', fixture.labels)
            self.assertIn('native-import', fixture.labels)
            self.assertEqual(archive.read_bytes(), original)
            with zipfile.ZipFile(result['archive']) as z:
                self.assertEqual(z.read(archive.name), original)

    def test_resume_wrong_receipt_hash_stops_before_any_command(self):
        fixture = Fixture()
        with tempfile.TemporaryDirectory() as root:
            archive, _ = self.resume_fixture(root)
            result = self.prepare(root, fixture, resume_archive=archive, resume_sha256='0' * 64)
            self.assertEqual(result['runtime_preparation'], 'FAIL')
            self.assertEqual(fixture.labels, [])

    def test_missing_old_venv_never_falls_back_to_install(self):
        fixture = Fixture()
        with tempfile.TemporaryDirectory() as root:
            archive, sha = self.resume_fixture(root)
            (Path(root) / (archive.stem + '-venv') / 'bin/python').unlink()
            result = self.prepare(root, fixture, resume_archive=archive, resume_sha256=sha)
            self.assertEqual(result['runtime_preparation'], 'FAIL')
            self.assertIn('no automatic reinstall', result['error'])
            self.assertEqual(fixture.labels, [])

    def test_pinned_cuda_check_succeeds_with_stable_extensions_and_no_legacy_module(self):
        # CPU-only negative control: valid pinned CUDA names exist, legacy _C does not.
        names = ('torch', 'vllm', 'triton', 'transformers',
                 'vllm._C_stable_libtorch', 'vllm._moe_C_stable_libtorch')
        modules = {n: types.ModuleType(n) for n in names}
        for module in modules.values():
            module.__version__ = 'SYNTHETIC'
            module.__file__ = '/SYNTHETIC/no-real-library-loaded'
        modules['torch'].__version__ = '2.13.0+SYNTHETIC'
        modules['vllm'].__version__ = '0.31.0'
        modules['vllm'].__path__ = []
        modules['vllm']._C_stable_libtorch = modules['vllm._C_stable_libtorch']
        modules['vllm']._moe_C_stable_libtorch = modules['vllm._moe_C_stable_libtorch']
        with patch.dict(sys.modules, modules), patch.object(sys, 'prefix', '/SYNTHETIC_VENV'), \
             contextlib.redirect_stdout(io.StringIO()):
            exec(runtime.NATIVE_IMPORT, {})
            obsolete = runtime.NATIVE_IMPORT.replace('vllm._C_stable_libtorch', 'vllm._C')
            with self.assertRaises(ModuleNotFoundError): exec(obsolete, {})


if __name__ == '__main__':
    unittest.main()
