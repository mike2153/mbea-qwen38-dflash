"""CPU-only admission, scheduler, device-barrier and real HTTP protocol tests."""
import asyncio
import importlib.util
import hashlib
import json
import os
import runpy
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / 'patches'
sys.path.insert(0, str(ROOT / 'scripts'))
import quiesce


def load(name):
    spec = importlib.util.spec_from_file_location(name, FORK / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOKEN = 'a' * 64
NONCE = 'b' * 32
ENV = {'ATLAS_QUIESCE': '1', 'ATLAS_EXPERIMENT_NAME': 'exp-test', 'ATLAS_CONTROL_TOKEN': TOKEN}
middleware = load('radiance_quiesce')


class Engine:
    def __init__(self):
        self.vllm_config = NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                             data_parallel_size=1, _api_process_count=1),
                             scheduler_config=NS(async_scheduling=False))
        self.output_processor = NS(request_states={}, parent_requests={})
        self.pause_gate = asyncio.Event()
        self.pause_gate.set()
        self.sync_gate = asyncio.Event()
        self.sync_gate.set()
        self.paused = False
        self.pause_count = 0
        self.sync_count = 0
        self.cancelled = False
        self.bad_reply = False

    async def pause_generation(self, *, mode, clear_cache):
        assert mode == 'abort' and clear_cache is False
        self.pause_count += 1
        await self.pause_gate.wait()
        self.output_processor.request_states.clear()
        self.output_processor.parent_requests.clear()
        self.paused = True

    async def is_paused(self):
        return self.paused

    async def collective_rpc(self, method, args):
        assert method == 'atlas_quiesce_sync' and self.paused
        self.sync_count += 1
        try:
            await self.sync_gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return [{'protocol': 1, 'nonce': '0' * 32 if self.bad_reply else args[0], 'device_synchronized': True}]


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = Engine()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

        async def app(scope, receive, send):
            if scope['path'] == '/v1/chat/completions':
                self.entered.set()
                await self.release.wait()
            await middleware.respond(send, 200, {'served': True})

        with patch.dict(os.environ, ENV):
            self.m = middleware.ControlMiddleware(app)
        self.tasks = []

    async def asyncTearDown(self):
        self.release.set()
        self.engine.pause_gate.set()
        self.engine.sync_gate.set()
        if self.m.task:
            await asyncio.gather(self.m.task, return_exceptions=True)
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

    async def call(self, path='/atlas/control', method='POST', data=None, token=TOKEN):
        events = []
        scope = {'type': 'http', 'method': method, 'path': path,
                 'headers': [(b'x-atlas-control', token.encode() if isinstance(token, str) else token)],
                 'app': NS(state=NS(engine_client=self.engine))}
        request = {'boot_id': self.m.boot_id, 'nonce': NONCE, 'wait_s': .05}
        if data:
            request.update(data)

        async def receive():
            return {'type': 'http.request', 'body': json.dumps(request).encode()}

        async def send(event):
            events.append(event)

        await self.m(scope, receive, send)
        return events[0]['status'], json.loads(events[1]['body'])

    async def test_healthy_idle_is_not_scheduler_or_device_ack(self):
        self.engine.pause_gate.clear()
        status, result = await self.call()
        self.assertEqual(status, 504)
        self.assertEqual(result['phase'], 'scheduler-abort')
        self.assertTrue(self.m.closed)
        self.assertEqual(self.engine.sync_count, 0)
        self.assertEqual((await self.call('/health', 'GET'))[0], 200)
        self.assertEqual((await self.call('/v1/chat/completions'))[0], 503)
        self.engine.pause_gate.set()
        status, result = await self.call()
        self.assertEqual(status, 200)
        self.assertTrue(result['device_synchronized'])

    async def test_admitted_tokenization_race_drains_before_sync(self):
        task = asyncio.create_task(self.call('/v1/chat/completions'))
        self.tasks.append(task)
        await self.entered.wait()
        self.assertEqual((await self.call())[0], 504)
        self.assertEqual(self.engine.sync_count, 0)
        # Work arriving after the first scheduler pause still has to be aborted and unwound.
        self.engine.output_processor.request_states['late-request'] = object()
        self.release.set()
        await task
        status, result = await self.call()
        self.assertEqual(status, 200)
        self.assertEqual(result['active_api_requests'], 0)
        self.assertGreaterEqual(self.engine.pause_count, 3)
        self.assertFalse(self.engine.output_processor.request_states)
        self.assertEqual(self.engine.sync_count, 1)
        self.assertEqual((await self.call('/resume'))[0], 503)

    async def test_stalled_device_survives_http_timeout_and_nonce_retry(self):
        self.engine.sync_gate.clear()
        self.assertEqual((await self.call())[0], 504)
        self.assertEqual(self.m.phase, 'device-sync')
        self.assertFalse(self.engine.cancelled)
        self.assertFalse(self.m.task.done())
        self.assertEqual((await self.call(data={'nonce': 'c' * 32}))[0], 409)
        self.engine.sync_gate.set()
        self.assertEqual((await self.call())[0], 200)
        self.assertEqual(self.engine.sync_count, 1)

    async def test_disconnected_control_caller_cannot_cancel_device_sync(self):
        self.engine.sync_gate.clear()
        caller = asyncio.create_task(self.call(data={'wait_s': 30}))
        while not self.engine.sync_count:
            await asyncio.sleep(0)
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.assertFalse(self.m.task.done())
        self.assertFalse(self.engine.cancelled)
        self.engine.sync_gate.set()
        self.assertEqual((await self.call())[0], 200)

    async def test_bad_worker_reply_fails_permanently_closed(self):
        self.engine.bad_reply = True
        self.assertEqual((await self.call())[0], 503)
        self.assertTrue(self.m.closed)
        self.assertEqual(self.m.phase, 'failed')
        self.assertEqual((await self.call('/v1/chat/completions'))[0], 503)
        self.assertEqual((await self.call())[0], 503)
        self.assertEqual(self.engine.sync_count, 1)

    async def test_auth_boot_and_config_rejection_do_not_begin_shutdown(self):
        for token in ('wrong', b'\xff'):
            self.assertEqual((await self.call(token=token))[0], 401)
        for data, status in (({'boot_id': 'c' * 32}, 409), ({'nonce': 'bad'}, 400), ({'wait_s': True}, 400)):
            self.assertEqual((await self.call(data=data))[0], status)
        for field in ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size', '_api_process_count'):
            config = self.engine.vllm_config.parallel_config
            setattr(config, field, 2)
            self.assertEqual((await self.call())[0], 503)
            setattr(config, field, 1)
        self.engine.vllm_config.scheduler_config.async_scheduling = True
        self.assertEqual((await self.call())[0], 503)
        self.assertFalse(self.m.closed)
        self.assertEqual(self.engine.pause_count, 0)

    async def test_real_http_client_matches_boot_nonce_and_device_ack(self):
        loop = asyncio.get_running_loop()
        protocol = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.process()

            def do_POST(self):
                self.process()

            def process(self):
                body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                status, data = asyncio.run_coroutine_threadsafe(protocol.call(
                    self.path, self.command, json.loads(body) if body else None,
                    self.headers.get('X-Atlas-Control', '')), loop).result(5)
                raw = json.dumps(data).encode()
                self.send_response(status)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = {'name': 'exp-test', 'base': f'http://127.0.0.1:{server.server_port}',
                      'token': TOKEN, 'boot_id': self.m.boot_id}
            with tempfile.TemporaryDirectory() as tmp:
                result = await asyncio.to_thread(quiesce.acknowledge, config, 'exp-test', tmp, .2)
                self.assertEqual(result['phase'], 'quiesced')
                evidence = json.loads((Path(tmp) / 'quiesce-ack.json').read_text())
                self.assertEqual(evidence['nonce'], self.m.nonce)
                self.assertEqual(self.engine.sync_count, 1)
                await asyncio.to_thread(quiesce.acknowledge, config, 'exp-test', tmp, .2)
                self.assertEqual(self.engine.sync_count, 1)
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join()


class InstallationTests(unittest.TestCase):
    def test_opt_in_idempotent_install_has_no_import_time_gpu_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            worker = site / 'vllm/v1/worker/gpu_worker.py'
            worker.parent.mkdir(parents=True)
            worker.write_text('class Worker(object): pass\n')
            with patch('sysconfig.get_paths', return_value={'purelib': tmp}), patch.dict(os.environ, {'ATLAS_QUIESCE': '0'}):
                runpy.run_path(str(FORK / 'patch_quiesce.py'), run_name='__main__')
            self.assertEqual(worker.read_text(), 'class Worker(object): pass\n')
            installer = load('patch_quiesce')
            with patch.object(installer, 'validate_source_contract'), patch.dict(os.environ, ENV):
                installer.install(tmp)
                installer.install(tmp)
                self.assertEqual(worker.read_text().count('# E007'), 1)
                torch = Mock()
                with patch.dict(sys.modules, {'torch': torch}):
                    module = load('radiance_quiesce')
                    class Worker:
                        device = 'test-device'
                    module.install_worker(Worker)
                    torch.cuda.synchronize.assert_not_called()
                    result = Worker().atlas_quiesce_sync(NONCE)
                    torch.cuda.synchronize.assert_called_once_with('test-device')
                    self.assertEqual(result['nonce'], NONCE)
                    self.assertTrue(result['device_synchronized'])

    def test_worker_sync_exception_never_acknowledges(self):
        class Worker:
            device = 'test'
        torch = Mock()
        torch.cuda.synchronize.side_effect = RuntimeError('device failure')
        with patch.dict(os.environ, ENV), patch.dict(sys.modules, {'torch': torch}):
            middleware.install_worker(Worker)
            with self.assertRaisesRegex(RuntimeError, 'device failure'):
                Worker().atlas_quiesce_sync(NONCE)

    def test_changed_native_method_fails_before_installation(self):
        installer = load('patch_quiesce')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'core.py'
            method = 'def pause(self):\n        return False'
            path.write_text('class Core:\n    ' + method + '\n')
            contract = {'core.py': {'Core': {'pause': hashlib.sha256(method.encode()).hexdigest()}}}
            installer.validate_source_contract(tmp, contract)
            path.write_text(path.read_text().replace('False', 'True'))
            with self.assertRaisesRegex(RuntimeError, 'source contract changed'):
                installer.validate_source_contract(tmp, contract)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.config = {'name': 'exp-test', 'base': 'http://127.0.0.1:8180', 'token': TOKEN, 'boot_id': NONCE}
        self.identity = {'protocol': 1, 'experiment': 'exp-test', 'boot_id': NONCE}

    def test_boot_mismatch_never_posts_and_bad_ack_is_never_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(quiesce, 'request', return_value={**self.identity, 'boot_id': 'c' * 32}) as req:
                with self.assertRaisesRegex(RuntimeError, 'boot changed'):
                    quiesce.acknowledge(self.config, 'exp-test', tmp)
                self.assertEqual(req.call_count, 1)
                self.assertFalse((Path(tmp) / 'quiesce-attempt.json').exists())
            with patch.object(quiesce, 'request', side_effect=[self.identity, self.identity]):
                with self.assertRaisesRegex(RuntimeError, 'mismatched'):
                    quiesce.acknowledge(self.config, 'exp-test', tmp)
                self.assertTrue((Path(tmp) / 'quiesce-attempt.json').exists())
                self.assertFalse((Path(tmp) / 'quiesce-ack.json').exists())

    def test_timeout_retains_attempt_for_same_nonce_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(2):
                with patch.object(quiesce, 'request', side_effect=[self.identity, TimeoutError('stalled')]):
                    with self.assertRaisesRegex(RuntimeError, 'container preserved'):
                        quiesce.acknowledge(self.config, 'exp-test', tmp)
                attempt = json.loads((Path(tmp) / 'quiesce-attempt.json').read_text())
                if _ == 0:
                    nonce = attempt['nonce']
                self.assertEqual(attempt['nonce'], nonce)
                self.assertFalse((Path(tmp) / 'quiesce-ack.json').exists())

    def test_production_ports_remote_hosts_and_wrong_name_rejected_without_http(self):
        for base in ('http://127.0.0.1:80', 'http://127.0.0.1:443', 'http://example.com:8180',
                     'http://127.0.0.1:8180/other', 'http://user@127.0.0.1:8180'):
            with patch.object(quiesce, 'request') as req, self.assertRaises(ValueError):
                quiesce.identity({**self.config, 'base': base}, 'exp-test')
            req.assert_not_called()
        with self.assertRaises(ValueError):
            quiesce.identity(self.config, 'qwen38-nvfp4-wsl')


if __name__ == '__main__':
    unittest.main()
