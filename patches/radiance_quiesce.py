"""Single-worker shutdown admission gate and acknowledged scheduler/device drain.

Loaded through --middleware and patch_quiesce.py. No torch import or device access on module load.
Once quiesce starts there is deliberately no resume endpoint. A stalled barrier preserves the worker.
"""
import asyncio
import hmac
import json
import os
import re
import uuid

PROTOCOL = 1
PATH = '/atlas/control'
HEX = re.compile(r'^[0-9a-f]{32}$')
READ_ONLY = {'/health', '/metrics', '/v1/models', '/version'}


def enabled():
    return os.environ.get('ATLAS_QUIESCE') == '1'


def experiment_config():
    name = os.environ.get('ATLAS_EXPERIMENT_NAME', '')
    token = os.environ.get('ATLAS_CONTROL_TOKEN', '')
    if not re.fullmatch(r'(?:exp-[a-z0-9-]+|qwen38-dflash)', name) or not re.fullmatch(r'[0-9a-f]{64}', token):
        raise RuntimeError('quiesce requires a managed server identity and a private control token')
    return name, token


def install_worker(worker_class):
    if not enabled():
        return
    experiment_config()
    if hasattr(worker_class, 'atlas_quiesce_sync'):
        return

    def atlas_quiesce_sync(self, nonce):
        if not isinstance(nonce, str) or not HEX.fullmatch(nonce):
            raise ValueError('invalid quiesce nonce')
        import torch
        # The RPC runs after scheduler pause. If device completion stalls, this never acknowledges.
        torch.cuda.synchronize(self.device)
        return {'protocol': PROTOCOL, 'nonce': nonce, 'device_synchronized': True,
                'worker_pid': os.getpid()}

    worker_class.atlas_quiesce_sync = atlas_quiesce_sync


async def respond(send, status, data):
    body = json.dumps(data).encode()
    await send({'type': 'http.response.start', 'status': status,
                'headers': [(b'content-type', b'application/json'), (b'cache-control', b'no-store'),
                            (b'content-length', str(len(body)).encode())]})
    await send({'type': 'http.response.body', 'body': body})


class ControlMiddleware:
    def __init__(self, app):
        self.app = app
        self.enabled = enabled()
        self.name, self.token = experiment_config() if self.enabled else ('', '')
        self.boot_id = uuid.uuid4().hex
        self.active = 0
        self.closed = False
        self.task = None
        self.nonce = None
        self.phase = 'accepting'

    def status(self):
        return {'protocol': PROTOCOL, 'experiment': self.name, 'boot_id': self.boot_id,
                'phase': self.phase, 'admission_closed': self.closed, 'active_api_requests': self.active}

    @staticmethod
    def validate_engine(engine):
        config = engine.vllm_config
        parallel = config.parallel_config
        if any(getattr(parallel, k, None) != 1 for k in
               ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size', '_api_process_count')):
            raise RuntimeError('quiesce supports the single-worker server only')
        if config.scheduler_config.async_scheduling:
            raise RuntimeError('async scheduling is outside the validated quiesce contract')
        for name in ('pause_generation', 'is_paused', 'collective_rpc'):
            if not callable(getattr(engine, name, None)):
                raise RuntimeError('engine lacks the native quiesce contract')
        if not hasattr(engine, 'output_processor'):
            raise RuntimeError('engine output tracking unavailable')

    def pending_outputs(self, engine):
        # Missing tracking is not interpreted as an empty request set.
        op = engine.output_processor
        return len(op.request_states) + len(op.parent_requests)

    async def drain(self, engine, nonce):
        try:
            self.phase = 'scheduler-abort'
            await engine.pause_generation(mode='abort', clear_cache=False)
            self.phase = 'api-drain'
            # An admitted request may still be tokenizing at the first pause. Keep aborting late
            # admissions while the permanent API gate rejects new ones. All work stays paused.
            while self.active or self.pending_outputs(engine):
                await asyncio.sleep(.02)
                await engine.pause_generation(mode='abort', clear_cache=False)
            self.phase = 'scheduler-drain'
            await engine.pause_generation(mode='abort', clear_cache=False)
            if not await engine.is_paused() or self.pending_outputs(engine) or self.active:
                raise RuntimeError('scheduler/API drain changed before device barrier')
            self.phase = 'device-sync'
            # No executor timeout that could kill a worker. HTTP callers may time out independently.
            replies = await engine.collective_rpc('atlas_quiesce_sync', args=(nonce,))
            if (not isinstance(replies, list) or len(replies) != 1
                    or not isinstance(replies[0], dict)
                    or replies[0].get('protocol') != PROTOCOL
                    or replies[0].get('nonce') != nonce
                    or replies[0].get('device_synchronized') is not True):
                raise RuntimeError('invalid device completion acknowledgement')
            if not await engine.is_paused() or self.pending_outputs(engine) or self.active:
                raise RuntimeError('scheduler/API drain changed after device barrier')
            self.phase = 'quiesced'
            return {**self.status(), 'nonce': nonce, 'scheduler_paused': True,
                    'device_synchronized': True, 'workers': 1}
        except BaseException:
            self.phase = 'failed'
            raise

    async def control(self, scope, receive, send):
        headers = dict(scope.get('headers', []))
        supplied = headers.get(b'x-atlas-control', b'')
        if not hmac.compare_digest(supplied, self.token.encode('ascii')):
            await respond(send, 401, {'error': 'control authentication required'})
            return
        try:
            engine = scope['app'].state.engine_client
            self.validate_engine(engine)
        except (AttributeError, KeyError, RuntimeError) as exc:
            await respond(send, 503, {'error': str(exc), **self.status()})
            return
        if scope['method'] == 'GET':
            await respond(send, 200, self.status())
            return
        if scope['method'] != 'POST':
            await respond(send, 405, {'error': 'GET or POST required'})
            return
        body = bytearray()
        while True:
            event = await receive()
            if event['type'] == 'http.disconnect':
                return
            body.extend(event.get('body', b''))
            if len(body) > 1024:
                await respond(send, 413, {'error': 'control request too large'})
                return
            if not event.get('more_body'):
                break
        try:
            request = json.loads(body)
            nonce = request['nonce']
            wait = request.get('wait_s', 30)
            if not isinstance(nonce, str) or not HEX.fullmatch(nonce):
                raise ValueError('nonce must be 32 lowercase hex characters')
            if type(wait) not in (int, float) or not 0 < wait <= 45:
                raise ValueError('wait_s must be in (0, 45]')
        except (TypeError, ValueError, KeyError):
            await respond(send, 400, {'error': 'invalid control request'})
            return
        if request.get('boot_id') != self.boot_id:
            await respond(send, 409, {'error': 'server boot changed'})
            return
        if self.task is not None and nonce != self.nonce:
            await respond(send, 409, {'error': 'quiesce already started with a different nonce'})
            return
        if self.task is None:
            self.closed, self.nonce = True, nonce
            self.task = asyncio.create_task(self.drain(engine, nonce))
            # Consume background exceptions if the HTTP caller has disconnected or timed out.
            self.task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        try:
            result = await asyncio.wait_for(asyncio.shield(self.task), timeout=wait)
            await respond(send, 200, result)
        except asyncio.TimeoutError:
            await respond(send, 504, {**self.status(), 'error': 'quiesce pending; preserve worker'})
        except Exception:
            await respond(send, 503, {**self.status(), 'error': 'quiesce failed; preserve worker'})

    async def __call__(self, scope, receive, send):
        if not self.enabled or scope['type'] not in ('http', 'websocket'):
            return await self.app(scope, receive, send)
        if scope['type'] == 'http' and scope['path'] == PATH:
            return await self.control(scope, receive, send)
        readonly = (scope['type'] == 'http' and scope['method'] in ('GET', 'HEAD')
                    and scope['path'] in READ_ONLY)
        if readonly:
            return await self.app(scope, receive, send)
        if self.closed:
            if scope['type'] == 'websocket':
                await send({'type': 'websocket.close', 'code': 1013})
            else:
                await respond(send, 503, {'error': 'server quiescing; admission closed'})
            return
        self.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.active -= 1
