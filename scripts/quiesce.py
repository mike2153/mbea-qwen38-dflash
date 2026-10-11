"""Authenticated experiment drain client. HTTP timeouts never signal or kill a worker."""
import datetime
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

PROTOCOL = 1
PATH = '/atlas/control'


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def validate_config(config, name):
    if not re.fullmatch(r'(?:exp-[a-z0-9-]+|qwen38-dflash)', name) or config.get('name') != name:
        raise ValueError('quiesce requires the matching managed server identity')
    url = urllib.parse.urlsplit(config['base'])
    if (url.scheme != 'http' or url.hostname not in ('localhost', '127.0.0.1')
            or url.port is None or not 1024 <= url.port <= 65535
            or url.path or url.query or url.fragment or url.username or url.password):
        raise ValueError('quiesce only supports local unprivileged ports')
    if not re.fullmatch(r'[0-9a-f]{64}', config['token']):
        raise ValueError('invalid control credential')


def request(config, body=None, timeout=45):
    # Do not follow a redirect with the private control header or use a system HTTP proxy.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(config['base'] + PATH, data=data,
                                 headers={'X-Atlas-Control': config['token'], 'Content-Type': 'application/json'})
    with opener.open(req, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError('control did not acknowledge')
        value = json.loads(response.read(16385))
    if not isinstance(value, dict):
        raise RuntimeError('control returned an invalid response')
    return value


def identity(config, name):
    validate_config(config, name)
    value = request(config, timeout=5)
    if (value.get('protocol') != PROTOCOL or value.get('experiment') != name
            or not re.fullmatch(r'[0-9a-f]{32}', str(value.get('boot_id')))):
        raise RuntimeError('experiment control identity mismatch')
    return value


def acknowledge(config, name, directory, timeout=40):
    """Pin this boot and nonce on disk; repeated calls poll the same irreversible barrier."""
    directory = Path(directory)
    try:
        if not 0 < timeout <= 45:
            raise ValueError('quiesce wait must be in (0, 45]')
        observed = identity(config, name)
        if observed['boot_id'] != config.get('boot_id'):
            raise RuntimeError('server boot changed since readiness')
        attempt_path = directory / 'quiesce-attempt.json'
        if attempt_path.exists():
            attempt = json.loads(attempt_path.read_text(encoding='utf-8'))
            if (attempt.get('boot_id') != config['boot_id'] or attempt.get('experiment') != name
                    or not re.fullmatch(r'[0-9a-f]{32}', str(attempt.get('nonce')))):
                raise RuntimeError('quiesce attempt belongs to a different boot or experiment')
        else:
            attempt = {'boot_id': config['boot_id'], 'experiment': name, 'nonce': uuid.uuid4().hex,
                       'requested_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            save(attempt_path, attempt)
        t0 = time.monotonic()
        result = request(config, {**attempt, 'wait_s': timeout}, timeout=timeout + 5)
        required = {'protocol': PROTOCOL, 'experiment': name, 'boot_id': config['boot_id'],
                    'nonce': attempt['nonce'], 'phase': 'quiesced', 'admission_closed': True,
                    'active_api_requests': 0, 'scheduler_paused': True, 'device_synchronized': True,
                    'workers': 1}
        if any(type(result.get(k)) is not type(v) or result[k] != v for k, v in required.items()):
            raise RuntimeError('incomplete or mismatched scheduler/device acknowledgement')
        save(directory / 'quiesce-ack.json', {**result, 'received_at':
             datetime.datetime.now(datetime.timezone.utc).isoformat(), 'wait_seconds': time.monotonic() - t0})
        return result
    except Exception as exc:
        raise RuntimeError(f'GPU_RECOVERY_REQUIRED: engine quiesce unconfirmed ({exc}); container preserved') from exc
