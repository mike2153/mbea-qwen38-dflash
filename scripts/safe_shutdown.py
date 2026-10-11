"""Fail-closed experiment shutdown. Never SIGKILL, force-remove or touch production."""
import json
import re
import time
from pathlib import Path

CRASH = re.compile(r'force killing remaining process|Segfault|core dumped|hipError|invalid host handle|GPU_RECOVERY_REQUIRED', re.I)


def stop_experiment(name, docker, log_path=None, timeout=45, clock=time.monotonic, sleep=time.sleep,
                    barrier=None, container_id=None):
    if not re.fullmatch(r'(?:exp-[a-z0-9-]+|qwen38-dflash)', name):
        raise ValueError('safe shutdown only supports managed server containers')

    def fail(message):
        raise RuntimeError('GPU_RECOVERY_REQUIRED: ' + message + '; container preserved')

    def state():
        rc, raw = docker('inspect', '-f', '{{json .}}', target, timeout=10)
        try:
            record = json.loads(raw)
            value = record['State']
        except (ValueError, KeyError, TypeError):
            fail('container state unavailable')
        if rc or not isinstance(value, dict) or 'Running' not in value:
            fail('container state unavailable')
        if record.get('Name') != '/' + name or (container_id and record.get('Id') != container_id):
            fail('container identity changed')
        return value

    # Signals target the recorded immutable ID, so reusing an experiment name cannot hit another boot.
    target = container_id or name
    if container_id and not re.fullmatch(r'[0-9a-f]{64}', container_id):
        fail('invalid container identity')
    current = state()
    if current['Running']:
        if barrier is None or container_id is None:
            fail('engine barrier/container identity missing; no termination signal sent')
        barrier()  # Must raise unless this boot acknowledged scheduler drain AND device completion.
        rc, logs = docker('logs', target, timeout=10)
        if rc or CRASH.search(logs):
            fail('abnormal worker logs before shutdown; no termination signal sent')
        # Unlike docker stop -t N this has no implicit SIGKILL deadline.
        rc, _ = docker('kill', '--signal=SIGTERM', target, timeout=10)
        if rc:
            fail('SIGTERM delivery failed')
        deadline = clock() + timeout
        while clock() < deadline:
            current = state()
            if not current['Running']:
                break
            sleep(.5)
        else:
            fail('shutdown stalled; no kill escalation')
    rc, logs = docker('logs', target, timeout=10)
    if log_path:
        Path(log_path).write_text(logs, encoding='utf-8')
    if current.get('ExitCode') != 0 or current.get('OOMKilled') or rc or CRASH.search(logs):
        fail(f'abnormal shutdown (exit={current.get("ExitCode")})')
    rc, _ = docker('rm', target, timeout=10)
    if rc:
        fail('clean container removal failed')
