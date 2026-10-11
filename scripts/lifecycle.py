"""Identity-bound server lifecycle. A failed drain preserves the container and its evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import time
import urllib.request

import quiesce
from safe_shutdown import stop_experiment

LABEL = 'io.mbea.qwen38.managed'


def docker(*args, timeout=30):
    result = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout + result.stderr


def validate(name, port):
    if not re.fullmatch(r'(?:exp-[a-z0-9-]+|qwen38-dflash)', name) or not 1024 <= port <= 65535:
        raise ValueError('Use qwen38-dflash or exp-* and an unprivileged port')


def inspect(name):
    rc, text = docker('inspect', name)
    if rc:
        if 'No such object' in text or 'No such container' in text:
            return None
        raise RuntimeError('Docker inspection failed; no lifecycle action taken')
    value = json.loads(text)[0]
    if value['Name'] != '/' + name or value['Config'].get('Labels', {}).get(LABEL) != '1':
        raise RuntimeError('Existing container is not managed by this release; preserved')
    return value


def active(root):
    pointer = root / 'active.json'
    if not pointer.exists():
        raise RuntimeError('Lifecycle identity missing; container preserved')
    run = root / json.loads(pointer.read_text())['run']
    return run, json.loads((run / 'control.json').read_text())


def prepare(root, name, port):
    previous = inspect(name)
    if previous is not None:
        if previous['State']['Running']:
            raise RuntimeError('Container already exists; stop it cleanly before starting again')
        # Only this installation's recorded, clean exit may be removed. Bad exits remain.
        stop(root, name)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    run = root / (time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(4))
    run.mkdir(mode=0o700)
    config = {'name': name, 'base': f'http://127.0.0.1:{port}', 'token': secrets.token_hex(32)}
    quiesce.save(run / 'control.json', config)
    os.chmod(run / 'control.json', 0o600)
    env = run / 'control.env'
    with env.open('x') as file:
        file.write(f'ATLAS_QUIESCE=1\nATLAS_EXPERIMENT_NAME={name}\nATLAS_CONTROL_TOKEN={config["token"]}\n')
    os.chmod(env, 0o600)
    quiesce.save(root / 'active.json', {'run': run.name})
    return str(env)


def record(root, name):
    run, config = active(root)
    item = inspect(name)
    if not item or not re.fullmatch(r'[0-9a-f]{64}', item['Id']):
        raise RuntimeError('Started container identity unavailable; preserved')
    config['container_id'] = item['Id']
    quiesce.save(run / 'control.json', config)


def ready(root, name, minimum=180000):
    run, config = active(root)
    item = inspect(name)
    if not item or item['Id'] != config['container_id']:
        raise RuntimeError('Container identity changed during startup')
    identity = quiesce.identity(config, name)
    if identity.get('phase') != 'accepting' or identity.get('admission_closed') is not False:
        raise RuntimeError('Server is not accepting requests; finish its clean shutdown')
    config['boot_id'] = identity['boot_id']
    quiesce.save(run / 'control.json', config)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(config['base'] + '/v1/models', timeout=10) as response:
        models = json.load(response)
    context = int(models['data'][0].get('max_model_len') or 0)
    quiesce.save(run / 'readiness.json', {'context': context, 'boot_id': identity['boot_id']})
    if context < minimum:
        stop(root, name)
        raise RuntimeError(f'Context {context} is below {minimum}; cleanly stopped. Warm caches, then retry.')
    return context


def stop(root, name):
    item = inspect(name)
    if item is None:
        return
    run, config = active(root)
    if config.get('name') != name or config.get('container_id') != item['Id']:
        raise RuntimeError('Lifecycle identity mismatch; container preserved')
    stop_experiment(name, docker, log_path=run / 'container.log', container_id=item['Id'],
                    barrier=lambda: quiesce.acknowledge(config, name, run))
    quiesce.save(run / 'stopped.json', {'clean_exit': True, 'container_id': item['Id']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'record', 'ready', 'warmup-ready', 'stop'])
    parser.add_argument('data', type=Path)
    parser.add_argument('name')
    parser.add_argument('port', type=int)
    args = parser.parse_args()
    validate(args.name, args.port)
    root = args.data / 'runtime' / args.name
    if args.command == 'prepare':
        print(prepare(root, args.name, args.port))
    elif args.command == 'record':
        record(root, args.name)
    elif args.command in ('ready', 'warmup-ready'):
        minimum = 0 if args.command == 'warmup-ready' else 180000
        print(f'Available context: {ready(root, args.name, minimum):,} tokens')
    else:
        stop(root, args.name)


if __name__ == '__main__':
    main()
