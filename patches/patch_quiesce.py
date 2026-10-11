#!/usr/bin/env python3
"""Install the pinned, authenticated scheduler/device shutdown barrier."""
import ast
import hashlib
import json
import os
import shutil
import sysconfig
from pathlib import Path

SENTINEL = '# E007 acknowledged experiment quiesce (patch_quiesce.py)'


def validate_source_contract(site_packages, contract=None):
    """Refuse changed native drain semantics before modifying anything or initializing the GPU."""
    if contract is None:
        contract = json.loads(Path(__file__).with_name('quiesce_contract.json').read_text(encoding='utf-8'))
    for relative, classes in contract.items():
        source = (Path(site_packages) / relative).read_text(encoding='utf-8')
        tree = ast.parse(source)
        for cls, methods in classes.items():
            matches = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
            if len(matches) != 1:
                raise RuntimeError(f'quiesce source contract changed: {relative}:{cls}')
            for name, expected in methods.items():
                nodes = [n for n in matches[0].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
                if (len(nodes) != 1 or hashlib.sha256(ast.get_source_segment(source, nodes[0]).encode()).hexdigest() != expected):
                    raise RuntimeError(f'quiesce source contract changed: {cls}.{name}')


def install(site_packages):
    site = Path(site_packages)
    validate_source_contract(site)
    worker = site / 'vllm/v1/worker/gpu_worker.py'
    source = worker.read_text(encoding='utf-8')
    if 'class Worker(' not in source:
        raise RuntimeError('unsupported vLLM worker source')
    shutil.copy(Path(__file__).with_name('radiance_quiesce.py'), site / 'radiance_quiesce.py')
    if SENTINEL not in source:
        worker.write_text(source + '\n\n' + SENTINEL + '\nimport radiance_quiesce as _quiesce\n'
                          '_quiesce.install_worker(Worker)\n', encoding='utf-8')


if __name__ == '__main__' and os.environ.get('ATLAS_QUIESCE') == '1':
    install(sysconfig.get_paths()['purelib'])
