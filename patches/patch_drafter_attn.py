"""Route the DFlash2 drafter's attention through the bit-exact drafter_attn kernel.

Copies radiance_drafter_attn.py next to vLLM and appends a hook to the GPU worker. With
RADIANCE_DRAFTER_ATTN_A=1 the hook swaps the TRITON_ATTN reference before graph capture and loads the
kernel's code object right after the worker loads the model. A first launch inside graph capture crashes
the WSL runtime, so that warm-up is required. Without the switch the hook does nothing. Idempotent.
"""
from pathlib import Path
import shutil
import sys
import sysconfig

SENTINEL = 'drafter attention kernel (patch_drafter_attn.py)'
HOOK = f'''

# ---- {SENTINEL} ----
import os as _drafter_attn_os
if _drafter_attn_os.environ.get("RADIANCE_DRAFTER_ATTN_A") == "1":
    import functools as _drafter_attn_functools
    import radiance_drafter_attn as _drafter_attn
    _drafter_attn.install()
    _drafter_attn_load_model = Worker.load_model

    @_drafter_attn_functools.wraps(_drafter_attn_load_model)
    def _drafter_attn_load_and_warm(self, *args, **kwargs):
        result = _drafter_attn_load_model(self, *args, **kwargs)
        _drafter_attn.warmup()
        return result

    Worker.load_model = _drafter_attn_load_and_warm
'''


def patch(path):
    path = Path(path)
    source = path.read_text(encoding='utf-8')
    if SENTINEL in source:
        return
    if 'class Worker(' not in source:
        raise RuntimeError('unsupported GPU worker source; no file changed')
    path.write_text(source + HOOK, encoding='utf-8')


if __name__ == '__main__':
    site = Path(sysconfig.get_paths()['purelib'])
    shutil.copy(Path(__file__).with_name('radiance_drafter_attn.py'), site / 'radiance_drafter_attn.py')
    patch(sys.argv[1] if len(sys.argv) > 1 else site / 'vllm/v1/worker/gpu_worker.py')
    print('[patch_drafter_attn] applied')
