#!/usr/bin/env python3
"""T02-M02 stage A (experiment fork, LEDGER L-034): drop two dead kernels per GDN layer, bit-exact.

RADIANCE_GDN_NOFILL=1 (default off, the file is then untouched):
  * `core_attn_out.zero_()` in QwenGDN._forward_core_rocm becomes a pending flag; radiance_gdn.forward_core_fused
    zeroes the buffer itself before any kernel on every path EXCEPT the fused decode update, which writes all rows
    [0, cu[N]) and zeroes the pad rows itself (rx5 kernel; vLLM PR 28182 is why the fill exists).
  * `z_out[:] = z` is skipped; radiance_gdnmerge passes the strided z slice of the in_proj output straight to the
    norm-quant (which takes a row stride, 16480 % 8 == 0).
Idempotent; FATAL if the anchors moved.
"""
import os
import sys
import sysconfig
from pathlib import Path

if os.environ.get("RADIANCE_GDN_NOFILL", "0") != "1":
    print("[patch_gdn_nofill] off (RADIANCE_GDN_NOFILL != 1)")
    sys.exit(0)

P = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
SENT = "RADIANCE_GDN_NOFILL"
FILL = "        core_attn_out.zero_()\n        num_tokens_all = qkvz.shape[0]\n"
COPY = "        z_out[:] = z\n        self._forward_core(\n"
src = P.read_text()
if SENT in src:
    print(f"[patch_gdn_nofill] already applied: {P}")
    sys.exit(0)
if src.count(FILL) != 1 or src.count(COPY) != 1:
    print(f"[patch_gdn_nofill] FATAL: anchors not found exactly once in {P}", file=sys.stderr)
    sys.exit(1)
src = src.replace(FILL, "        # RADIANCE_GDN_NOFILL: radiance_gdn.forward_core_fused zeroes it unless the fused update runs\n"
                        "        self._rad_pending_zero = True\n        num_tokens_all = qkvz.shape[0]\n", 1)
src = src.replace(COPY, "        # RADIANCE_GDN_NOFILL: the caller reads z as a strided view of qkvz; no copy\n"
                        "        self._forward_core(\n", 1)
P.write_text(src)
print(f"[patch_gdn_nofill] applied: {P}")
