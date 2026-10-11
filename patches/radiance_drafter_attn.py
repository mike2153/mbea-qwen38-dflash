"""E004 / T03-M03 (experiment fork, LEDGER L-036): the drafter's attention on our bit-exact kernel.

RADIANCE_DRAFTER_ATTN_A=1: vLLM's TRITON_ATTN backend (used only by the DFlash2 drafter in this serve; the target is
on R4D) calls unified_attention; this swaps the backend's reference for a wrapper that launches drafter_attn from
site-packages/drafter_attn.so (built at container start from /engine/kernels/drafter_attn.hip) for the drafter's call
shape — non-causal sliding window, fp8 per-tensor KV, no softcap/alibi/sinks/descale-q — and falls back to Triton for
anything else or if the launcher rejects the geometry. Installed before graph capture (radiance_atlas.install_worker).
"""
import ctypes
import os
import sys
import sysconfig

import torch

_S = {"lib": None, "orig": None, "ours": 0, "fallback": 0}


def _lib():
    if _S["lib"] is None:
        so = os.environ.get("RADIANCE_DRAFTER_ATTN_SO") or os.path.join(sysconfig.get_paths()["purelib"], "drafter_attn.so")
        lib = ctypes.CDLL(so)
        P, F, I, LL = ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_longlong
        lib.drafter_attn.argtypes = [P] * 9 + [F] + [I] * 7 + [LL] * 11 + [P]   # as engine/tests/test_drafter_attn.py
        lib.drafter_attn.restype = I
        _S["lib"] = lib
        rc = lib.drafter_attn_warmup()  # load the code object before any capture (E002 lesson)
        if rc:
            _S["lib"] = None
            raise RuntimeError(f"drafter_attn pre-capture warmup failed: rc={rc}")
        sys.stderr.write(f"[radiance.drafter_attn] kernel loaded from {so} (warmup rc={rc})\n")
    return _S["lib"]


def _wrapped(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal, window_size,
             block_table, softcap, q_descale, k_descale, v_descale, **kw):
    from vllm.v1.attention.ops.triton_unified_attention import KVQuantMode
    extras_off = all(kw.get(n) is None for n in ("alibi_slopes", "sinks", "output_scale", "qq_bias", "mm_prefix_range",
                                                 "rswa_prefix_lens", "rswa_window", "k_scale_cache", "v_scale_cache"))
    # max_seqlen_q > 1 forces Triton's 2D kernel (triton_unified_attention.py use_3d rule) — the one ours mirrors
    ok = (int(max_seqlen_q) > 1 and not causal and not softcap and q_descale is None and extras_off
          and kw.get("chunk_lookback", -1) == -1
          and not kw.get("use_alibi_sqrt") and kw.get("kv_quant_mode") == KVQuantMode.FP8_PER_TENSOR
          and window_size is not None and window_size[0] >= 0 and window_size[1] == 0
          and k_descale is not None and v_descale is not None and q.dtype == torch.bfloat16 and out.dtype == torch.bfloat16)
    if ok:
        rc = _lib().drafter_attn(out.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(), block_table.data_ptr(),
                                 seqused_k.data_ptr(), cu_seqlens_q.data_ptr(), k_descale.data_ptr(), v_descale.data_ptr(),
                                 float(softmax_scale), q.shape[0], seqused_k.shape[0], q.shape[1], k.shape[2], q.shape[2],
                                 k.shape[1], int(window_size[0]) + 1, block_table.stride(0), q.stride(0), q.stride(1),
                                 out.stride(0), out.stride(1), *k.stride()[:3], *v.stride()[:3],
                                 torch.cuda.current_stream().cuda_stream)
        if rc == 0:
            _S["ours"] += 1
            return
    _S["fallback"] += 1
    if _S["fallback"] <= 3:
        sys.stderr.write(f"[radiance.drafter_attn] Triton fallback (call #{_S['fallback']})\n")
    return _S["orig"](q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal,
                      window_size, block_table, softcap, q_descale, k_descale, v_descale, **kw)


def warmup():
    """Call after worker device/model initialization, before profiling and graph capture."""
    _lib()


def install():
    from vllm.v1.attention.backends import triton_attn
    if _S["orig"] is None:
        _S["orig"] = triton_attn.unified_attention
        triton_attn.unified_attention = _wrapped
        sys.stderr.write("[radiance.drafter_attn] TRITON_ATTN unified_attention -> drafter_attn (with fallback)\n")
