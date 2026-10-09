#!/bin/bash
# Runs INSIDE the container. Applies the radiance patch chain to the image's vLLM, compiles the
# W4A8 GEMM kernel for gfx1201, swaps in the pinned libr4d, then hands off to vLLM through
# available_memory.py (which sizes --gpu-memory-utilization from free VRAM at startup).
# /patches = radiance checkout (pinned + patches/radiance-wsl-r9700.patch), /r4d = libr4d build.
set -e
SP=/opt/vllm/lib/python3.12/site-packages
cd /patches
python3 patch_quark_mxfp4.py
python3 patch_nvfp4_mxfp4.py
python3 patch_wsl_dxg.py           # AMD-SMI is limited under WSL; detect ROCm through HIP
python3 patch_tp3_pad.py
python3 patch_ar_maxbytes.py
python3 patch_topk_triton_rows.py
python3 patch_dflash_calib.py
python3 patch_dflash_mxfp4_kv.py
python3 patch_rmsquant_fusion.py
python3 patch_verify_head.py
python3 patch_kv_group_size.py
python3 patch_topk_composite.py
python3 patch_gdn_shared_build.py
python3 patch_dflash_selector_topk.py
python3 patch_gdn_merge_inproj.py
python3 patch_dynwidth.py
python3 patch_async_dynwidth.py
python3 patch_step_trace.py
python3 patch_ar_geometry.py
python3 patch_ar_qbits.py
python3 patch_ar_3rank.py
python3 patch_gdn_glue.py
python3 patch_qwen3_thinkoff.py \
  || echo "[radiance] WARNING: thinkoff patch did not apply; thinking-off requests will return empty content"
# Lossless DFlash2 probabilistic drafting + block verification: fixes three upstream bugs (see the
# file). Bind-mounted from this repo's patches/ by qwen38.sh start.
python3 patch_dflash2_temperature.py
cp mxfp4-configs/*.json "$SP"/aiter/ops/triton/configs/gemm/
cp radiance_preamble.py /opt/radiance_preamble.py
cp radiance_nvfp4.py radiance_mxfp4.py radiance_gdn.py radiance_gdn_lazy.py radiance_rmsquant.py \
   radiance_drafthead.py radiance_verifyhead.py radiance_gdnmerge.py radiance_aroverlap.py \
   radiance_topk.py radiance_arnq.py radiance_tp3pad.py "$SP"/
hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=gfx1201 \
  $(python3 -m pybind11 --includes) radiance_mxfp4_fp8.hip -o "$SP"/radiance_mxfp4_fp8.so
# The image's own libr4d predates the gated-delta-net overflow fix and NaNs this model.
[ -f /r4d/r4d.so ] || { echo "[radiance] ERROR: /r4d/r4d.so missing -- run setup first" >&2; exit 1; }
cp /r4d/r4d.so "$SP"/r4d.so
echo "[radiance] using pinned libr4d from /r4d"
# Leave /patches before exec: a stale .so in the working directory would shadow site-packages.
cd /
exec python3 /patches/available_memory.py "$@"
