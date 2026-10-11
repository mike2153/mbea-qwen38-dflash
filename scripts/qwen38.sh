#!/bin/bash
# Qwen3.8 + DFlash on one AMD Radeon AI PRO R9700 under WSL2. Runs inside WSL as root.
#
#   qwen38.sh setup            one-time: deps, ROCDXG, image, models, kernels (~40 GB download)
#   qwen38.sh start [--long]   start the server on :8080 (at least 180k context)
#   qwen38.sh stop | status | logs
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${QWEN38_DATA:-/root/qwen38-dflash}       # radiance checkout, libr4d build, compile caches
MODELS=${MODELS:-/root/models}
PORT=${PORT:-8080}
CONTAINER=${CONTAINER:-qwen38-dflash}

# Everything below is pinned; these are the exact versions the published numbers were measured on.
IMAGE=stilldeadcode/vllm-radiance@sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25  # tag 0.9.3
RADIANCE_REPO=https://codeberg.org/ggz14/radiance-vllm-mxfp4
RADIANCE_COMMIT=7d9a15a51d83733cc079fcf32ef4e3ec6b04ad7a
R4D_REPO=https://codeberg.org/StillDeadcode/libr4d.git
R4D_PIN=b9e42ab7202f53a3bc13d415f5d41481f9ca311b
MODEL_REPO=amd/Qwen3.8-27B-Quark-AWQ-MXFP4
MODEL_REV=5233554c5fa56afda40150556b95573c2d7d29c0
DRAFT_REPO=tcclaviger/Qwen3.8-27B-DFlash2-FP8
DRAFT_REV=ee0cb26a8279b7910cc28d82a8a3e15e4728d56f
ROCDXG_DEB=https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb

RUNTIME_IMAGE=mbea-qwen38-dflash:2026.10

SNAP=$MODELS/Qwen3.8-27B-MXFP4-official
SRC=$MODELS/$MODEL_REPO
DRAFTER=$MODELS/$DRAFT_REPO

step() { echo; echo "=== $* ==="; }
ok()   { echo "  ok: $*"; }
die()  { echo "ERROR: $1" >&2; shift; for l in "$@"; do echo "  $l" >&2; done; exit 1; }

# Device + library passthrough for the WSL GPU bridge (no /dev/kfd or /dev/dri under WSL).
gpu_args() {
  local rocdxg g gid
  rocdxg=$(readlink -f /opt/rocm/lib/librocdxg.so)
  GPU_ARGS=(--device /dev/dxg
    -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so:ro
    -v "$rocdxg":/usr/lib/librocdxg.so:ro
    -v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf:ro
    -e HSA_ENABLE_DXG_DETECTION=1)
  for g in render video; do
    gid=$(getent group "$g" | cut -d: -f3) && GPU_ARGS+=(--group-add "$gid")
  done
  return 0
}

hf_get() { # repo revision local-dir(under $MODELS)
  docker run --rm --network=host -e HF_TOKEN="${HF_TOKEN:-}" -v "$MODELS":/models \
    --entrypoint python3 "$IMAGE" -c '
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
' "$1" "$2" "/models/${3#"$MODELS"/}"
}

cmd_setup() {
  step "1/9  host"
  [ "$(id -u)" = 0 ] || die "run as root (the Windows wrapper does: wsl -u root)"
  [ -e /dev/dxg ] || die "/dev/dxg is missing" "this must run inside WSL2 with the AMD Windows driver installed"
  [ -e /usr/lib/wsl/lib/libdxcore.so ] || die "/usr/lib/wsl/lib/libdxcore.so is missing -- update WSL (wsl --update)"
  local free_gib
  free_gib=$(df -BG --output=avail /root | tail -1 | tr -dc '0-9')
  [ "$free_gib" -ge 45 ] || echo "  WARNING: ${free_gib} GiB free in WSL; a full setup needs ~45 GiB"
  ok "WSL2 GPU bridge present, ${free_gib} GiB free"

  step "2/9  packages (git, curl, python3, docker)"
  local need=()
  for c in git curl python3 docker; do command -v $c >/dev/null || need+=("$c"); done
  if [ ${#need[@]} -gt 0 ]; then
    apt-get update -q
    apt-get install -y -q git curl python3 docker.io
  fi
  docker info >/dev/null 2>&1 || systemctl enable --now docker 2>/dev/null || service docker start
  docker info >/dev/null 2>&1 || die "docker is installed but not running" "enable systemd in /etc/wsl.conf, then wsl --shutdown"
  ok "docker $(docker version --format '{{.Server.Version}}')"

  step "3/9  ROCDXG (ROCm <-> WSL GPU bridge)"
  if [ -e /opt/rocm/lib/librocdxg.so ]; then
    ok "already installed"
  else
    curl -fsSL -o /tmp/rocdxg-roct.deb "$ROCDXG_DEB"
    dpkg -i /tmp/rocdxg-roct.deb && rm -f /tmp/rocdxg-roct.deb
    ok "installed rocdxg-roct 1.2.2"
  fi

  step "4/9  container image (~14 GB)"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
  ok "vllm-radiance 0.9.3"

  step "5/9  GPU check inside the container"
  gpu_args
  docker run --rm "${GPU_ARGS[@]}" --entrypoint python3 "$IMAGE" -c '
import warnings; warnings.filterwarnings("ignore")  # "Cannot initialize amdsmi" is expected under WSL
import torch
assert torch.cuda.is_available(), "no GPU visible through ROCDXG"
p = torch.cuda.get_device_properties(0)
print(f"  ok: {p.name}, {p.total_memory / 2**30:.1f} GiB, {p.gcnArchName}")
assert "gfx1201" in p.gcnArchName, "this build targets gfx1201 (R9700) only"
'

  step "6/9  compile the pinned runtime image (CPU only)"
  mkdir -p "$DATA/cache"
  docker build --network host -t "$RUNTIME_IMAGE" "$REPO"
  ok "runtime patches and HIP kernels compiled into $RUNTIME_IMAGE"

  step "7/9  models (~21 GB)"
  local repo rev dir
  for spec in "$MODEL_REPO $MODEL_REV $SRC" "$DRAFT_REPO $DRAFT_REV $DRAFTER"; do
    read -r repo rev dir <<<"$spec"
    # snapshot_download resumes, so an interrupted download just runs again
    if [ -f "$dir/model.safetensors" ] && ! ls "$dir"/.cache/huggingface/download/*.incomplete >/dev/null 2>&1; then
      ok "$repo present"
    else
      hf_get "$repo" "$rev" "$dir"
    fi
  done
  python3 "$REPO/scripts/fix_checkpoint.py" "$SRC" "$SNAP"

  step "8/9  runtime package"
  docker image inspect "$RUNTIME_IMAGE" >/dev/null
  ok "pinned image present; no kernel compilation needed during start"

  step "9/9  first start (compiles Triton/inductor kernels into the cache, ~5-10 min)"
  # The compile run's autotuning buffers inflate vLLM's memory profile, so a cold start gets only
  # ~100k context instead of ~216k. Doing it once here means the user's first real start is warm.
  if [ -f "$DATA/cache/.warm" ]; then
    ok "kernel cache already warm"
  elif [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = true ]; then
    touch "$DATA/cache/.warm"; ok "server is running, so the cache is already warm"
  else
    QWEN38_WARMUP=1 cmd_start && cmd_stop && touch "$DATA/cache/.warm"
  fi

  echo
  echo "=== setup complete. Start with:  qwen38.sh start   (Windows: .\\qwen38.ps1 start) ==="
}

cmd_start() {
  [ "${1:-}" = --long ] && export VRAM_RESERVE_GIB=0.5 VRAM_CAP=0.95
  docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1 && [ -f "$SNAP/config.json" ] || die "not set up yet -- run: qwen38.sh setup"
  if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = true ]; then
    python3 "$REPO/scripts/lifecycle.py" ready "$DATA" "$CONTAINER" "$PORT"
    ok "$CONTAINER is already running"; return 0
  fi
  local control_env
  control_env=$(python3 "$REPO/scripts/lifecycle.py" prepare "$DATA" "$CONTAINER" "$PORT")
  ! ss -ltn "sport = :$PORT" | grep -q LISTEN || die "port $PORT is already in use"
  gpu_args
  docker run -d --name "$CONTAINER" --label io.mbea.qwen38.managed=1 --network host --ipc host --privileged \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
    --shm-size 64m "${GPU_ARGS[@]}" \
    -v "$MODELS":/models -v "$DATA/cache":/cache \
    --env-file "$REPO/config/dflash.env" --env-file "$control_env" \
    -e VRAM_RESERVE_GIB="${VRAM_RESERVE_GIB:-1.0}" -e VRAM_CAP="${VRAM_CAP:-0.95}" \
    "$RUNTIME_IMAGE" \
    /models/Qwen3.8-27B-MXFP4-official \
    --served-model-name Qwen3.8-DFlash --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 1 --language-model-only \
    --gpu-memory-utilization 0.90 --max-model-len -1 --max-num-seqs 1 --max-num-batched-tokens 2048 \
    --kv-cache-dtype fp8 --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype float16 \
    --mamba-cache-mode align --enable-prefix-caching \
    --attention-backend R4D \
    --speculative-config '{"method":"dflash","model":"/models/tcclaviger/Qwen3.8-27B-DFlash2-FP8","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
    --no-async-scheduling \
    --shutdown-timeout 30 --api-server-count 1 --middleware radiance_quiesce.ControlMiddleware \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40],"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true}}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20}' \
    --chat-template /patches/qwen-fixed-v22.3.jinja >/dev/null
  python3 "$REPO/scripts/lifecycle.py" record "$DATA" "$CONTAINER" "$PORT"
  echo "Loading (2-3 min warm; the first start compiles kernels and can take 10+ min)..."
  local t0=$SECONDS
  until curl -sf "localhost:$PORT/health" >/dev/null; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != true ]; then
      docker logs --tail 40 "$CONTAINER" >&2
      python3 "$REPO/scripts/lifecycle.py" stop "$DATA" "$CONTAINER" "$PORT"
      die "the container exited during startup (log tail above)"
    fi
    [ $((SECONDS - t0)) -lt 2400 ] || die "not ready after 40 min -- see: qwen38.sh logs"
    sleep 5
  done
  local readiness=ready
  [ "${QWEN38_WARMUP:-0}" = 1 ] && readiness=warmup-ready
  python3 "$REPO/scripts/lifecycle.py" "$readiness" "$DATA" "$CONTAINER" "$PORT"
  echo "Ready in $((SECONDS - t0)) s."
  cmd_status
}

cmd_status() {
  local st
  st=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "not created")
  echo "container: $CONTAINER ($st)"
  if curl -sf "localhost:$PORT/health" >/dev/null; then
    curl -sf "localhost:$PORT/v1/models" | python3 -c '
import json, sys
m = json.load(sys.stdin)["data"][0]
print("endpoint:  http://localhost:%s/v1  model=%s  context=%s tokens"
      % (sys.argv[1], m["id"], format(m.get("max_model_len") or 0, ",")))' "$PORT"
  else
    echo "endpoint:  not ready"
  fi
}

cmd_stop() {
  python3 "$REPO/scripts/lifecycle.py" stop "$DATA" "$CONTAINER" "$PORT"
  ok "clean shutdown acknowledged"
}

cmd_validate() {
  python3 "$REPO/bench/validate.py" --base "http://127.0.0.1:$PORT" \
    --out "$DATA/reports/api-$(date +%Y%m%d-%H%M%S).json"
}

case "${1:-}" in
  setup)  cmd_setup ;;
  start)  shift; cmd_start "$@" ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  validate) cmd_validate ;;
  logs)   docker logs -f --tail 200 "$CONTAINER" ;;
  *) sed -n '2,7p' "$0"; exit 2 ;;
esac
