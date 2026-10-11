# mbea-qwen38-dflash

Qwen3.8-27B with DFlash2 on one AMD Radeon AI PRO R9700 (32 GB), under Windows 11 and WSL2. The server provides an OpenAI-compatible API with streaming, reasoning and tool calls. It keeps the proven probabilistic drafting and block-verification settings.

This release packages the pinned runtime into a Docker image, adds the validated startup memory repair, and requires an acknowledged GPU drain before shutdown. The experimental native engine is documented [separately](docs/NATIVE_ENGINE.md); its speed figures are not measurements of this HTTP container.

## What changed

| Before | This release |
|---|---|
| Patch and compile the runtime at each start | Build the runtime image during installation; reuse it at startup |
| Startup head check creates a 2.37 GiB temporary | Equivalent zero-weight check uses about 10.5 KiB of scratch |
| Memory budget omits runtime overhead | Budget includes 2.2 GiB overhead and an explicit Windows reserve |
| Context silently shrinks when memory is scarce | Normal startup requires at least 180,000 tokens |
| Timed Docker stop can force-kill a worker | Authenticated scheduler pause, API drain and device acknowledgement precede SIGTERM; no kill escalation |

Model weights, quantization, chat template, sampling defaults and the seven-token DFlash2 configuration are unchanged. Rejected kernel combinations, wider drafter history and unqualified tree verification are not enabled.

Release qualification results will be recorded in [the release report](docs/RELEASE-2026-10.md). Earlier measurements and their methods remain in [BENCHMARKS.md](docs/BENCHMARKS.md).

## Requirements

- AMD Radeon AI PRO R9700, gfx1201, 32 GB. This build targets that card.
- Windows 11 with WSL2 and an AMD PRO driver supporting WSL. Tested with PRO 26.7.1.
- At least 64 GB RAM recommended and approximately 65 GB free Linux disk space for models, images, build layers and caches.
- Approximately 35 GB of initial downloads. Model downloads resume if interrupted.

The installer uses Ubuntu, Docker and ROCDXG. Model weights are downloaded from their original repositories; they are not included in this repository or the runtime image.

## Install and start

Run in PowerShell:

```powershell
git clone https://github.com/mike2153/mbea-qwen38-dflash
cd mbea-qwen38-dflash
.\qwen38.ps1 install
.\qwen38.ps1 start
```

If Ubuntu needs installing, create its Linux user when prompted and run `install` again. If PowerShell blocks scripts, use `powershell -ExecutionPolicy Bypass -File .\qwen38.ps1 install`.

The endpoint is [localhost:8080/v1](http://localhost:8080/v1), model `Qwen3.8-DFlash`.

| Command | Purpose |
|---|---|
| `.\qwen38.ps1 install` | Download the pinned dependencies/models, build the image and warm compile caches |
| `.\qwen38.ps1 start` | Start with a 1.0 GiB budgeted Windows reserve and require at least 180k context |
| `.\qwen38.ps1 start -Long` | Budget a 0.5 GiB Windows reserve; actual context depends on available memory |
| `.\qwen38.ps1 status` | Show container state and the actual context capacity |
| `.\qwen38.ps1 validate` | Check health, context, streaming, reasoning, arithmetic, tools and request validation |
| `.\qwen38.ps1 bench` | Run the included short-prompt performance benchmark |
| `.\qwen38.ps1 logs` | Follow server logs |
| `.\qwen38.ps1 stop` | Drain GPU work and remove the container only after a clean exit |

In WSL, run `bash scripts/qwen38.sh setup`, `start`, `status`, `validate` or `stop` as root. Building the Docker image itself needs no GPU access:

```bash
docker build --network host -t mbea-qwen38-dflash:2026.10 .
```

The managed `qwen-server` installation is separate. These scripts do not replace its container. A container from an older release without a recorded lifecycle identity is not automatically adopted or removed.

## API

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-DFlash","messages":[{"role":"user","content":"Write a Rust function that reverses a linked list."}]}'
```

- Reasoning is enabled by default. This pinned API returns it in `reasoning`; it accepts the older `reasoning_content` message field too.
- Set `"chat_template_kwargs":{"enable_thinking":false}` to disable thinking per request.
- Tool calls use the `qwen3_coder` parser and automatic tool choice.
- Defaults are temperature 0.7, top-p 0.95 and top-k 20; requests can override them.
- The profile serves one sequence at a time. It is tuned for a single coding agent.

## Configuration

Set environment variables in WSL before running the shell script:

| Variable | Default | Meaning |
|---|---|---|
| `VRAM_RESERVE_GIB` | `1.0` | Windows reserve after the separate runtime-overhead allowance; minimum 0.5 |
| `VRAM_CAP` | `0.95` | Maximum fraction available to vLLM's budget |
| `PORT` | `8080` | HTTP port |
| `CONTAINER` | `qwen38-dflash` | Managed name; isolated tests may use `exp-*` |
| `MODELS` | `/root/models` | Model checkpoints |
| `QWEN38_DATA` | `/root/qwen38-dflash` | Compile caches and private lifecycle records |
| `HF_TOKEN` | unset | Optional Hugging Face download token |

The budget refuses to trade away its configured reserve to fit the context floor. Readiness independently checks actual context. Close other GPU applications if there is insufficient memory. Compilation warm-up may temporarily fit less context; the installer sends no inference request during that warm-up.

The inference settings are in [dflash.env](config/dflash.env), the runtime build is in [Dockerfile](Dockerfile), and container arguments are in [qwen38.sh](scripts/qwen38.sh). All upstream image, runtime, kernel and model revisions are pinned.

## Validation and measurements

The release gate compares the candidate with a fresh reference using 24 coding seeds, 46 external Rust tests per answer, greedy and sampled long-context prompts, API checks, a one-hour workload and clean lifecycle tests. Finite tests can establish that no regression was observed on that suite; they cannot prove unchanged intelligence on every possible task.

The included CPU suite runs without a GPU:

```bash
python3 -m unittest discover -s tests -v
```

The included [API validator](bench/validate.py) writes aggregate checks without generated content. The full private coding-agent evaluation corpus and raw responses are not published.

Historical short coding prompts measured about 125–134 tok/s, while harder long-context agent work ran around 70–95 tok/s. These are workload-dependent measurements, not a guaranteed floor. See the [benchmark history](docs/BENCHMARKS.md) and [release report](docs/RELEASE-2026-10.md) for the exact scope, context lengths, quality scores and startup timings. No 150 tok/s claim is made.

## Troubleshooting

| Symptom | Action |
|---|---|
| GPU bridge missing | Check the AMD WSL driver and update WSL; this installer does not reset or terminate WSL |
| Insufficient VRAM | Close GPU-heavy applications and retry with the normal profile |
| Context below 180k after compilation | The server stops cleanly; retry with the warmed cache |
| Stop reports an unconfirmed drain | Keep the container and diagnostic files. No forced removal or automatic kill is attempted |
| Container exits during startup | Read the retained logs before retrying |
| Port already in use | Select a free port with `PORT=...` |

A clean stop preserves models and caches. Lifecycle credentials and acknowledgements are stored privately under the data directory and should not be published.

## Credits and licenses

This project packages [radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4), [vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance), [libr4d](https://codeberg.org/StillDeadcode/libr4d), [vLLM](https://github.com/vllm-project/vllm) and [ROCDXG](https://github.com/ROCm/librocdxg). Thanks to their authors for the runtime and RDNA4 kernels.

The models are [AMD's Qwen3.8-27B MXFP4 checkpoint](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) and [tcclaviger's DFlash2 FP8 drafter](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8), based on [Qwen](https://huggingface.co/Qwen).

Third-party dependencies are fetched from their sources under their own terms. This repository's scripts and patches are MIT licensed; see [LICENSE](LICENSE). Experimental, community-maintained, tested on one R9700; not affiliated with the upstream authors.
