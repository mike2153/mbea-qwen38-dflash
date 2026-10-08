# Benchmarks

**Machine:** AMD Radeon AI PRO R9700 32 GB (gfx1201) · Ryzen 9 9950X · 96 GB DDR5 · Windows 11 IoT Enterprise LTSC 2024 · WSL2 kernel 6.18 · Ubuntu 24.04 · Docker 29.1 · AMD Software PRO 26.7.1 · Resizable BAR off.
**Software:** `stilldeadcode/vllm-radiance:0.9.3` + radiance `7d9a15a` + this repo's patch · libr4d `b9e42ab-rx9` · AMD Quark MXFP4 checkpoint · DFlash2-FP8 drafter, 7 speculative tokens.
**Dates:** 2026-10-07 to 2026-10-08. Each configuration ran in a fresh container.

## Method

| Metric | How it is measured |
|---|---|
| Decode tok/s | completion tokens / (total time − time to first token), streamed. Median of 3 runs on a 1,912-token Rust coding spec ([`bench/prompt.txt`](../bench/prompt.txt)), thinking on, ~7,000 output tokens per run |
| Greedy / sampled | temperature 0 / the server default (0.7, top-p 0.95, top-k 20) |
| Prefill tok/s | prompt tokens / time to first token. Every request has a unique nonce, so prefix caching cannot help |
| Draft acceptance | accepted / proposed draft tokens, from `/metrics` deltas over the runs |
| Context | KV-cache pool reported by vLLM at startup. "Verified" is the longest prompt from which all 8 planted facts were retrieved correctly |
| Coding quality | the generated `lib.rs` is compiled with `rustc --edition=2024 -D warnings` and run against 46 hidden tests; 6 runs per configuration |

Run-to-run noise is about ±4 % on decode and ±2 perfect runs out of 6. Treat smaller differences as noise.

## Shipped profile

`RADIANCE_DRAFT_RERANK=80` + `RADIANCE_VERIFY_HEAD=1`, 1.5 GiB of VRAM reserved for Windows. Two independent starts:

| Start | Decode greedy (runs) | Decode sampled | Prefill | TTFT | Acceptance | Tokens / verify | KV pool | 46/46 runs |
|---|---|---|---|---|---|---|---|---|
| A | **127.0** (124.0 · 127.0 · 132.8) | 123.9 | 2,966 tok/s | 0.66 s | 59.0 % | 5.13 | 215,991 | 4 / 6 |
| B | **133.7** (132.5 · 133.7 · 135.6) | 126.0 | n/a | 0.67 s | 63.0 % | 5.41 | 215,991 | 6 / 6 |

A third start, from a clean install of this repository (`.\qwen38.ps1 bench`, 4,096-token cap): **125.3** greedy (119.6 · 125.3 · 125.9), 116.7 sampled, 2,744 tok/s prefill, 61.9 % acceptance, KV pool 215,991 tokens.

Acceptance by draft position (start B): 91 % · 80 % · 70 % · 61 % · 53 % · 46 % · 39 %.

### Long context

Retrieval prompts. Prefill is the time to first token; decode is measured on the answer.

| Prompt tokens | Prefill | Prefill tok/s | Decode tok/s | Facts retrieved |
|---:|---:|---:|---:|---:|
| 32,444 | 10.9 s | 2,977 | 164.8 | 8 / 8 |
| 97,734 | 41.1 s | 2,378 | 136.0 | 8 / 8 |
| 163,506 | 83.0 s | 1,970 | 144.6 | 8 / 8 |
| 257,768 * | 163.5 s | 1,577 | 114.5 | 8 / 8 |

\* With `-Long` (1.0 GiB reserve, 0.95 cap): KV pool 260,879 tokens.

## Every configuration tried

| Configuration | Decode greedy · sampled | vs baseline | Acceptance | KV pool · max verified | Task time (s) | 46/46 runs |
|---|---|---|---|---|---|---|
| Baseline (no rerank / verify head) | 122.6 · 117.7 | – | 61.2 % | 205,920 · 203,147 | 60.2 | 5 / 6 |
| Baseline repeat (drift control) | 118.5 · 112.8 | −3.4 % | 59.4 % | 216,480 · – | 65.6 | 3 / 6 |
| **Rerank 80 + verify head (shipped)** | **127.0 · 123.9** | +3.6 % | 59.0 % | 215,991 · 163,506 | 58.9 | 4 / 6 |
| **Rerank 80 + verify head, repeat** | **133.7 · 126.0** | +9.1 % | 63.0 % | 215,991 · – | 53.2 | 6 / 6 |
| + 1.0 GiB reserve / 0.95 cap (`-Long`) | 123.1 · 118.7 | +0.4 % | 62.8 % | 260,879 · 257,768 | 57.9 | 5 / 6 |
| Reserve change only | 119.7 · 118.5 | −2.4 % | 59.9 % | 260,480 · 163,506 | 71.0 | 4 / 6 |
| + `GPU_MAX_HW_QUEUES=2` | 129.9 · 127.5 | +6.0 % | 61.4 % | 216,480 · – | 57.5 | 4 / 6 |
| + probabilistic drafting | 131.0 · 128.9 | +6.9 % | 60.2 % | 216,480 · – | 54.6 | 5 / 6 |

The HW-queue and probabilistic-drafting variants fall inside the spread of the shipped profile, so they were not adopted.

## Reproduce

```powershell
.\qwen38.ps1 start
.\qwen38.ps1 bench        # coding task greedy + sampled, plus a 32k-token prompt
```

`bench` prints a Markdown table in the same format. It uses a 4,096-token output cap (`--max-tokens`) to keep the run to a few minutes, so its decode figures can differ slightly from the ~7,000-token runs above.
