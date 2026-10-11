# Changelog

## 2026-10-11 — GDN and drafter attention kernels (image 2026.10.1)

- Enable two HIP kernels, compiled into the image, that replace the Gated DeltaNet verify update and the DFlash2 drafter attention with byte-exact equivalents. Sources are in `kernels/`; their comments cite development notes that are not part of this repository.
- Measured on one R9700 against the previous runtime, same requests in the same session, prefill excluded: 87.4 → 91.7 tok/s over 18 sampled requests at 34k–180k context (+4.9%); thinking 89.0 → 93.1, answer and tool tokens 84.9 → 89.5; 90.5 → 94.5 at 100k+ (+4.4%).
- All 18 greedy, 18 sampled and 4 coding responses were identical to the previous runtime. This was a shortened check: no one-hour workload and 4 of the usual 24 coding seeds. An earlier 78-response comparison of the same kernels had one differing sampled coding response; it did not reproduce in replays and its cause is unknown.
- Name the warm-cache marker after the image tag, so `install` re-warms the compile cache after an update. The first start with a new runtime recompiles the target graph and fits roughly half the usual context until the next start.

## 2026-10-11 — qualified runtime package

- Add a reproducible Dockerfile using the existing pinned vLLM-radiance image, radiance revision and libr4d revision. Compile and apply patches during image construction instead of at each start.
- Replace the verify-head startup zero check with the validated equivalent predicate, reducing temporary GPU allocation from 2.37 GiB to about 10.5 KiB.
- Budget runtime overhead separately from Windows reserve. Default to 1.0 GiB reserved, retain a 0.5 GiB minimum, and require at least 180k context at normal readiness.
- Set native shutdown grace to 30 seconds. Require a matching container/boot identity, closed admission, scheduler pause, API drain and device synchronization before SIGTERM. Preserve a stalled or abnormal container rather than escalating to SIGKILL.
- Make the thinking-off patch mandatory and add an API validator for streaming, reasoning, tool calls, arithmetic, context and invalid requests.
- Add CPU regression coverage for memory policy, patch idempotence, admission races, control authentication and shutdown failures.
- Keep the established model revisions, quantization, seven-token DFlash2 speculation, probabilistic drafting, block verification and sampling defaults.
- Document the native prototype separately. Its one-prompt speed and weight-loading results are not shipped HTTP performance claims. Failed kernel combinations, wider-history drafting and unqualified tree work remain excluded.

See [release qualification](docs/RELEASE-2026-10.md) for the measured scope and limits.

## 2026-10-09 — probabilistic drafting

- Fix temperature-scaled draft probabilities, independent draft noise and fresh block-verification random numbers.
- Enable lossless probabilistic drafting and block verification.
- Record paired long-context measurements and the 24-seed coding comparison in [BENCHMARKS.md](docs/BENCHMARKS.md).

## 2026-10-08 — initial package

- Add the Windows/WSL installer, pinned model and runtime setup, benchmark client and WSL/R9700 patches.
