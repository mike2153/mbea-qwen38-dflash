# Changelog

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
