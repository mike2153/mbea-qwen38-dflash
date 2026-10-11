# Release qualification — 2026-10-11

Status: running. This report is not a passed release gate yet.

The candidate keeps the current reference model, arithmetic and sampling configuration. The release changes packaging, startup allocation, memory budgeting and shutdown. The native prototype remains separate.

Completed preparation: image built without GPU devices; 25 CPU tests pass; the pinned source contract for scheduler/device drain is checked during the build. An initial API-helper attempt expected the obsolete reasoning field and was corrected after checking the installed schema. Its failure and clean shutdown were retained; no coding-quality requests ran in that attempt.

The fresh gate covers 18 greedy long-context prompts, 54 paired sampled requests (18 prompts × three seeds), and 24 coding seeds × 46 external Rust tests per arm. The candidate must complete at least one hour of workload plus API and cancellation/restart checks, with at least 180k context and 0.5 GiB sampled physical Windows reserve. Raw prompts, generations, lifecycle credentials and host traces remain private.
