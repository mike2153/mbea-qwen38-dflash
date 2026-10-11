# Native-engine research result

The packaged HTTP server uses the qualified vLLM-radiance runtime. The experimental native C++/HIP engine is a separate development result and is not enabled by this release.

| Measurement | Reference | Native | Scope |
|---|---:|---:|---|
| Mean decode | 83.341 tok/s | 94.631 tok/s | One matched 103,843-token coding-agent prompt, 759 timed tokens |
| Speed met for 90% of decode time | 53 tok/s | 69 tok/s | Same prompt, duration-weighted one-second intervals; only 8–9 seconds per arm |
| Prepared target-layer weight loading | 69.619 s | 27.034 s | Includes cache checksum validation; excludes compilation, other weights, graph capture and HTTP readiness |

All 764 greedy output tokens matched through EOS, with 200,000-token native capacity. Decode includes thinking, answer and tool tokens; it excludes loading, prefill and the same five bootstrap tokens. The matched comparison imported a saved prefix. A later test computed the full native prefix and matched all 764 response tokens at approximately 100 tok/s; that later number was unpaired and does not establish an additional speed improvement.

The native engine now has exact target prefill, drafter context caches, bootstrap proposals and complete response tests. Those tests were joined through saved native checkpoints. Continuous cold startup, general request handling, HTTP serving, broad greedy coverage and sampled-distribution qualification remain unfinished. Its measured gains therefore **do not describe this repository's shipped HTTP container**.

The 150 tok/s research target was not reached. Wider drafter history changed a response and was rejected. Tree verification did not complete GPU qualification. Neither is part of the release. Private prompts, model states, caches and generated responses are not published.
