#!/usr/bin/env python3
"""Make DFlash2 probabilistic drafting (and block verification) lossless.

Three defects, all inert for the deployed greedy-draft + standard-verification serve (the walk
passes temperature 0 there, so it draws no noise and its stored scores are never read; the
rejection change is behind USE_BLOCK_VERIFICATION). Proof and numbers: check_rejection_math.py.

1. p1-temp (dflash2/speculator.py walk): the walk samples from softmax(scores / T)
   (gumbel_noised_argmax divides by the temperature) but stored the RAW scores, which the
   rejection test reads as q against temperature-scaled target logits. Store scores / T.
   Temperature-0 rows are verified greedily and never read q; they keep the raw scores.

2. p1-noise (dflash2/speculator.py walk): the walk keyed its Gumbel noise (seed, Q-1, token),
   the exact key the target's residual resample uses for position Q, so after a rejection the
   residual draw reused the draft's noise, conditioned on that draft having won. Biased even with
   fix 1. Salt the walk's seed with the step's anchor position: an independent stream, fresh per
   verify step.

3. p1-blocku (rejection_sampler_utils.py): block verification reads u at EVERY draft position,
   including those past the accepted prefix, and the next step re-verifies those positions with
   the same position-keyed u (tl_rand32(seed, pos)). The conditioned u biases the output, with
   greedy drafts too. Key block-mode u by the step's start position so it is fresh per step.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
SPEC = SP / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "dflash2" / "speculator.py"
RS = SP / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "rejection_sampler_utils.py"

apply(
    SPEC,
    "    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)\n"
    "    previous = 0\n",
    "    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)\n"
    "    # radiance (patch_dflash2_temperature.py, p1-noise): the draft's own Gumbel stream, fresh\n"
    "    # per verify step. Keyed (seed, Q-1) it was the noise the target's residual resample uses\n"
    "    # at position Q, which biases the residual after a rejection. The anchor (first mask\n"
    "    # position) is unique per step.\n"
    "    seed = seed ^ ((tl.load(sample_pos_ptr + row * num_steps) + 1) * 0x2545F4914F6CDD1)\n"
    "    previous = 0\n",
    "p1-noise",
    "dflash2 walk: independent per-step draft noise",
)

apply(
    SPEC,
    "        tl.store(\n"
    "            realized_scores_ptr + candidate_base + offsets,\n"
    "            scores,\n"
    "            mask=mask & valid,\n"
    "        )\n",
    "        # radiance (patch_dflash2_temperature.py, p1-temp): store the distribution actually\n"
    "        # sampled (gumbel_noised_argmax divides by the temperature); the rejection test reads\n"
    "        # these as q against temperature-scaled target logits. Temperature-0 rows are verified\n"
    "        # greedily and never read q, so they keep the raw scores.\n"
    "        tl.store(\n"
    "            realized_scores_ptr + candidate_base + offsets,\n"
    "            scores / tl.where(temperature != 0.0, temperature, 1.0),\n"
    "            mask=mask & valid,\n"
    "        )\n",
    "p1-temp",
    "dflash2 walk: temperature-scaled q",
)

apply(
    RS,
    "        u = tl_rand32(seed, pos, includes_zero=False)\n"
    "        if USE_BLOCK_VERIFICATION and not is_greedy:\n",
    "        u = tl_rand32(seed, pos, includes_zero=False)\n"
    "        if USE_BLOCK_VERIFICATION:\n"
    "            # radiance (patch_dflash2_temperature.py, p1-blocku): block verification reads u\n"
    "            # past the accepted prefix and the next step re-verifies those positions; a\n"
    "            # position-keyed u is then conditioned. Key it by this step's start position.\n"
    "            u = tl_rand32(\n"
    "                seed ^ ((tl.load(pos_ptr + start_idx) + 1) * 0x1D8E4E27C47D124F),\n"
    "                pos,\n"
    "                includes_zero=False,\n"
    "            )\n"
    "        if USE_BLOCK_VERIFICATION and not is_greedy:\n",
    "p1-blocku",
    "block verification: fresh u per verify step",
)
