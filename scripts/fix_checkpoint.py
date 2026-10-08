#!/usr/bin/env python3
"""Make AMD's Qwen3.8-27B-Quark-AWQ-MXFP4 release loadable without rewriting its weights.

AMD's quantization_config.exclude names the bf16 MTP layers as *tensor* names
("mtp.fc.weight") among *module* names, so Quark's module match never fires, the MXFP4 scheme
lands on a bf16 head and vLLM asserts at load. Stripping ".weight" fixes the match.
language_model_only=true also skips building and profiling the unused vision tower (~1 GiB VRAM).

Builds DST as a directory of relative symlinks to SRC plus the fixed config.json, so the
19 GB safetensors file is shared, not copied.   usage: fix_checkpoint.py SRC DST
"""
import json
import os
import sys


def fix(cfg):
    cfg = json.loads(json.dumps(cfg))
    q = cfg["quantization_config"]
    q["exclude"] = [e[:-len(".weight")] if e.startswith("mtp.") and e.endswith(".weight") else e
                    for e in q["exclude"]]
    cfg["language_model_only"] = True
    return cfg


def main(src, dst):
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.startswith(".") or name == "config.json":
            continue
        link = os.path.join(dst, name)
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(os.path.join(src, name), dst), link)
    with open(os.path.join(src, "config.json")) as f:
        cfg = fix(json.load(f))
    out = os.path.join(dst, "config.json")
    if os.path.exists(out):
        with open(out) as f:
            if json.load(f) == cfg:  # leave an already-correct file byte-for-byte untouched
                print(f"  ok: loadable checkpoint at {dst}")
                return
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"  ok: loadable checkpoint at {dst}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        out = fix({"quantization_config": {"exclude": ["mtp.fc.weight", "model.visual.pos_embed",
                                                       "lm_head", "mtp.norm.weight"]}})
        assert out["quantization_config"]["exclude"] == ["mtp.fc", "model.visual.pos_embed",
                                                         "lm_head", "mtp.norm"], out
        assert out["language_model_only"] is True
        print("ok")
    elif len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        sys.exit(__doc__)
