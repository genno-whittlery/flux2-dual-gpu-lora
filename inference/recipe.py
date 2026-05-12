#!/usr/bin/env python3
"""Dual-GPU FLUX.2 + LoRA inference — declarative YAML render jobs.

    <ai-toolkit-venv>/python inference/recipe.py job.yaml

job.yaml:
    base: F:\\models\\diffusion_models\\FLUX.2-dev   # optional (defaults to FLUX2_BASE / built-in)
    lora: C:\\ai-toolkit\\output\\suzurin-flux2-v2\\suzurin-flux2-v2.safetensors   # optional
    lora_rank: 16
    lora_alpha: 16
    out: C:\\tmp\\suzurin-v2-render
    width: 1024
    height: 1024
    steps: 20
    cfg: 4.0
    seed: 42
    strengths: [0, 0.6, 1.0]
    ext: png
    prompts:
      counter-greet: "suzurin standing behind a tea-house counter ..."
      pour-tea: "suzurin side-view pouring tea ..."

Each (prompt × strength) pair is written to ``<out>/<name>__s<NNN>.<ext>``.
"""
import argparse
import os
import sys
import traceback

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.engine import DEFAULT_BASE, LoraSpec, load_flux2, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Run a declarative FLUX.2 dual-GPU render recipe.")
    ap.add_argument("recipe", help="recipe YAML file")
    args = ap.parse_args()

    with open(args.recipe, "r", encoding="utf-8") as fh:
        r = yaml.safe_load(fh)
    if not r.get("prompts"):
        raise SystemExit("recipe must define a non-empty `prompts` map")

    lora = LoraSpec(r["lora"], r.get("lora_rank", 16), r.get("lora_alpha", 16)) if r.get("lora") else None
    sd, pipeline, net = load_flux2(r.get("base", DEFAULT_BASE), lora)
    render(sd, pipeline, r["prompts"], out=r["out"], network=net,
           strengths=tuple(r.get("strengths", [1.0])),
           width=r.get("width", 1024), height=r.get("height", 1024),
           steps=r.get("steps", 20), cfg=r.get("cfg", 4.0), seed=r.get("seed", 42), ext=r.get("ext", "png"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
