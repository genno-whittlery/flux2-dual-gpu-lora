#!/usr/bin/env python3
"""Dual-GPU FLUX.2 + LoRA inference — command line.

    <ai-toolkit-venv>/python inference/cli.py \\
        --lora out/suzurin-flux2-v2/suzurin-flux2-v2.safetensors \\
        --prompts prompts.json --out renders/ --strengths 0,0.6,1.0

`--prompts` is a JSON object ``{name: prompt, ...}``. Each (prompt × strength) pair is
written to ``<out>/<name>__s<NNN>.<ext>``. Run inside ai-toolkit's venv; see engine.py
for the env-var contract (FLUX2_DUAL_GPU / FLUX2_TE_DEVICE / FLUX2_BASE / AITK_DIR).
"""
import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.engine import DEFAULT_BASE, LoraSpec, load_flux2, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Dual-GPU FLUX.2 + LoRA inference (ai-toolkit modules, no run.py).")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="FLUX.2-dev path (raw BFL single-file checkpoint dir or .safetensors)")
    ap.add_argument("--lora", help="trained ai-toolkit LoRA .safetensors (omit for base-model renders)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--prompts", required=True, help="JSON file: {name: prompt, ...}")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42, help="single seed (default 42; ignored if --seeds is given)")
    ap.add_argument("--seeds", help="comma-separated seeds, e.g. 42,43,44 (overrides --seed; adds __seedN to output filenames)")
    ap.add_argument("--strengths", default="1.0", help="comma-separated LoRA multipliers, e.g. 0,0.6,1.0")
    ap.add_argument("--ext", default="png")
    args = ap.parse_args()

    with open(args.prompts, "r", encoding="utf-8") as fh:
        prompts = json.load(fh)
    strengths = tuple(float(x) for x in args.strengths.split(",") if x.strip() != "")
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "") if args.seeds else None
    lora = LoraSpec(args.lora, args.lora_rank, args.lora_alpha) if args.lora else None

    sd, pipeline, net = load_flux2(args.base, lora)
    render(sd, pipeline, prompts, out=args.out, network=net, strengths=strengths, seeds=seeds,
           width=args.width, height=args.height, steps=args.steps, cfg=args.cfg,
           seed=args.seed, ext=args.ext)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
