# Dual-GPU FLUX.2 LoRA training in OneTrainer

`Flux2DualGpu.py` — drop into `<OneTrainer>/modules/util/`. Splits the diffusers `Flux2Transformer2DModel` across two CUDA devices at the `single_transformer_blocks` midpoint. Enables FLUX.2 LoRA training on pairs of 24+ GB consumer GPUs (2× RTX 3090 / 4090 / 5090) — on a single 24 GB card the transformer can't fit alongside activations even with WDDM paging.

`EncodeMistralText.py` — patched copy of `mgds/pipelineModules/EncodeMistralText.py` (the third-party data-loader OneTrainer vendors): moves `tokens` + `attention_mask` to `text_encoder.device` before the forward call, so CPU-hosted Mistral receives CPU-side inputs instead of cuda:0 ones. Drop into `<OneTrainer-venv>/src/mgds/src/mgds/pipelineModules/`, or consume the upstream branch [`genno-whittlery/mgds:te-device-fix`](https://github.com/genno-whittlery/mgds/tree/te-device-fix).

## Integration

Two edits in `modules/model/Flux2Model.py`: force the text encoder onto CPU; route `transformer_to(device)` to the distribute call. The exact byte-surgery (and the per-block forward-pre-hook registration — important for FLUX.2's modulation/`temb` loop pattern) lives in the raw patcher scripts: [`patches/onetrainer/`](../../patches/onetrainer/).

## Env vars

| Variable | Default | Effect |
|---|---|---|
| `FLUX2_DUAL_GPU` | `false` | Set `true` to enable the dual-GPU path |
| `FLUX2_DUAL_GPU_SPLIT_AT` | `num_single // 2` (= 24) | Override the split index |

## Status — ✅ validated end-to-end (2× RTX 5090, 2026-05-11)

Full 32-step epoch on the sumi v8 dataset in **both** of OneTrainer's 8-bit-compute modes: `INT_W8A8` 0.95 s/it (int8 tensor cores), `FLOAT_8` (= W8A8 fp8 compute, not weight-only) 1.75 s/it (fp8 tensor cores); LoRA checkpoints saved (402 MB / 384 MB). OneTrainer doesn't ship a "weight-only fp8" mode for the dit — its `LinearW8A8` class always quantizes activations too — so this isn't apples-to-apples with the weight-only ports (ai-toolkit / musubi / DiffSynth, all ~2.2–2.9 s/it bf16-compute). What it does show: the **dual-GPU split is independent of compute-precision choice** — distribution shape is 20.7 GB cuda:0 / 12.6 GB cuda:1 across all four trainers and all three precision classes.

Upstream PR [Nerogar/OneTrainer#1450](https://github.com/Nerogar/OneTrainer/pull/1450) was closed on a stale-issue-reference technicality (technical content not reviewed). The fork branch [`genno-whittlery/OneTrainer:dual-gpu-flux2`](https://github.com/genno-whittlery/OneTrainer/tree/dual-gpu-flux2) is the canonical place to consume it — apply locally to a `Nerogar/OneTrainer` checkout, or use the fork directly.

## Setup notes

- **Base model**: OneTrainer needs the diffusers-folder format (`black-forest-labs/FLUX.2-dev` from HF), not the single-file `.safetensors` — ~108 GB (60 GB transformer + 45 GB Mistral + scheduler/VAE/configs).
- **Per-trainer venv recommended** — `C:\OneTrainer\.venv` with Python 3.12 + OneTrainer's pinned `torch 2.9.1+cu128` (Blackwell-compatible despite cu128 — PTX forward-compat carries fp8 + bf16 fine).
- **Latent cache is one-time** — ~10 min for 32 images at 512² on CPU VAE; `workspace_dir`-keyed, reused on subsequent runs unless the dataset/config changes.
