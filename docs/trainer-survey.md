# FLUX.2 LoRA trainer landscape — porting survey

**Status as of 2026-05-10.**

Survey of FLUX.2-dev LoRA training tools to identify the next port target for the dual-GPU model-parallel patch landed here for ai-toolkit ([PR #829](https://github.com/ostris/ai-toolkit/pull/829)).

## The matrix

| Trainer | Stars / activity | FLUX.2-dev LoRA | Dual-GPU **model-parallel**? | Port complexity | FLUX.2 OOM-class issues |
|---|---|---|---|---|---|
| **ai-toolkit** (ostris) | ~9k, very active | ✅ ref impl | ✅ this repo's patch | done | many; patch addresses |
| **musubi-tuner** (kohya-ss) | 1.8k, very active | ✅ since 2026-01 | ❌ Accelerate DP + block-swap only | medium | #923, #938; recurring 24 GB OOM |
| **OneTrainer** (Nerogar) | 3.0k, active | ✅ Dev + Klein | ❌ — issue #588 open since 2024-11 | medium-hard (custom abstraction layer) | #588, #1265 |
| **SimpleTuner** (bghira) | 2.8k, very active | ✅ FLUX.2 32B | ❌ FSDP2 / DeepSpeed only | hard — FSDP2 conflicts with the split shape | #997, #951, #303, #624 |
| **diffusers** `train_dreambooth_lora_flux2.py` | 28k repo, official example | ✅ ref | ❌ Accelerate + FSDP2; `--offload` for TE/VAE | easiest mechanical port | recurring #9732-class threads |
| **diffusion-pipe** (tdrussell) | 1.9k, active | ✅ since 2026-01 | ✅ **already pipeline-parallel** (DeepSpeed PP) | different shape — already does it | #173 |
| **DiffSynth-Studio** (modelscope) | 12.4k, very active | ✅ recipe shipped | ❓ sequence-parallel for Wan 2.2; FLUX.2 split not documented | unknown — non-English-primary | not surveyed |
| **finetrainers** (HF) | smaller, video-focused | ❌ FLUX.1 only | n/a | not worth porting now | n/a |
| **kohya-ss/sd-scripts** | 7k, active | ❌ FLUX.1 only | n/a | skip — FLUX.2 work is in `musubi-tuner` | n/a |

## Recommended next port: **musubi-tuner**

Highest popularity × user-pain overlap × portability. kohya's audience overlaps heavily with the consumer-GPU FLUX.2 trainees this patch unblocks. FLUX.2 support is fresh (Jan 2026), so OOM workarounds are still being negotiated in real time (#923, #938). fp8 + text-encoder caching primitives are already in place, so the port reduces to the same three pieces it took in ai-toolkit:

1. TE-on-CPU (musubi already has `flux_2_cache_text_encoder_outputs`)
2. Transformer split at the single_blocks midpoint
3. LoRA per-layer device routing

kohya merges aggressively, which lowers the upstream-acceptance risk.

**Secondary target: diffusers `train_dreambooth_lora_flux2.py`** — easiest mechanical port (PEFT-wrapped LoRA, `--offload` already wired). Even if not upstreamed (it's an HF example, not a product), publishing a `train_dreambooth_lora_flux2_mp.py` companion gives any diffusers user a copyable reference.

## Skip / deprioritize

- **SimpleTuner** — FSDP2 path conflicts with a static cuda:0/cuda:1 split; quanto blocks multi-GPU (#717); group-offload is module-generic and fights an explicit split.
- **diffusion-pipe** — already does pipeline-parallel via DeepSpeed PP. Our patch's value inverts there — they'd want a *non*-PP fallback if anything, not a re-implementation of what they already do.
- **finetrainers** — no FLUX.2 yet.
- **kohya-ss/sd-scripts** — no FLUX.2 (use `musubi-tuner` instead).

## Confirming evidence from other communities

- **SimpleTuner #951 / #997 / #624 / #303** — long-running thread of users asking for "actually split the model across GPUs," consistently answered with FSDP/DDP which doesn't help the single-card-too-small case.
- **OneTrainer #588** — open since 2024-11, no merged solution.
- **kohya sd-scripts #1551 / #1721** — FLUX multi-GPU works on exactly 2 GPUs but breaks at >2. Supports the thesis that pair-of-consumer-cards is the realistic target deployment.
- **HF blog: "Flux 2 Dev on RTX A6000 48GB"** — community baseline for single-card FLUX.2 is 48 GB. 24 GB is below the floor. This patch redefines the floor as 2× 24 GB.

## Watch list

- **DiffSynth-Studio** (12.4k stars). FLUX.2-dev LoRA recipe shipped. Sequence-parallel work for Wan 2.2 suggests the project is willing to take real parallelism patches; primary docs are non-English so survey was lighter.
- **fluxgym** (cocktailpeanut) — UI wrapper over kohya. Porting upstream wins both.

## Sources

- [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) · [FLUX.2 issue #743](https://github.com/kohya-ss/musubi-tuner/issues/743)
- [OneTrainer](https://github.com/Nerogar/OneTrainer) · [multi-GPU issue #588](https://github.com/Nerogar/OneTrainer/issues/588)
- [SimpleTuner](https://github.com/bghira/SimpleTuner) · [FLUX multi-GPU #997](https://github.com/bghira/SimpleTuner/discussions/997)
- [diffusers FLUX.2 README](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux2.md) · [train_dreambooth_lora_flux2.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux2.py)
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)
- [DiffSynth-Studio FLUX.2-dev LoRA recipe](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/lora/FLUX.2-dev.sh)
- [Sayak Paul gist: split Flux transformer across 2× 16 GB](https://gist.github.com/sayakpaul/a9266fe2d0d510ec44a9cdc385b3dd74)
- [HF blog: Flux 2 Dev on single A6000 48 GB](https://huggingface.co/blog/YellowjacketGames/flux2dev-on-a6000-48gb)
