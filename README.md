# flux2-dual-gpu-lora

Train FLUX.2-dev LoRAs across any pair of 24+ GB CUDA GPUs (2× RTX 3090, 2× RTX 4090, 2× RTX 5090) with Mistral-3 kept in system RAM. Drop-in replacement for [ai-toolkit](https://github.com/ostris/ai-toolkit)'s FLUX.2 trainer, enabled via a single env var. First-known public implementation — closes the gap in ai-toolkit issue [#531](https://github.com/ostris/ai-toolkit/issues/531).

## The numbers

| Setup | Step rate | 400 steps |
|---|---|---|
| Single RTX 5090, ai-toolkit default | 14.4 s/it → 277 s/it (WDDM thrash) | 90 min → 30+ hours |
| **Dual RTX 5090, this patch** | **2.85 s/it sustained** | **~19 min** |

Per-GPU footprint at runtime: ~21 GB on GPU 0, ~12 GB on GPU 1, both alternating at 99% SM utilization (165–445 W swings). No thrashing, no degradation.

## Why this exists

FLUX.2-dev's transformer is ~30 GB at fp8 weight-only quantization; Mistral-3's text encoder is another ~24 GB. They don't co-reside on a 32 GB consumer GPU. ai-toolkit's default path keeps both on GPU and relies on Windows WDDM unified memory to silently page them through system RAM — works at low fragmentation, thrashes hard after a few hundred steps.

This patch does three things in stack:

1. **Mistral on CPU** — system RAM is cheap; Mistral is frozen during LoRA training and only needs to embed each prompt once at training start. After caching, it unloads entirely.
2. **Transformer split across two GPUs** — pipeline-parallel at the single_blocks midpoint, ~16 GB transformer per side, single ~18 MB activation crossing PCIe per forward.
3. **fp8 weight-only quantization** — Black Forest Labs' production deployment format (Float8WeightOnlyConfig via torchao). Activations and gradients stay bf16; only weights are quantized. This is meaningfully different from int8 weight-only, which hurts LoRA gradient quality.

## Who this helps

- **2× RTX 4090 / 2× RTX 3090 owners (expected — no device to test)** — each card carries 24 GB; with the transformer split ~16 GB per card the model fits where it couldn't on a single 24 GB card *at all*. On a single 4090 or 3090, FLUX.2 LoRA training doesn't run — the transformer overflows even with WDDM paging. The dual-GPU split should move it from impossible → buildable. This is the biggest reach of this patch: 24 GB cards are easy to get (3090s on the used market, 4090s widely available), 5090s are not. Same env-var path; `FLUX2_DUAL_GPU_SPLIT_AT` can be tuned if the default midpoint lands unevenly. Reports welcome.
- **2× RTX 5090 owners (validated end-to-end)** — moves training from 14.4 → 277 s/it (WDDM thrash on single 5090) to 2.85 s/it sustained. The reference setup for this patch's development.
- **Mixed setups (e.g., 5090 + 4090, 4090 + 3090)** — also expected to work, with `FLUX2_DUAL_GPU_SPLIT_AT` biased so the smaller card carries fewer blocks. Untested.

## Quick start

### 1. Requirements

- Two CUDA GPUs, ≥24 GB each (validated on 2× RTX 5090 / sm_120; 2× RTX 4090 and 2× RTX 3090 should also work but have not been tested)
- Host RAM: ≥128 GB recommended (Mistral bf16 in CPU memory needs ~50 GB)
- ai-toolkit checkout (validated against FLUX.2 branch as of 2026-05-10)
- PyTorch ≥2.4 with CUDA 12.x or 13.x

### 2. Install

Drop `flux2_model.py` into your ai-toolkit checkout, replacing the stock file:

```bash
cp flux2_model.py /path/to/ai-toolkit/extensions_built_in/diffusion_models/flux2/flux2_model.py
```

The patch is gated by `FLUX2_DUAL_GPU=true` — default behavior unchanged if the env var isn't set.

### 3. Configure

In your training yaml, ensure:

```yaml
config:
  process:
    - device: cuda:0
      datasets:
        - cache_text_embeddings: true   # one-shot CPU embedding cache
      train:
        unload_text_encoder: true       # unload Mistral after caching
      model:
        arch: flux2
        quantize: true                  # fp8 transformer
        quantize_te: false              # Mistral stays bf16 on CPU
```

### 4. Launch

```bash
export FLUX2_DUAL_GPU=true
export FLUX2_TE_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=<your_huggingface_token>

python run.py config/your-character.yaml
```

Expected console during model load:

```
Distributing transformer across cuda:0 and cuda:1
Single-block split: 24 on cuda:0, 24 on cuda:1
Keeping Mistral on CPU for quantization
```

Then training proceeds at ~2.8 s/it.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `FLUX2_DUAL_GPU` | `false` | Set to `true` to enable the dual-GPU path |
| `FLUX2_TE_DEVICE` | (unset, = `device_torch`) | Set to `cpu` to keep Mistral in system RAM (independently useful without `FLUX2_DUAL_GPU`) |
| `FLUX2_DUAL_GPU_SPLIT_AT` | `n_single_blocks // 2` (= 24) | Override the single_blocks split index |
| `FLUX2_MISTRAL_PATH` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Local path to Mistral checkpoint (avoids HF download) |
| `FLUX2_MEMORY_PROBE_STEPS` | (unset) | Optional diagnostic. Comma-separated relative step counts; emits `torch.cuda.memory_summary()` at each. Requires the companion patch in `patches/`. |

## How it works

The patch applies six surgical changes against ai-toolkit's `Flux2Model`:

1. **Module distribution** — after `quantize_model` returns, `Flux2.img_in / time_in / txt_in / pe_embedder / modulation modules / all 8 DoubleStreamBlocks / first 24 SingleStreamBlocks` go to cuda:0; the remaining 24 SingleStreamBlocks + `final_layer` go to cuda:1.

2. **Forward override** — `Flux2.forward` is replaced with a method that inserts one `.to(cuda:1)` boundary mid-`single_blocks` (smallest cross-PCIe payload: ~18 MB at 512²) and moves output back to cuda:0 at the end for downstream loss / scatter ops.

3. **`transformer.to()` override** — strips device arguments so `set_device_state` and ~four other ai-toolkit sites can't collapse the split back to a single device. Dtype changes still pass through.

4. **`text_encoder_to` override** — `SDTrainer.hook_before_train_loop` hardcodes a move of Mistral to `device_torch`. Override `Flux2Model.text_encoder_to` to route through `te_device_torch` instead.

5. **LoRA per-layer device routing** — patches `ToolkitNetworkMixin.force_to` and `LoRANetwork.apply_to` to walk LoRA modules and move each to its parent layer's device. Each LoRA's `forward` is wrapped to lazily re-pin parameters on device mismatch — safety net against any future re-collection by ai-toolkit.

6. **`broadcast_and_multiply` device alignment** — auto-moves the network-level `torch_multiplier` to match LoRA output device when the multiply happens.

Per-step cross-PCIe traffic: ~36 MB (one forward + one backward × 18 MB activation). At PCIe Gen5 x16 (~50 GB/s practical), this is ~1 ms — negligible vs the ~2.8 s compute step.

## Examples

`examples/inference_with_mistral_on_cpu.py` — minimal ComfyUI-API script showing the matching inference pattern (CLIPLoader with `device="cpu"`) so character LoRAs trained with this patch can run cleanly on a 24 GB inference GPU too.

## Limitations

- **fp8 weight-only training, not pure bf16.** Pure-bf16 would need a third 32 GB GPU or aggressive layer offloading. Quality is comparable to BFL's production inference; the int8 quantization quality concerns don't apply to fp8.
- **Single-batch pipeline-parallel.** Only one micro-batch in flight at a time; one GPU computes while the other waits. Multi-microbatch pipelining (gpipe / 1F1B-style) would push utilization higher — open for v2.
- **ai-toolkit-version-coupled.** The monkey-patches reference specific class names (`ToolkitNetworkMixin`, `LoRANetwork`) and function paths (`network_mixins.broadcast_and_multiply`). If ai-toolkit refactors these, the patches need updating.
- **Validated for FLUX.2-dev specifically.** FLUX.2-klein and FLUX.2-pro variants likely work given the same architecture but haven't been exercised.
- **2× RTX 4090 / 2× RTX 3090 / mixed setups untested.** Architecturally the split applies (each card carries ~16 GB of transformer weights, well under 24 GB) but I don't have the rack to validate. Anyone running this on 2× 4090 or 2× 3090 — please open an issue with step rate + GPU memory snapshot.

## Diagnostic: per-step memory probe

The companion patch in `patches/base_sd_train_process_memory_probe.patch` adds a `torch.cuda.memory_summary()` callback at user-specified relative training steps via `FLUX2_MEMORY_PROBE_STEPS=1,50,100,500`. Useful for confirming on your own workload whether the single-GPU thrashing is a working-set leak (allocated growing) or a per-step over-capacity transient (peak > VRAM at every step).

For Suzurin training on a single RTX 5090, the step-1 probe was unambiguous:

```
Allocated  cur=30.63 GB  peak=32.94 GB
Reserved   cur=31.14 GB  peak=34.05 GB
Active allocations: 2747
```

Peak allocated (32.94 GB) exceeds GPU capacity (31.84 GB). The bug isn't a leak — it's an intrinsic per-step forward transient that overshoots VRAM by ~1 GB. The dual-GPU path sidesteps it by keeping per-card peak well under capacity (cuda:0 ≈ 21 GB, cuda:1 ≈ 12 GB).

## Cross-PCIe traffic math

At hidden_size=6144, bf16, seq_len 1536 (512² + ~512 text tokens), batch 1:

- Per single_block boundary crossing: 1 × 1536 × 6144 × 2 = **~18 MB**
- One crossing forward + one backward per step = **~36 MB / step**
- PCIe Gen5 x16 practical: ~50 GB/s → 36 MB ≈ 1 ms of PCIe time per step

Boundary placement is mid-`single_blocks` rather than at the double→single transition because at that transition `img` and `txt` are still separate (~36 MB combined before cat) and modulation tensors are larger.

## Porting to other trainers

The patch lives in ai-toolkit today. A survey of other FLUX.2 LoRA trainers (musubi-tuner, OneTrainer, SimpleTuner, diffusers, diffusion-pipe, DiffSynth-Studio) and a recommended next port target lives at [`docs/trainer-survey.md`](docs/trainer-survey.md). Short version: **musubi-tuner** is the highest-leverage next target; **diffusers** `train_dreambooth_lora_flux2.py` is the easiest mechanical port.

## Contributing

PRs welcome, especially:

- Multi-microbatch pipelining
- Validation on FLUX.2-klein variants
- Three-GPU split (would enable pure bf16 training)
- Refactor to `Flux2DualGPUMixin` in advance of upstreaming to ai-toolkit

## Acknowledgments

- [ai-toolkit](https://github.com/ostris/ai-toolkit) by Ostris, LLC — the trainer this patches
- [Black Forest Labs](https://huggingface.co/black-forest-labs) — FLUX.2-dev
- [optimum.quanto](https://github.com/huggingface/optimum-quanto) + [torchao](https://github.com/pytorch/ao) — fp8 weight-only quantization

## License

MIT. See [LICENSE](./LICENSE).

---

Published by **The Whittlery** — the workshop face of a small brand family. The Whittlery's resident character is **Genno** (玄能, wooden mallet), the patient striker.
