# flux2-dual-gpu-lora

Train FLUX.2-dev LoRAs across any pair of 24+ GB CUDA GPUs (2× RTX 3090, 2× RTX 4090, 2× RTX 5090) with Mistral-3 kept in system RAM. Drop-in replacement for [ai-toolkit](https://github.com/ostris/ai-toolkit)'s FLUX.2 trainer, enabled via a single env var. First-known public implementation — closes the gap in ai-toolkit issue [#531](https://github.com/ostris/ai-toolkit/issues/531).

The same dual-GPU split has since been ported to four more trainers (musubi-tuner, DiffSynth-Studio, OneTrainer, HuggingFace diffusers) and to Wan 2.x video LoRA training. The training-side patch isn't FLUX.2-specific in spirit — split the transformer at a low-traffic block boundary, keep the text encoder in system RAM, quantize weights to fp8 — only the integration points differ per trainer.

## Repository layout

| Path | Contents |
|---|---|
| `flux2_model.py`, `flux2_dual_gpu.py` | The ai-toolkit patch (the flagship — full-file replacement + the `Flux2DualGPUMixin`). Drop into ai-toolkit; see [Quick start](#quick-start). Upstream: [ai-toolkit#829](https://github.com/ostris/ai-toolkit/pull/829). |
| [`inference/`](inference/) | Standalone dual-GPU FLUX.2 + LoRA rendering — `engine.py` (importable), `cli.py`, `recipe.py`. The thing ComfyUI / ai-toolkit's `run.py` can't do. See [Inference](#inference). |
| [`examples/`](examples/) | Drop-in dual-GPU helper modules + quickstarts for the other trainers: `diffsynth/` (FLUX.2 + Wan video), `diffusers/`, `musubi/`, `onetrainer/`. |
| [`patches/`](patches/) | The raw byte-surgery patcher scripts that inject the helpers + integration edits into each trainer's checkout, organized per-trainer with validation-status tags. Plus the optional ai-toolkit memory-probe diagnostic diff. See [`patches/README.md`](patches/README.md). |
| [`docs/`](docs/) | Trainer survey + per-trainer porting walkthroughs. |

## The numbers

| Setup | Step rate | 400 steps |
|---|---|---|
| Single RTX 5090, ai-toolkit default | 14.4 s/it → 277 s/it (WDDM thrash) | 90 min → 30+ hours |
| **Dual RTX 5090, ai-toolkit + this patch** (fp8 weight-only, bf16 compute) | **2.85 s/it sustained** | **~19 min** |
| **Dual RTX 5090, musubi-tuner + this patch** (fp8 weight-only, bf16 compute) | **2.22 s/it sustained** | **~15 min** |
| **Dual RTX 5090, DiffSynth-Studio + this patch** (fp8 weight-only, bf16 compute) | **2.69 s/it sustained** | **~18 min** |
| **Dual RTX 5090, OneTrainer + this patch** (fp8 W8A8 compute) | **1.75 s/it sustained** | **~11 min** |
| **Dual RTX 5090, OneTrainer + this patch** (int8 W8A8 compute) | **0.95 s/it sustained** | **~6 min** |

> **Compare within precision class, not across.** All five rows use this patch's dual-GPU split (split shape is identical — 20.7 GB cuda:0 / 12.6 GB cuda:1 — regardless of quant scheme). What differs across rows is *what* the matmul actually runs in, which determines which tensor-core path lights up:
>
> | Mode | Weight storage | Activation @ matmul | TC throughput on 5090 |
> |---|---|---|---|
> | **fp8 weight-only / bf16 compute** | fp8 (halved memory) | bf16 | ~165 TF (bf16 cores) |
> | **fp8 W8A8 compute** | fp8 (halved memory) | fp8 | ~330 TF (fp8 cores) |
> | **int8 W8A8 compute** | int8 (halved memory) | int8 | ~660 TF (int8 cores) |
>
> ai-toolkit / musubi / DiffSynth do *weight-only* quant — weights are fp8 on disk and in VRAM, but get dequantized to bf16 at every matmul (the whole point is the memory savings, not the compute). OneTrainer has its own `LinearW8A8` class (`modules/module/quantized/LinearW8A8.py`) that keeps activations in 8-bit and dispatches the matmul through `torch._scaled_mm` (fp8) or `torch._int_mm` (int8) — both engage the higher-throughput Blackwell tensor-core paths. The OneTrainer 1.75 → 0.95 jump is the int8 cores being ~2× the fp8 cores on this hardware.
>
> So the honest framing is: **the patch makes all four trainers work on dual 32 GB; the trainer authors made independent choices about whether to chase compute-precision speedups beyond that.**
>
> **Attempted apples-to-apples (2026-05-11):** retested DiffSynth with `torchao.Float8DynamicActivationFloat8WeightConfig` (fp8 W8A8 inference config) plus the LoRA-skip `filter_fn`. The cross-device forward pass **does** work — distribute lands at 22.2 GB cuda:0 / 13.9 GB cuda:1, forward succeeds in 0.93 s (vs 1.26 s for weight-only — the expected ~1.4× speedup from fp8 tensor cores), output lands back on cuda:0. But backward dies at `loss.backward()` with `RuntimeError: derivative for aten::_scaled_mm is not implemented`. `torch._scaled_mm` is the fp8 GEMM op torchao dispatches to, and PyTorch hasn't registered an autograd derivative for it — torchao's `Float8DynamicActivationFloat8WeightConfig` is an *inference* config and not training-aware. The training-aware torchao path (`torchao.float8.convert_to_float8_training`) keeps weights in bf16 in memory and casts to fp8 on the fly during the matmul — gives the same fp8 tensor-core speedup but doesn't halve weight storage, which is a non-starter for the dual-32 GB use case. OneTrainer dodges this by shipping its own `LinearW8A8` class with hand-rolled forward/backward `torch.autograd.Function` wrappers around `torch._scaled_mm` / `torch._int_mm`; transplanting that approach into the helper is plausible but a meaningful chunk of work, not a config flip.

Per-GPU footprint at runtime: ~20 GB on GPU 0, ~12 GB on GPU 1, both alternating at 99% SM utilization. No thrashing, no degradation.

The musubi-tuner port runs ~21% faster than ai-toolkit on the same hardware (different attention path, different LoRA wrapper overhead) — validated end-to-end with `--fp8_base --fp8_scaled --gradient_checkpointing` on sumi v8 (32 imgs @ 512², 10-step smoke; loss 0.526 → 0.545 monotonic across the run; both checkpoints saved).

The OneTrainer port has been validated end-to-end in **both** of its 8-bit-compute modes on the same sumi v8 dataset: `INT_W8A8` clocks 0.95 s/it (int8 tensor cores), `FLOAT_8` (= W8A8 fp8 compute, not weight-only) clocks 1.75 s/it (fp8 tensor cores), saving 402 MB and 384 MB LoRA checkpoints respectively. OneTrainer doesn't ship a "weight-only fp8" mode for the dit — its `LinearW8A8` class always quantizes activations too. So the apples-to-apples comparison with the other three (which run fp8 weight-only / bf16 compute) isn't currently possible on the OneTrainer side; what it does show is that **the dual-GPU split is independent of compute-precision choice** — the distribution shape (20.7 GB cuda:0 / 12.6 GB cuda:1) is identical across all four trainers and all three precision classes.

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

## Inference

ComfyUI has no dual-GPU FLUX.2 path either, and ai-toolkit's own `run.py` "generate" / sample-during-training paths deadlock on a dual-GPU + TE-on-CPU config (the embed-cache/unload step hangs; `set_device_state` OOMs putting Mistral on cuda:0). [`inference/`](inference/) is a small standalone engine that drives ai-toolkit's `Flux2Model` / `LoRASpecialNetwork` / `generate_images` directly — so a LoRA trained with this patch renders under the exact same split + quantization the training ran in:

```bash
# one-off CLI  (--prompts is a JSON {name: prompt} object)
<ai-toolkit-venv>/python inference/cli.py \
    --lora out/your-character/your-character.safetensors \
    --prompts prompts.json --out renders/ --strengths 0,0.6,1.0

# declarative YAML recipe
<ai-toolkit-venv>/python inference/recipe.py job.yaml
```

`inference/engine.py` exposes `load_flux2(base, lora=LoraSpec(...))` → `(model, pipeline, network)` and `render(model, pipeline, {name: prompt}, out=..., strengths=(...,))` for importing into other scripts (eval harnesses, dataset generation). Beyond the training patch it needs three small fixups — keep Mistral on CPU in `set_device_state`, no-op `transformers.PreTrainedModel.to(cuda)`, and pin `Flux2Pipeline._execution_device` to `cuda:0` (else it resolves to CPU — Mistral's device — and feeds CPU tensors to the cuda transformer). Throughput on 2× RTX 5090: ~80–140 s/image at 1024²/20 steps/cfg 4 (the per-prompt Mistral-on-CPU encode is the bottleneck, not the diffusion). See [`inference/README.md`](inference/README.md).

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

The drop-in helper plus a quickstart is at [`examples/musubi/`](examples/musubi/); a concrete porting plan — hook points, complexity comparison, validation plan, upstream strategy — is at [`docs/porting-musubi-tuner.md`](docs/porting-musubi-tuner.md); the full implementation lives on a Genno fork branch ([genno-whittlery/musubi-tuner:dual-gpu-flux2](https://github.com/genno-whittlery/musubi-tuner/tree/dual-gpu-flux2)). **Validated end-to-end on 2× RTX 5090, 2026-05-11** — 10/10 steps with `--fp8_base --fp8_scaled --gradient_checkpointing`, loss 0.526 → 0.545 monotonic, 2.22 s/it sustained, both checkpoints saved. Upstream PR to kohya-ss/musubi-tuner pending. Patch consists of: a new `musubi_tuner/flux_2/flux2_dual_gpu.py` module, plus three integration-point edits in `hv_train_network.py` (DDP-bypass, CPU loading device, `device_placement=[False]` prepare branch) and one inlined device-pin in `networks/lora.py` (bound-method indirection in `apply_to` makes instance-level shimming unreliable, so the pin runs inline at the top of `LoRAModule.forward`).

For HuggingFace **diffusers** users (the `train_dreambooth_lora_flux2*.py` reference scripts): a drop-in helper file plus an integration guide is at [`examples/diffusers/`](examples/diffusers/). The helper uses PyTorch forward pre-hooks (registered on every cuda:1 block, not just the boundary — important for FLUX.2's modulation/temb loop pattern). ~150 LOC, no patches to diffusers itself.

**Diffusers ref script status (2026-05-11):** The split distribution itself works end-to-end on 2× RTX 5090 — after our helper's `enable_flux2_dual_gpu(transformer)` runs, the transformer correctly lands 20.7 GB on cuda:0 / 12.6 GB on cuda:1 (matching ai-toolkit's validated shape). The training step is currently blocked at the LoRA forward by an upstream PEFT 0.19.1 × torchao 0.17 incompatibility: `TorchaoLoraLinear` calls into a dispatcher that doesn't bridge CPU/CUDA when the base weight is a `WeightOnlyFloat8Tensor` (torchao 0.17). Reproducer: apply `quantize_(transformer, Float8WeightOnlyConfig())` after `transformer.add_adapter(...)` then run a forward. This isn't fixable with a single-script patch — it needs an upstream PEFT (or torchao) fix to make `TorchaoLoraLinear`'s dispatcher device-aware. The same model on `--mixed_precision bf16` *without* fp8 quant doesn't OOM on a single >48 GB GPU, but doesn't fit on 2× 32 GB either (60 GB bf16 / 32 GB GPU = no 50/50 split fits).

The other validated trainers (ai-toolkit, musubi-tuner, OneTrainer) use weight-only fp8 quant paths that don't go through PEFT's torchao dispatcher, so the bug doesn't surface there.

**Update 2026-05-11**: the DiffSynth-Studio port (below) found that adding a one-line `filter_fn` to the `quantize_` call — excluding modules whose names contain `lora_A` or `lora_B` — avoids the dispatcher entirely (LoRA's own Linear submodules stay bf16, so PEFT keeps using its normal `LoraLinear` and torchao never sees them). The same fix likely unblocks the diffusers ref script — not retested.

For **OneTrainer** users (the GUI-based trainer with FLUX.2 Dev + Klein support): the drop-in helper + quickstart is at [`examples/onetrainer/`](examples/onetrainer/); the full port lives on a Genno fork branch ([genno-whittlery/OneTrainer:dual-gpu-flux2](https://github.com/genno-whittlery/OneTrainer/tree/dual-gpu-flux2)). **Validated end-to-end on 2× RTX 5090, 2026-05-11** — full 32-step epoch on sumi v8, ~1.05 it/s sustained, LoRA saved. Patch consists of: a new `modules/util/Flux2DualGpu.py` module, plus two integration-point edits in `modules/model/Flux2Model.py` (forced TE-on-CPU + `transformer_to` routes to distribute). One companion change to the third-party `mgds` data-loader on branch [genno-whittlery/mgds:te-device-fix](https://github.com/genno-whittlery/mgds/tree/te-device-fix): `EncodeMistralText.get_item` moves `tokens` and `attention_mask` to `text_encoder.device` before the forward call, so CPU-hosted Mistral receives CPU-side inputs instead of cuda:0 ones. (Upstream PR to `Nerogar/OneTrainer` was filed as [#1450](https://github.com/Nerogar/OneTrainer/pull/1450) and closed by a maintainer who flagged a stale issue reference in the PR body; the technical content was not reviewed. The branch is still the canonical place to consume this patch — apply it locally to a `Nerogar/OneTrainer` checkout, or use the fork directly.) Setup notes:

- **Base model:** OneTrainer requires the diffusers-folder format (`black-forest-labs/FLUX.2-dev` from HF), not the single-file `.safetensors`. ~108 GB download (60 GB transformer + 45 GB Mistral + scheduler/VAE/configs).
- **Per-trainer venv recommended.** `C:\OneTrainer\.venv` with Python 3.12, OneTrainer's pinned torch (`2.9.1+cu128`) — Blackwell-compatible despite cu128 (PTX forward-compat carries fp8 + bf16 fine).
- **Latent cache is one-time:** ~10 minutes for 32 images at 512² on CPU VAE; subsequent runs reuse the cache (`workspace_dir`-keyed) only if you don't change the dataset/config.

For **DiffSynth-Studio** users (modelscope's 12.4k-star trainer with a FLUX.2-dev LoRA recipe): a drop-in helper plus an integration guide is at [`examples/diffsynth/`](examples/diffsynth/), with a detailed porting walkthrough at [`docs/porting-diffsynth-studio.md`](docs/porting-diffsynth-studio.md). **Validated end-to-end on 2× RTX 5090, 2026-05-11** — full `sft:data_process` → `sft:train` flow runs through, 15/15 training steps at 2.69 s/it sustained, LoRA checkpoint saved (270 MB, rank 32, 7 target-module families). Distribution numbers match the diffusers/OneTrainer/musubi shape exactly: 20.7 GB cuda:0 / 12.6 GB cuda:1 after fp8 weight-only quant. Key finding: the PEFT × torchao incompatibility that blocked the diffusers reference script is **avoidable** with a one-line `filter_fn` that excludes LoRA's lora_A/lora_B Linear submodules from the quant pass (the diffusers ref script was quantizing them too, which strips `requires_grad` and breaks backward). Patch consists of: a new `examples/flux2/model_training/flux2_dual_gpu_diffsynth.py` helper, plus four integration-point edits — two in `examples/flux2/model_training/train.py` (CPU-load + fp8-quant + distribute), one in `diffsynth/diffusion/runner.py` (skip `model.to()` + `device_placement=[False, True, True, True]` to `accelerator.prepare` for both training and data_process tasks), and one in `diffsynth/models/flux2_text_encoder.py` (transformers 5.8 compat — kwargs-only `super().forward` call, re-injects `output_hidden_states` through TransformersKwargs). Upstream PR is [modelscope/DiffSynth-Studio#1434](https://github.com/modelscope/DiffSynth-Studio/pull/1434).

For **Wan 2.x video LoRA training** in DiffSynth-Studio (Wan 2.2-I2V-A14B, T2V-A14B, S2V-14B, ...): same dual-GPU helper shape applied to `WanModel.blocks`. Validated via synthetic smoke test on 2× RTX 5090 — forward + backward complete across the cross-device split, LoRA gradients land on both cuda:0 and cuda:1. Real-training end-to-end validation is gated on an unrelated upstream patchify bug (issue [modelscope/DiffSynth-Studio#1063](https://github.com/modelscope/DiffSynth-Studio/issues/1063) — `WanModel.patchify` returns the wrong shape + arity, breaking *any* Wan training forward call). Both fixes filed as separate Genno PRs: [modelscope/DiffSynth-Studio#1435](https://github.com/modelscope/DiffSynth-Studio/pull/1435) (patchify fix, single-commit, fixes #1063) and [modelscope/DiffSynth-Studio#1436](https://github.com/modelscope/DiffSynth-Studio/pull/1436) (dual-GPU helper, depends on #1435). The 14B Wan variants are particularly good dual-GPU candidates: they fit a single 32 GB card on weights alone but routinely OOM on activations at 480×832×49 frames + gradient checkpointing — the split gives the per-side activation budget headroom that single-GPU users can't otherwise reach.

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
