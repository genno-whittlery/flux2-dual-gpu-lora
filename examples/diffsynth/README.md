# Dual-GPU FLUX.2 LoRA training in DiffSynth-Studio

Drop-in helper for [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)'s FLUX.2 LoRA training scripts (`examples/flux2/model_training/`). Splits `Flux2DiT` across two CUDA devices so the model fits on pairs of 24 GB consumer GPUs (2× RTX 3090, 2× RTX 4090, 2× RTX 5090).

## Why this port is small

- `Flux2DiT` (in `diffsynth/models/flux2_dit.py`) is structurally identical to the diffusers `Flux2Transformer2DModel`. Same field names, same forward shape.
- DiffSynth uses **PEFT** for LoRA injection (`from peft import LoraConfig, inject_adapter_in_model` in `diffsynth/diffusion/training_module.py`). PEFT places LoRA params on the base layer's device automatically, so per-layer routing is free.
- The forward isn't overridden — pre-hooks bridge devices at the split point and at `norm_out` (the boundary back to cuda:0).

Total surface: one ~130-line helper file. No patches to DiffSynth itself required for the helper-based path.

## Usage

### 1. Copy the helper

```bash
curl -O https://raw.githubusercontent.com/genno-whittlery/flux2-dual-gpu-lora/main/examples/diffsynth/flux2_dual_gpu_diffsynth.py
```

### 2. Add 2 lines to your training script

In `examples/flux2/model_training/train.py` (or your derivative), after the `Flux2ImageTrainingModule` is built and LoRA injection has happened (which is inside `__init__` via `switch_pipe_to_training_mode`):

```python
from flux2_dual_gpu_diffsynth import enable_flux2_dual_gpu
enable_flux2_dual_gpu(training_module.pipe.dit)
```

If you use `accelerator.prepare(training_module, ...)`, pass `device_placement=False` for the training module so Accelerate doesn't flatten the distributed transformer back onto a single device:

```python
training_module = accelerator.prepare(
    training_module,
    device_placement=False,  # already distributed by enable_flux2_dual_gpu
)
```

### 3. Launch

```bash
export FLUX2_DUAL_GPU=true
# Optional: override the split index (default is num_single_blocks // 2)
# export FLUX2_DUAL_GPU_SPLIT_AT=24

accelerate launch --num_processes=1 examples/flux2/model_training/train.py \
    --model_paths "..." \
    --trainable_models "dit" \
    --lora_base_model "dit" \
    --lora_rank 16 \
    --task "sft" \
    ...
```

**Critical:** `--num_processes=1`. This is *model* parallelism (both GPUs cooperate on a single training step), not data parallelism (which would spawn one process per GPU). With `--num_processes=2` Accelerate would try to give each process its own GPU and the model wouldn't be split.

### 4. Verify the split landed

Add a print right after `enable_flux2_dual_gpu`:

```python
import torch
print(f"dit split_at = {getattr(training_module.pipe.dit, '_flux2_dual_gpu_split_at', 'NOT SPLIT')}")
for gpu in range(torch.cuda.device_count()):
    used_gb = torch.cuda.memory_allocated(gpu) / 1024**3
    print(f"  GPU{gpu}: {used_gb:.2f} GB allocated")
```

Expected on 2× RTX 5090, FLUX.2-dev fp8, rank 16:

```
dit split_at = 24
  GPU0: ~21 GB allocated
  GPU1: ~12 GB allocated
```

If both GPUs aren't being used, either `FLUX2_DUAL_GPU` is unset, `device_placement=False` was missed, or `enable_flux2_dual_gpu` was called before LoRA injection (call it after).

## Validation status — ✅ validated end-to-end (2× RTX 5090, 2026-05-11)

Full `sft:data_process` → `sft:train` flow runs through, 15/15 training steps at **2.69 s/it sustained**, LoRA checkpoint saved (270 MB, rank 32, 7 target-module families). Distribution matches the ai-toolkit / OneTrainer / musubi shape exactly: 20.7 GB cuda:0 / 12.6 GB cuda:1 after fp8 weight-only quant.

Key finding: the PEFT × torchao incompatibility that blocks the diffusers reference script (`TorchaoLoraLinear`'s dispatcher not bridging CPU/CUDA for `WeightOnlyFloat8Tensor` base weights) is **avoidable** here with a one-line `filter_fn` on the `quantize_` call that excludes modules whose names contain `lora_A` / `lora_B` — LoRA's own Linear submodules stay bf16, so PEFT keeps using its normal `LoraLinear` and torchao never sees them.

Beyond the drop-in helper, this port also needs four small integration edits — two in `examples/flux2/model_training/train.py` (CPU-load + fp8-quant + distribute), one in `diffsynth/diffusion/runner.py` (skip `model.to()` + `device_placement=[False, True, True, True]` to `accelerator.prepare`, for both the training and data_process tasks), one in `diffsynth/models/flux2_text_encoder.py` (transformers 5.8 compat — kwargs-only `super().forward`, re-injecting `output_hidden_states` through `TransformersKwargs`). The exact byte-surgery lives in [`patches/diffsynth/`](../../patches/diffsynth/). Detailed walkthrough: [`docs/porting-diffsynth-studio.md`](../../docs/porting-diffsynth-studio.md). Upstream PR: [modelscope/DiffSynth-Studio#1434](https://github.com/modelscope/DiffSynth-Studio/pull/1434).

## DiffSynth-specific considerations

- **DiffSynth's training pipeline supports FSDP**. The dual-GPU split here is *not* FSDP — it's a static model-parallel split across exactly two GPUs. Disable FSDP in your accelerate config (`fsdp: false` or no FSDP block) when using this helper.
- **DiffSynth's `pipe.dit` is loaded via `Flux2ImagePipeline.from_pretrained`** with the `device` argument. Pass `device="cpu"` to that call if you can — the helper will redistribute it across cuda:0/cuda:1. If you pass `device="cuda"`, the helper still works (it overrides placement after the fact), but you'll spike memory on cuda:0 briefly during load before the split lands.
- **`offload_models` / `fp8_models` in the training config** are orthogonal to this helper. Use `fp8_models=["dit"]` if you want fp8 weights (recommended on 24 GB cards). `offload_models` is for moving inactive models (vae / text encoder) to CPU — also orthogonal, and works with the split.

## Upstream PR

[modelscope/DiffSynth-Studio#1434](https://github.com/modelscope/DiffSynth-Studio/pull/1434).

---

# Bonus: dual-GPU Wan video LoRA training (`wan_dual_gpu_diffsynth.py`)

Same helper shape, applied to DiffSynth-Studio's `examples/wanvideo/model_training/`. Splits `WanModel` across two CUDA devices at the `blocks` midpoint — useful for the 14B Wan 2.x variants (I2V-A14B, T2V-A14B, S2V-14B), which fit a single 32 GB card on fp8-quantized *weights* alone but routinely OOM on *activations* at 480×832×49 frames + gradient checkpointing. The split gives the per-side activation headroom single-GPU users can't reach. Wan has one block type (`DiTBlock`) instead of FLUX.2's double + single split, so the helper only registers per-block hooks across one boundary.

Env vars: `WAN_DUAL_GPU=true` (enable), `WAN_DUAL_GPU_SPLIT_AT=15` (override; default `num_blocks // 2`).

**Status — 🚧 synthetic smoke validated, real-training end-to-end blocked upstream.** The cross-device split works: forward + backward complete across the boundary, LoRA gradients land on both cuda:0 and cuda:1 (smoke harness in [`patches/wan-smoke/`](../../patches/wan-smoke/)). Real-training validation is gated on an *unrelated* upstream patchify bug — `WanModel.patchify` returns the wrong shape + arity, breaking *any* Wan training forward call (issue [modelscope/DiffSynth-Studio#1063](https://github.com/modelscope/DiffSynth-Studio/issues/1063)). Both fixes filed as separate Genno PRs: [#1435](https://github.com/modelscope/DiffSynth-Studio/pull/1435) (patchify fix, single commit, fixes #1063) and [#1436](https://github.com/modelscope/DiffSynth-Studio/pull/1436) (dual-GPU helper, depends on #1435).
